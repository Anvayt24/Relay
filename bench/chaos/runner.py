"""Run one chaos scenario against one client, once: start fresh processes, drive load, measure.

Process layout (nothing shares an event loop with the load generator, so the generator's own
CPU can't make Relay look slow):

    load generator ──▶ Relay (uvicorn subprocess, in-memory state)
                  └──▶ primary simulator (subprocess, :P1) ◀── Relay
                  └──▶ fallback simulator (subprocess, :P2) ◀── Relay

The simulators are started once per scenario and reset between runs. Relay is restarted for
every run, so no breaker, pause or budget state leaks from one run into the next.

Arrivals are **open-loop Poisson**: requests arrive at a fixed rate whether or not earlier
ones have finished. A closed loop (N workers, each waiting for its reply) slows down exactly
when the system under test slows down, which hides the very overload a chaos test exists to
show (coordinated omission).
"""

import asyncio
import os
import random
import socket
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
import yaml

from bench.chaos import clients
from bench.chaos.metrics import (
    Result,
    burstiness,
    retry_amplification,
    summarize,
    wrong_rate,
)

ROOT = Path(__file__).resolve().parents[2]
SCENARIOS = Path(__file__).parent / "scenarios"
PRIMARY_MODEL, FALLBACK_MODEL = "sim-primary", "sim-fallback"
CLIENT_NAMES = ("relay", "naive_retry", "direct")


@dataclass(frozen=True)
class Scenario:
    name: str
    description: str
    duration_s: float
    rps: float
    stream_ratio: float
    relay_tenant: str
    incident_window_s: tuple[float, float] | None
    measure_truth: bool
    primary: dict[str, Any]
    fallback: dict[str, Any]
    scale: float  # smoke runs compress time by this factor

    @property
    def expected_requests(self) -> int:
        return int(self.rps * self.duration_s)


def load(name: str, *, smoke: bool = False) -> Scenario:
    spec = yaml.safe_load((SCENARIOS / f"{name}.yaml").read_text(encoding="utf-8"))
    full = float(spec["duration_s"])
    duration = float(spec["smoke_duration_s"]) if smoke else full
    window = spec.get("incident_window_s")
    scale = duration / full
    return Scenario(
        name=name,
        description=" ".join(str(spec["description"]).split()),
        duration_s=duration,
        rps=float(spec["rps"]),
        stream_ratio=float(spec["stream_ratio"]),
        relay_tenant=spec["relay_tenant"],
        incident_window_s=(window[0] * scale, window[1] * scale) if window else None,
        measure_truth=bool(spec.get("measure_truth", False)),
        primary=spec.get("primary", {}),
        fallback=spec.get("fallback", {}),
        scale=scale,
    )


# ---------------------------------------------------------------- scenario -> simulator YAML
def sim_scenario(scn: Scenario, side: str) -> dict[str, Any]:
    """The simulator scenario dict for ``side`` ("primary" | "fallback"): the shared base
    profile plus this scenario's overrides, with times compressed for smoke runs."""
    base = yaml.safe_load((SCENARIOS / "base.yaml").read_text(encoding="utf-8"))
    over = scn.primary if side == "primary" else scn.fallback
    model = PRIMARY_MODEL if side == "primary" else FALLBACK_MODEL
    quality = over.get("quality", base[side]["quality"])
    if isinstance(quality, dict) and "shift_to" in quality:
        at = int(quality["shift_at_fraction"] * scn.expected_requests)
        quality = {"base": quality["base"], "shifts": [[at, quality["shift_to"]]]}
    elif not isinstance(quality, dict):
        quality = {"base": quality}
    profile: dict[str, Any] = {
        "latency": base["models"]["latency"],
        "length": base["models"]["length"],
        "quality": quality,
    }
    if "errors" in over:
        profile["errors"] = over["errors"]
    incidents = [
        {**i, "start_s": i["start_s"] * scn.scale, "end_s": i["end_s"] * scn.scale}
        for i in over.get("incidents", [])
    ]
    return {"models": {model: profile}, "incidents": incidents}


def relay_config(scn: Scenario, p_port: int, f_port: int, edge_state: str) -> dict[str, Any]:
    """Relay's config for a run: two openai-kind providers pointing at the simulators."""
    price = {"price_in": 1.0, "price_out": 2.0}
    graph = {"sim/primary": [{"to": "sim/fallback", "seed_state": edge_state}]}
    tenant = {
        "limits": {"rpm": 1_000_000, "tpm": 1_000_000_000, "concurrency": 100_000},
        "budget_usd_daily": 1_000_000,
        "budget_mode": "soft",
        "allowed_providers": ["primary", "backup"],
        "deadlines": {"first_token_s": 1.0, "interactive_s": 15, "batch_s": 30},
    }
    return {
        "price_table_version": "chaos-1",
        "providers": {
            "primary": {
                "kind": "openai",
                "base_url": f"http://127.0.0.1:{p_port}/v1",
                "api_key_env": "CHAOS_KEY",
                "limit_scope": "model",
            },
            "backup": {
                "kind": "openai",
                "base_url": f"http://127.0.0.1:{f_port}/v1",
                "api_key_env": "CHAOS_KEY",
                "limit_scope": "model",
            },
        },
        "models": {
            "sim/primary": {
                "provider": "primary",
                "provider_model": PRIMARY_MODEL,
                "tier": "strong",
                **price,
            },
            "sim/fallback": {
                "provider": "backup",
                "provider_model": FALLBACK_MODEL,
                "tier": "cheap",
                **price,
            },
        },
        "routes": {
            "main": {
                "aliases": {"relay/auto": "sim/primary", "relay/strong": "sim/primary"},
                "fallback_graph": graph,
            }
        },
        "tenants": {
            "chaos": {**tenant, "route": "main", "fallback_policy": "strict"},
            "chaos_cold": {**tenant, "route": "main", "fallback_policy": "strict"},
        },
    }


# ---------------------------------------------------------------- processes
def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _spawn(args: list[str], env: dict[str, str], cwd: Path) -> subprocess.Popen[bytes]:
    return subprocess.Popen(
        [sys.executable, *args],
        env=env,
        cwd=cwd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


async def _wait_ready(url: str, proc: subprocess.Popen[bytes], timeout_s: float = 30) -> None:
    deadline = time.monotonic() + timeout_s
    async with httpx.AsyncClient(timeout=2) as c:
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                raise RuntimeError(f"process for {url} exited with {proc.returncode}")
            try:
                if (await c.get(url)).status_code < 500:
                    return
            except httpx.HTTPError:
                pass
            await asyncio.sleep(0.2)
    raise TimeoutError(f"{url} not ready after {timeout_s}s")


def _stop(proc: subprocess.Popen[bytes]) -> None:
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()


@contextmanager
def workdir() -> Any:
    with tempfile.TemporaryDirectory(prefix="relay-chaos-") as d:
        yield Path(d)


class Harness:
    """Two simulators that live for a whole scenario, and Relay instances that don't."""

    def __init__(self, scn: Scenario, tmp: Path) -> None:
        self.scn, self.tmp = scn, tmp
        self.p_port, self.f_port = free_port(), free_port()
        self.p_url = f"http://127.0.0.1:{self.p_port}"
        self.f_url = f"http://127.0.0.1:{self.f_port}"
        self._procs: list[subprocess.Popen[bytes]] = []
        self._env = {
            **os.environ,
            "PYTHONPATH": str(ROOT),
            "CHAOS_KEY": "chaos-not-a-secret",
            # Never read the developer's .env database or Redis settings.
            "RELAY_DATABASE_URL": "",
            "RELAY_REDIS_URL": "",
        }

    async def start_sims(self) -> None:
        for side, port, url in (
            ("primary", self.p_port, self.p_url),
            ("fallback", self.f_port, self.f_url),
        ):
            path = self.tmp / f"sim_{side}.yaml"
            path.write_text(yaml.safe_dump(sim_scenario(self.scn, side)), encoding="utf-8")
            proc = _spawn(
                ["-m", "relay.simulator", "--scenario-file", str(path), "--port", str(port),
                 "--control", "--seed", "7"],
                self._env,
                self.tmp,
            )  # fmt: skip
            self._procs.append(proc)
            await _wait_ready(f"{url}/v1/models", proc)

    async def reset_sims(self) -> None:
        async with httpx.AsyncClient(timeout=5) as c:
            for url in (self.p_url, self.f_url):
                await c.post(f"{url}/sim/reset")

    async def start_relay(self) -> tuple[subprocess.Popen[bytes], str]:
        edge = "undecided" if self.scn.relay_tenant == "chaos_cold" else "safe"
        cfg = self.tmp / "relay.chaos.yaml"
        cfg.write_text(
            yaml.safe_dump(relay_config(self.scn, self.p_port, self.f_port, edge)),
            encoding="utf-8",
        )
        port = free_port()
        env = {
            **self._env,
            "RELAY_ENV": "dev",
            "RELAY_AUTH_DISABLED": "true",
            "RELAY_AUTH_DISABLED_TENANT": self.scn.relay_tenant,
            "RELAY_CONFIG_PATH": str(cfg),
            "RELAY_LOG_LEVEL": "WARNING",
        }
        proc = _spawn(
            ["-m", "uvicorn", "relay.api.app:create_app", "--factory", "--port", str(port),
             "--log-level", "warning"],
            env,
            self.tmp,
        )  # fmt: skip
        self._procs.append(proc)
        url = f"http://127.0.0.1:{port}"
        await _wait_ready(f"{url}/healthz", proc)
        return proc, url

    def stop(self) -> None:
        for p in self._procs:
            _stop(p)
        self._procs.clear()


# ---------------------------------------------------------------- one run
async def drive(
    scn: Scenario, make_request: Callable[[bool, float], Any], rng: random.Random
) -> list[Result]:
    """Open-loop Poisson arrivals for ``scn.duration_s``, then wait for stragglers."""
    t0 = time.monotonic()
    tasks: list[asyncio.Task[Result]] = []
    next_at = 0.0
    while next_at < scn.duration_s:
        await asyncio.sleep(max(0.0, t0 + next_at - time.monotonic()))
        stream = rng.random() < scn.stream_ratio
        tasks.append(asyncio.create_task(make_request(stream, t0)))
        next_at += rng.expovariate(scn.rps)
    return list(await asyncio.gather(*tasks))


async def run_once(scn: Scenario, client_name: str, harness: Harness, seed: int) -> dict[str, Any]:
    await harness.reset_sims()
    relay_proc: subprocess.Popen[bytes] | None = None
    relay_url = ""
    if client_name == "relay":
        relay_proc, relay_url = await harness.start_relay()
    rng = random.Random(seed)
    async with httpx.AsyncClient(
        timeout=httpx.Timeout(30), limits=httpx.Limits(max_connections=2000)
    ) as http:

        def make(stream: bool, t0: float) -> Any:
            if client_name == "relay":
                return clients.relay_client(http, relay_url, stream, t0)
            if client_name == "direct":
                return clients.direct_client(http, harness.p_url, PRIMARY_MODEL, stream, t0)
            return clients.naive_retry_client(
                http, harness.p_url, harness.f_url, PRIMARY_MODEL, FALLBACK_MODEL, stream, t0
            )

        results = await drive(scn, make, rng)
        stats = {
            side: (await http.get(f"{url}/sim/stats")).json()
            for side, url in (("primary", harness.p_url), ("fallback", harness.f_url))
        }
        wrong = await _wrong_answers(http, harness, results) if scn.measure_truth else {}
    if relay_proc is not None:
        _stop(relay_proc)
        harness._procs.remove(relay_proc)

    summary = summarize(results)
    upstream = stats["primary"]["calls"] + stats["fallback"]["calls"]
    out: dict[str, Any] = {
        **summary,
        "retry_amplification": retry_amplification(upstream, len(results)),
        "primary_calls": stats["primary"]["calls"],
        "fallback_calls": stats["fallback"]["calls"],
        **wrong,
    }
    if scn.incident_window_s:
        lo, hi = scn.incident_window_s
        out["primary_burstiness"] = burstiness(
            {int(k): v for k, v in stats["primary"]["arrivals"].items()},
            start_bucket=int(lo * 10),
            end_bucket=int(hi * 10),
            offered_per_bucket=scn.rps * 0.1,  # 100 ms buckets
        )
        window = [r for r in results if lo <= r.started_s < hi]
        s = summarize(window)
        out["incident_success_rate"] = s["success_rate"]
    return out


async def _wrong_answers(
    http: httpx.AsyncClient, harness: Harness, results: list[Result]
) -> dict[str, float]:
    """Ground truth for every served answer: was the (hidden) correctness flag set?"""
    first: list[bool] = []
    second: list[bool] = []
    half = harness.scn.duration_s / 2
    for r in results:
        if not r.ok or r.index is None or r.served_by is None:
            continue
        primary = r.served_by == "primary"
        url, model = (harness.p_url, PRIMARY_MODEL) if primary else (harness.f_url, FALLBACK_MODEL)
        truth = (
            await http.get(f"{url}/sim/truth", params={"model": model, "index": r.index})
        ).json()
        (first if r.started_s < half else second).append(bool(truth["correct"]))
    return {
        "wrong_rate_first_half": wrong_rate(first),
        "wrong_rate_second_half": wrong_rate(second),
    }


async def run_scenario(
    name: str, clients_to_run: tuple[str, ...], runs: int, *, smoke: bool = False
) -> dict[str, Any]:
    scn = load(name, smoke=smoke)
    per_client: dict[str, list[dict[str, Any]]] = {c: [] for c in clients_to_run}
    with workdir() as tmp:
        harness = Harness(scn, tmp)
        try:
            await harness.start_sims()
            for client_name in clients_to_run:
                for i in range(runs):
                    per_client[client_name].append(await run_once(scn, client_name, harness, i))
        finally:
            harness.stop()
    return {
        "scenario": name,
        "description": scn.description,
        "smoke": smoke,
        "duration_s": scn.duration_s,
        "rps": scn.rps,
        "runs": runs,
        "clients": per_client,
    }
