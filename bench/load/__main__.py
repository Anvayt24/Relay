"""``python -m bench.load``: M1(a), the gateway's added latency, for 1 and 2 replicas.

For each replica count it brings the load stack up, then runs k6 twice at the same constant
arrival rate: once straight at the simulator, once through nginx and Relay. The overhead is
the difference at p50/p95/p99. Anything else in the path (nginx, Redis) is part of what
"through Relay" costs, and is named in the report.

k6 runs in a pinned ``grafana/k6`` image, so there is nothing to install.
"""

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

HERE = Path(__file__).parent
COMPOSE = ["docker", "compose", "-f", str(HERE / "docker-compose.yml")]
RESULTS = HERE.parent / "results" / "load"


def run(cmd: list[str], env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    import os

    return subprocess.run(
        cmd, check=True, text=True, capture_output=True, env={**os.environ, **(env or {})}
    )


VARY = "0"  # set by --cache: unique prompts, so every request exercises the cache path


def k6(name: str, target: str, rate: int, duration: str, model: str) -> dict[str, Any]:
    out = f"/out/{name}.json"
    cmd = [
        *COMPOSE, "--profile", "k6", "run", "--rm", "k6", "run",
        "-e", f"TARGET={target}", "-e", f"RATE={rate}", "-e", f"DURATION={duration}",
        "-e", f"MODEL={model}", "-e", f"VARY={VARY}", "--summary-export", out, "/scripts/k6.js",
    ]  # fmt: skip
    run(cmd)
    return json.loads((RESULTS / f"{name}.json").read_text(encoding="utf-8"))


def trend(summary: dict[str, Any]) -> dict[str, float]:
    m = summary["metrics"]["http_req_duration"]
    return {k: m[k] for k in ("med", "p(95)", "p(99)", "avg")}


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m bench.load")
    parser.add_argument("--rates", default="50,100,200", help="requests/second to test")
    parser.add_argument("--duration", default="30s")
    parser.add_argument("--replicas", default="1,2")
    parser.add_argument(
        "--cache",
        action="store_true",
        help="M1(b): cache on + Postgres/pgvector, unique prompts (embedding on every request)",
    )
    args = parser.parse_args()
    global VARY
    if args.cache:
        VARY = "1"
        COMPOSE.extend(["-f", str(HERE / "docker-compose.cache.yml")])
    RESULTS.mkdir(parents=True, exist_ok=True)
    rates = [int(x) for x in args.rates.split(",")]

    ncpu = run(["docker", "info", "--format", "{{.NCPU}}"]).stdout.strip()
    report: dict[str, Any] = {
        "duration": args.duration,
        "docker_cpus": ncpu,
        "runs": {},
        "cache": args.cache,
    }
    try:
        for n in (int(x) for x in args.replicas.split(",")):
            # Each container runs 2 worker processes (see the compose file).
            scale = f"api={n}"
            up = [*COMPOSE, "up", "-d", "--build", "--force-recreate", "--scale", scale, "--wait"]
            run(up, {"REPLICA_COUNT": str(2 * n)})
            # Warm-up: connection pools, the page cache, Redis script loading.
            k6("warmup", "http://lb:8080", min(rates), "10s", "relay/auto")
            for rate in rates:
                tag = f"{n}_{rate}"
                direct = k6(f"direct_{tag}", "http://sim:8100", rate, args.duration, "sim-model")
                through = k6(f"relay_{tag}", "http://lb:8080", rate, args.duration, "relay/auto")
                d, r = trend(direct), trend(through)
                m = through["metrics"]
                report["runs"][f"{n}x{rate}"] = {
                    "replicas": n,
                    "offered_rps": rate,
                    "achieved_rps": m["http_reqs"]["rate"],
                    "dropped": m.get("dropped_iterations", {}).get("count", 0),
                    "direct_ms": d,
                    "relay_ms": r,
                    "overhead_ms": {k: r[k] - d[k] for k in d},
                    "failed_rate": m["http_req_failed"]["value"],
                }
                print(f"replicas={n} rate={rate}: p95 overhead "
                      f"{r['p(95)'] - d['p(95)']:.1f} ms, achieved "
                      f"{m['http_reqs']['rate']:.0f} rps", flush=True)  # fmt: skip
    finally:
        subprocess.run([*COMPOSE, "down", "-v"], check=False, capture_output=True)

    suffix = "_cache" if args.cache else ""
    (RESULTS / f"load{suffix}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (RESULTS / f"summary{suffix}.md").write_text(markdown(report), encoding="utf-8")


def markdown(report: dict[str, Any]) -> str:
    cache = bool(report.get("cache"))
    title = "M1(b): cache on, embedding on every request" if cache else "M1(a): passthrough"
    what = (
        "Postgres + pgvector sit behind the cache and every prompt is unique, so each request "
        "embeds, queries pgvector, calls the provider and writes the answer back: the heaviest "
        "path the cache adds. "
        if cache
        else "Log rows are kept in memory (no Postgres); no cache, so this is passthrough. "
    )
    lines = [
        f"# Load results, {title} (simulator upstream)",
        "",
        f"Constant arrival rate for {report['duration']} per point. k6 (pinned image) -> nginx "
        "-> Relay replicas (2 vCPU and 2 worker processes each) -> simulator (20 ms first "
        f"token). Docker host CPUs: {report['docker_cpus']}. Admission state in Redis. {what}"
        "Overhead = through-Relay minus direct at the same percentile, so it includes nginx "
        "and Redis. A row where achieved < offered, or failures > 0, is a saturated gateway: "
        "its latencies measure queueing, not overhead.",
        "",
        "| replicas | offered | achieved | dropped | p50 overhead | p95 overhead | p99 overhead "
        "| failed |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for v in report["runs"].values():
        o = v["overhead_ms"]
        lines.append(
            f"| {v['replicas']} | {v['offered_rps']} | {v['achieved_rps']:.0f} | "
            f"{v['dropped']:.0f} | {o['med']:.1f} ms | {o['p(95)']:.1f} ms | "
            f"{o['p(99)']:.1f} ms | {v['failed_rate'] * 100:.2f}% |"
        )
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
