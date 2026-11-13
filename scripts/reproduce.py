"""Regenerate every number the README and the guide cite, and prove they did not change.

    python -m scripts.reproduce            # make reproduce

Each step rewrites its own outputs from frozen data and fixed seeds, offline: no model, no
network, no provider key. The script reads the written files before and after, so it fails if a
regeneration produced different numbers from the ones committed. That is the claim the README
makes ("only numbers `make reproduce` produces"), checked rather than asserted.

What "different" means:
- a `.json` file is compared number by number, to a relative tolerance of 1e-9. The last digit
  of a float differs between machines and library versions without any number really changing;
- a `.md` report is compared as text (line endings aside): its numbers are already rounded.

Charts (.png) are regenerated but not compared: their bytes depend on the plotting library.

Load and chaos results come from timing runs in Docker, which cannot be bit-identical, so they
are not here: `make reproduce-bench` re-runs them and reports mean and spread.
"""

import json
import math
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
REL_TOL = 1e-9

# (what, the command, the files it writes that are compared)
STEPS: list[tuple[str, list[str], list[str]]] = [
    (
        "semantic cache evaluation",
        ["-m", "evals.cache.evaluate"],
        ["evals/cache/metrics.json", "evals/cache/report.md"],
    ),
    (
        "router evaluation, simulated world",
        ["-m", "evals.router.evaluate"],
        ["evals/router/sim/metrics.json", "evals/router/sim/report.md"],
    ),
    (
        "router evaluation, real prompts",
        ["-m", "evals.router.evaluate", "--world", "real"],
        ["evals/router/real/metrics.json", "evals/router/real/report.md"],
    ),
    (
        "evidence engine experiments",
        ["-m", "evals.evidence.evaluate"],
        ["evals/evidence/metrics.json", "evals/evidence/report.md"],
    ),
    (
        "judge calibration on public human-labeled data",
        ["-m", "evals.judge.public.report"],
        ["evals/judge/public/calibration.json", "evals/judge/public/report.md"],
    ),
    (
        "the demo's evidence data",
        ["-m", "scripts.sync_demo_data"],
        [
            "relay/demo/data/cards.json",
            "relay/demo/data/judge_calibration.json",
            "relay/demo/data/evidence_sim.json",
        ],
    ),
]


def read(rel: str) -> str | None:
    path = ROOT / rel
    if not path.exists():
        return None
    return path.read_bytes().decode("utf-8").replace("\r\n", "\n")


def close(a: Any, b: Any) -> bool:
    """Equal, with floats compared to REL_TOL."""
    if isinstance(a, float) and isinstance(b, float):
        return (math.isnan(a) and math.isnan(b)) or math.isclose(
            a, b, rel_tol=REL_TOL, abs_tol=1e-12
        )
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(close(a[k], b[k]) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(close(x, y) for x, y in zip(a, b, strict=True))
    return bool(a == b)


def same(rel: str, before: str | None, after: str | None) -> bool:
    if before is None or after is None:
        return before == after
    if rel.endswith(".json"):
        return close(json.loads(before), json.loads(after))
    return before == after


def main() -> int:
    changed: list[str] = []
    for what, args, outputs in STEPS:
        before = {rel: read(rel) for rel in outputs}
        started = time.monotonic()
        print(f"== {what} ...", flush=True)
        run = subprocess.run([sys.executable, *args], cwd=ROOT, capture_output=True, text=True)
        if run.returncode != 0:
            print(run.stdout[-2000:], run.stderr[-2000:], sep="\n")
            print(f"FAILED: {what} (exit {run.returncode})")
            return 1
        moved = [rel for rel in outputs if not same(rel, before[rel], read(rel))]
        changed += moved
        status = "unchanged" if not moved else "CHANGED: " + ", ".join(moved)
        print(f"   {status} ({time.monotonic() - started:.0f}s)", flush=True)
    if changed:
        print(
            "\nreproduce FAILED: regenerating gave different numbers from the committed ones.\n"
            "If that is intended (an approved change), commit the new files; otherwise find "
            "what changed:\n  " + "\n  ".join(changed)
        )
        return 1
    print("\nreproduce OK: every cited number was regenerated and matches.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
