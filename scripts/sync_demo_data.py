"""Write the small JSON files the demo's evidence view serves (relay/demo/data/).

    python -m scripts.sync_demo_data           # rewrite them from the evaluation outputs
    python -m scripts.sync_demo_data --check   # fail if they are out of date (CI, via a test)

The evaluation outputs live in evals/, which the container image does not carry. These files
are a reduced copy of exactly what the page shows, so the image needs nothing else, and
`make reproduce` regenerates them from the same runs that produce every cited number.
"""

import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "relay" / "demo" / "data"


def _load(path: str) -> Any:
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def build() -> dict[str, Any]:
    calibration = _load("evals/judge/public/calibration.json")
    judges = {}
    for name, entry in calibration["judges"].items():
        whole = entry["all"]
        judges[name] = {
            "rubric": entry["rubric"],
            "n": whole["n"],
            "abstained": whole["abstained"],
            "status": whole["status"],
            "kappa": whole["kappa"],
            "loss_recall": whole["loss_recall"],
            "accuracy": whole["accuracy"],
        }
    evidence = _load("evals/evidence/metrics.json")
    return {
        "cards.json": _load("evals/evidence/real/cards.json"),
        "judge_calibration.json": {
            "judges": judges,
            "gate": calibration["gate"],
            "human_agreement_mtbench": calibration["human_agreement_mtbench"],
            "caveat": calibration["caveat"],
        },
        "evidence_sim.json": {
            "canary": evidence["canary"],
            "gate": evidence["gate"],
            "m8": {
                "cheap_judge": evidence["cost"]["undecided_rate"]["share"],
                "sonnet_judge": evidence["cost"]
                .get("sonnet_judge", {})
                .get("undecided_rate", {})
                .get("share"),
            },
        },
    }


def render(value: Any) -> str:
    return json.dumps(value, indent=2, sort_keys=True) + "\n"


def main() -> int:
    files = build()
    if "--check" in sys.argv:
        stale = [n for n, v in files.items() if (OUT / n).read_text(encoding="utf-8") != render(v)]
        if stale:
            print("out of date (run python -m scripts.sync_demo_data):", ", ".join(stale))
            return 1
        return 0
    OUT.mkdir(parents=True, exist_ok=True)
    for name, value in files.items():
        (OUT / name).write_text(render(value), encoding="utf-8", newline="\n")
    print("wrote", ", ".join(files))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
