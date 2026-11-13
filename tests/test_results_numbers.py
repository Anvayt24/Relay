"""guide/results.md quotes numbers; each one must still be what the committed result files say.

A page that states "99.9%" is a claim. If a rerun moves the number, this fails until the page is
updated, so the documentation cannot drift away from the evidence.
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAGE = (ROOT / "guide" / "results.md").read_text(encoding="utf-8")


def load(path: str) -> dict:  # type: ignore[type-arg]
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def cited(text: str) -> None:
    assert text in PAGE, f"guide/results.md no longer says {text!r}"


def test_chaos_numbers_match_the_committed_runs() -> None:
    for scenario in ("m2_headline", "hard_outage", "storm_429"):
        data = load(f"bench/results/chaos/{scenario}.json")
        assert data["runs"] == 5 and not data["smoke"]
        for client, stats in data["clients"].items():
            cited(f"{stats['success_rate']['mean'] * 100:.1f}%")
            cited(f"{stats['p95_s']['mean']:.2f} s")
            cited(f"{stats['retry_amplification']['mean']:.2f}")
            assert client in {"relay", "naive_retry", "direct"}


def test_cache_router_and_judge_numbers_match_their_metrics() -> None:
    cache = load("evals/cache/metrics.json")
    assert cache["guard_on"]["threshold"] == 0.8625
    cited("0.8625")
    sim = load("evals/router/sim/metrics.json")
    cited(f"{sim['test']['headline']['cheap_share'] * 100:.1f}%")
    judges = load("evals/judge/public/calibration.json")["judges"]
    sonnet = judges["anthropic/sonnet"]["all"]
    cited(f"{sonnet['kappa']['value']:.3f}")
    cited(f"{sonnet['loss_recall']['value']:.3f}")
    cited(f"{judges['gemini/flash-lite']['all']['kappa']['value']:.3f}")


def test_evidence_numbers_match_the_experiments() -> None:
    m = load("evals/evidence/metrics.json")
    cited(f"{m['coverage']['worst']['wald'] * 100:.1f}%")
    cited(f"{m['coverage']['worst']['bootstrap'] * 100:.1f}%")
    cited(f"**{m['detection']['monitors']['fixed_threshold']['median_delay']:.0f}**")
    cited(f"**{m['detection']['monitors']['e1_matched']['median_delay']:.0f}**")
    canary = m["canary"]
    cited(f"{canary['bad']['rolled_back'] * 100:.0f}%")
    cited(f"{canary['good']['promote_recommended'] * 100:.0f}%")
    cost = m["cost"]
    cited(f"{cost['undecided_rate']['share'] * 100:.1f}%")
    cited(f"{cost['decided_rate']['share'] * 100:.1f}%")
    cited(f"{cost['sonnet_judge']['undecided_rate']['share'] * 100:.1f}%")
    cited(f"{cost['sonnet_judge']['decided_rate']['share'] * 100:.1f}%")
