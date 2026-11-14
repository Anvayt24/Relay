"""Router evaluation (PRD E-A, M4; ADR 0018).

    python -m evals.router.evaluate                  # write metrics.json, report.md, curve.png
    python -m evals.router.evaluate --check          # fail if the committed baseline regressed
    python -m evals.router.evaluate --world real     # the real (uncalibrated-judge) dataset

Inputs are frozen: the dataset folder (prompts + embeddings) and the **committed artifact**.
Inference is numpy only, and the bootstrap is seeded, so the result is deterministic and CI can
gate on it. (Re-training is a separate step, ``train.py``: sklearn's solver can differ slightly
between platforms, which would break an exact-equality gate. A test checks that re-training
reproduces the artifact's predictions.)

**Policy quality.** A policy sends each request cheap or strong. Its quality is the mean over
requests of 1 if it was sent strong (a tie with the strong model by construction, FR-7) and
otherwise the cheap model's win-or-tie outcome. Always-strong is therefore 1.0; always-cheap
is the cheap model's win-or-tie rate; a router sits in between. The x-axis is the **share sent
strong**.

**Choosing the threshold (M4).** On the *validation* split only: the most aggressive threshold
(largest cheap share) whose one-sided 95% lower bound on policy quality is >= 0.95, scanning down
from the safest threshold and stopping at the first failure (so one lucky dip below the line
can't pick it). Then it is *reported* on the *test* split. On validation the quality estimate
is **PPI-corrected** (judge labels on every prompt, corrected by the gold subset): the real
procedure, because the judge alone is biased.

**Three estimates of test quality, side by side:** ground truth (known in the simulation),
judge-only (biased by the judge's errors), and PPI-corrected (the circularity fix PRD FR-7
asks for). For real data there is no ground truth: quality is judge-only unless a human-labeled
gold subset exists, and the report says so.
"""

import argparse
import itertools
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import rankdata

from evals.router import data as data_mod
from evals.router.data import Dataset
from relay.config.loader import load_config
from relay.evidence.stats import MIN_PPI_LABELED, ppi_mean_ci
from relay.routing.artifact import RouterArtifact
from relay.routing.artifact import load as load_artifact

ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / "relay" / "routing" / "artifacts"
M4_FLOOR = 0.95  # lower bound of policy quality (PRD M4)
M4_TARGET_SHARE = 0.40  # share sent to the cheap model (PRD M4 target)
THRESHOLDS = np.round(np.arange(0.0, 1.0 + 1e-9, 0.005), 4)
N_BOOT = 1000
SEED = 0
ALPHA_ONE_SIDED = 0.10  # PPI two-sided 90% interval: its lower end is a one-sided 95% bound
# Illustrative request size for the cost line (tokens); the shipped list prices do the rest.
ILLUSTRATIVE_TOKENS = (100, 300)


# ---------------------------------------------------------------- pieces
def artifact_path(world: str) -> Path:
    return ARTIFACTS / f"{world}-world.json"


def split_view(ds: Dataset, scores: np.ndarray, split: str) -> dict[str, np.ndarray]:
    m = ds.mask(split)
    gold = ds.gold[m]
    return {
        "scores": scores[m],
        "y_judge": ds.y_judge[m],
        "y_true": ds.y_true[m],
        "gold": gold,
        "family": ds.families[m],
        "w": ds.col("w")[m],
    }


def reference_labels(view: dict[str, np.ndarray]) -> tuple[np.ndarray, str]:
    """The labels quality is measured against: ground truth when known, else the judge's."""
    if np.isfinite(view["y_true"]).all():
        return view["y_true"], "ground_truth"
    return view["y_judge"], "judge_only"


def policy_terms(cheap: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Per-request quality contribution: 1 if sent strong, else the cheap outcome."""
    return np.where(cheap, labels, 1.0)


def ppi_policy_quality(view: dict[str, np.ndarray], cheap: np.ndarray) -> Any:
    """PPI-corrected policy quality on a split, or None without enough gold labels.

    The policy's per-request outcome is known exactly for strong-routed requests (1), so PPI
    is applied to the whole outcome vector: judge outcomes on the judge-only prompts, corrected
    by truth-vs-judge on the gold prompts."""
    gold = view["gold"]
    if gold.sum() < MIN_PPI_LABELED or not np.isfinite(view["y_true"][gold]).all():
        return None
    f = policy_terms(cheap, view["y_judge"])
    y = policy_terms(cheap, view["y_true"])
    return ppi_mean_ci(f[~gold], y[gold], f[gold], alpha=ALPHA_ONE_SIDED)


def lower_bound_one_sided(view: dict[str, np.ndarray], cheap: np.ndarray) -> tuple[float, str]:
    """One-sided 95% lower bound on policy quality, and how it was obtained."""
    ppi = ppi_policy_quality(view, cheap)
    if ppi is not None:
        return float(ppi.ci_low), "ppi"
    labels, basis = reference_labels(view)
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(labels), size=(N_BOOT, len(labels)))
    boot = policy_terms(cheap, labels)[idx].mean(axis=1)
    return float(np.percentile(boot, 5)), f"bootstrap_{basis}"


def select_threshold(ds: Dataset, scores: np.ndarray) -> dict[str, Any]:
    """The M4 threshold from the validation split alone (see module docstring)."""
    val = split_view(ds, scores, "val")
    chosen: float | None = None
    rows = []
    for t in THRESHOLDS[::-1]:  # safest (highest) first
        cheap = val["scores"] >= t
        lower, how = lower_bound_one_sided(val, cheap)
        rows.append((float(t), float(cheap.mean()), lower, how))
        if lower >= M4_FLOOR:
            chosen = float(t)
        else:
            break
    out: dict[str, Any] = {
        "threshold": 1.0 if chosen is None else chosen,
        "met": chosen is not None,
    }
    if chosen is not None:
        t, share, lower, how = next(r for r in rows if r[0] == chosen)
        out["validation"] = {"cheap_share": share, "quality_lower_95": lower, "basis": how}
    return out


def auc(scores: np.ndarray, positive: np.ndarray) -> float:
    """Probability a random win-or-tie prompt outscores a random loss prompt (Mann-Whitney)."""
    pos = positive.astype(bool)
    n_pos, n_neg = int(pos.sum()), int((~pos).sum())
    if n_pos == 0 or n_neg == 0:
        return math.nan
    ranks = rankdata(scores)
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


WORLD_MODELS = {
    "sim": ("mock/cheap", "mock/strong"),
    "real": ("groq/gpt-oss-20b", "groq/gpt-oss-120b"),
}


def cost_saved_fraction(cheap_share: float, world: str = "sim") -> float | None:
    """List-price saving vs always-strong at an illustrative request size, using the route's
    cheap and strong models from the shipped config."""
    config = load_config(None)
    cheap_name, strong_name = WORLD_MODELS[world]
    cheap, strong = config.models[cheap_name], config.models[strong_name]
    p, c = ILLUSTRATIVE_TOKENS

    def cost(m: Any) -> float:
        return float(m.price_in) * p / 1e6 + float(m.price_out) * c / 1e6

    return cheap_share * (1.0 - cost(cheap) / cost(strong))


# ---------------------------------------------------------------- the evaluation
def evaluate_world(ds: Dataset, artifact: RouterArtifact, threshold: float | None = None) -> dict:
    scores = artifact.score_matrix(ds.embeddings, ds.scalars)
    selection = select_threshold(ds, scores)
    t_star = selection["threshold"] if threshold is None else threshold
    test = split_view(ds, scores, "test")
    labels, basis = reference_labels(test)
    n = len(labels)
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, n, size=(N_BOOT, n))  # one resample reused by every policy: paired

    def quality_ci(terms: np.ndarray) -> dict[str, float]:
        boot = terms[idx].mean(axis=1)
        return {
            "quality": float(terms.mean()),
            "ci_low": float(np.percentile(boot, 2.5)),
            "ci_high": float(np.percentile(boot, 97.5)),
            "lower_95_one_sided": float(np.percentile(boot, 5)),
        }

    cheap = test["scores"] >= t_star
    share = float(cheap.mean())
    router_q = quality_ci(policy_terms(cheap, labels))

    ppi = ppi_policy_quality(test, cheap)
    judge_terms = policy_terms(cheap, test["y_judge"])
    headline: dict[str, Any] = {
        "threshold": t_star,
        "cheap_share": share,
        "quality_basis": basis,
        "quality": router_q["quality"],
        "quality_lower_95": router_q["lower_95_one_sided"],
        "judge_only_quality": float(judge_terms.mean()),
        "meets_quality_floor": bool(router_q["lower_95_one_sided"] >= M4_FLOOR),
        "meets_share_target": bool(share >= M4_TARGET_SHARE),
        "cost_saved_fraction_illustrative": cost_saved_fraction(share, ds.name),
    }
    if ppi is not None:
        headline["ppi"] = ppi.as_dict()
        headline["ppi_quality_lower_95"] = float(ppi.ci_low)

    # --- baselines and the router's curve (test split)
    always_cheap = quality_ci(policy_terms(np.ones(n, bool), labels))
    base_rate = float(labels.mean())
    random_policy = []
    for p in np.round(np.linspace(0, 1, 11), 1):
        terms_mean = (1 - p) + p * labels  # expected outcome per request under random(p)
        boot = terms_mean[idx].mean(axis=1)
        random_policy.append(
            {
                "share_cheap": float(p),
                "quality": float(terms_mean.mean()),
                "ci_low": float(np.percentile(boot, 2.5)),
                "ci_high": float(np.percentile(boot, 97.5)),
            }
        )
    oracle_cheap = labels == 1  # sends cheap exactly when the cheap answer is fine
    curve = []
    for t in np.round(np.arange(0.0, 1.0 + 1e-9, 0.05), 2):
        c = test["scores"] >= t
        q = quality_ci(policy_terms(c, labels))
        s = float(c.mean())
        curve.append(
            {
                "threshold": float(t),
                "share_cheap": s,
                "share_strong": 1 - s,
                "random_quality_at_share": 1 - s * (1 - base_rate),
                **q,
            }
        )

    families: dict[str, Any] = {}
    for fam in sorted(set(test["family"].tolist())):
        m = test["family"] == fam
        families[fam] = {
            "n": int(m.sum()),
            "router_cheap_share": float(cheap[m].mean()),
            "cheap_win_or_tie_rate": float(labels[m].mean()),
        }

    bins = np.quantile(test["scores"], np.linspace(0, 1, 11))
    calibration = []
    for lo, hi in itertools.pairwise(bins):
        m = (test["scores"] >= lo) & (test["scores"] <= hi)
        if m.any():
            calibration.append(
                {
                    "mean_score": float(test["scores"][m].mean()),
                    "observed_win_or_tie": float(labels[m].mean()),
                    "n": int(m.sum()),
                }
            )

    return {
        "what": "router evaluation on the held-out test split (ADR 0018)",
        "world": ds.name,
        "dataset": {
            "rows": len(ds.rows),
            "splits": {s: int(ds.mask(s).sum()) for s in data_mod.SPLITS},
            "test_gold": int(test["gold"].sum()),
            "embedding_model": ds.manifest.get("embedding_model"),
            "label_note": ds.manifest.get("label"),
            "sha256": ds.manifest.get("sha256"),
        },
        "artifact": {
            "version": artifact.version,
            "sha256": artifact.sha256(),
            "embedding_model": artifact.embedding_model,
            "threshold_in_artifact": artifact.threshold,
        },
        "targets": {"quality_lower_bound_floor": M4_FLOOR, "cheap_share": M4_TARGET_SHARE},
        "threshold_selection": selection,
        "test": {
            "headline": headline,
            "baselines": {
                "always_strong": {"share_cheap": 0.0, "quality": 1.0},
                "always_cheap": {"share_cheap": 1.0, **always_cheap},
                "random": random_policy,
                "oracle": {
                    "share_cheap": float(oracle_cheap.mean()),
                    "quality": 1.0,
                    "note": "sends cheap exactly when the cheap answer is win-or-tie: an upper bound",  # noqa: E501
                },
            },
            "curve": curve,
            "auc": auc(test["scores"], labels),
            "families": families,
            "calibration": calibration,
        },
    }


# ---------------------------------------------------------------- the gate
def headline_numbers(metrics: dict[str, Any]) -> dict[str, float]:
    """What CI protects: small, rounded, and defined for both worlds."""
    h = metrics["test"]["headline"]
    return {
        "threshold": round(float(h["threshold"]), 4),
        "cheap_share": round(float(h["cheap_share"]), 4),
        "quality_lower_95": round(float(h["quality_lower_95"]), 4),
        "auc": round(float(metrics["test"]["auc"]), 4),
    }


# Which direction is a regression. The threshold must not change at all: it is a decision.
HIGHER_IS_BETTER = ("cheap_share", "quality_lower_95", "auc")
TOLERANCE = 5e-4


def compare(old: dict[str, float], new: dict[str, float]) -> list[str]:
    problems = []
    if old["threshold"] != new["threshold"]:
        problems.append(f"threshold {old['threshold']} -> {new['threshold']}")
    for key in HIGHER_IS_BETTER:
        if new[key] < old[key] - TOLERANCE:
            problems.append(f"{key} {old[key]} -> {new[key]}")
    return problems


# ---------------------------------------------------------------- report
def write_report(metrics: dict[str, Any], path: Path) -> None:
    h = metrics["test"]["headline"]
    sel = metrics["threshold_selection"]
    base = metrics["test"]["baselines"]
    simulated = metrics["world"] == "sim"
    lines = [
        f"# Router evaluation: {metrics['world']} world",
        "",
        (
            "**Simulated world.** Difficulty was built into the prompts (ADR 0018), so this "
            "measures the *machinery* (training, thresholding, bootstrap, PPI, the CI gate), "
            "not real-world savings."
            if simulated
            else "**Real data, uncalibrated judge.** Labels come from an LLM judge that has not "
            "passed the M7 gate (no human labels yet), the set is small, so treat every interval "
            "as wide and nothing here may drive a decision."
        ),
        "",
        f"Router `{metrics['artifact']['version']}` (embedding `{metrics['artifact']['embedding_model']}`). "  # noqa: E501
        f"Test split: {metrics['dataset']['splits']['test']} prompts, "
        f"{metrics['dataset']['test_gold']} of them gold-labeled.",
        "",
        f"## Headline (M4: lower 95% bound of policy quality >= {M4_FLOOR}, share to cheap >= "
        f"{M4_TARGET_SHARE:.0%})",
        "",
        f"- Threshold, chosen on **validation** only: **{h['threshold']}** "
        + ("" if sel["met"] else "(no threshold met the floor on validation: nothing goes cheap)"),
        f"- On **test**, **{h['cheap_share']:.1%}** of requests go to the cheap model "
        f"(target {M4_TARGET_SHARE:.0%}: {'met' if h['meets_share_target'] else 'NOT met'}).",
        f"- Quality vs {h['quality_basis'].replace('_', ' ')}: **{h['quality']:.4f}**, one-sided 95% "  # noqa: E501
        f"lower bound **{h['quality_lower_95']:.4f}** "
        f"(floor {M4_FLOOR}: {'met' if h['meets_quality_floor'] else 'NOT met'}).",
        f"- Judge-only estimate (biased by the judge's errors): {h['judge_only_quality']:.4f}.",
    ]
    if "ppi" in h:
        p = h["ppi"]
        lines.append(
            f"- **PPI-corrected** estimate: {p['estimate']:.4f}, one-sided 95% lower bound "
            f"{h['ppi_quality_lower_95']:.4f}; the {p['n_labeled']} gold labels are worth about "
            f"{p['ess_gain']:.1f}x as many human labels (lambda {p['lam']:.2f})."
        )
    if h.get("cost_saved_fraction_illustrative") is not None:
        lines.append(
            f"- List-price cost saved vs always-strong: {h['cost_saved_fraction_illustrative']:.1%} "  # noqa: E501
            f"(illustrative: {ILLUSTRATIVE_TOKENS[0]} prompt + {ILLUSTRATIVE_TOKENS[1]} "
            "completion tokens per request, list prices from relay.yaml)."
        )
    lines += [
        "",
        "## Baselines (test)",
        "",
        "| policy | share cheap | quality |",
        "|---|---|---|",
        "| always strong | 0% | 1.0000 |",
        f"| always cheap | 100% | {base['always_cheap']['quality']:.4f} "
        f"({base['always_cheap']['ci_low']:.3f}-{base['always_cheap']['ci_high']:.3f}) |",
        f"| oracle (sees the truth) | {base['oracle']['share_cheap']:.1%} | 1.0000 |",
        f"| **router** | {h['cheap_share']:.1%} | {h['quality']:.4f} |",
        "",
        f"AUC (score vs the cheap answer being win-or-tie): {metrics['test']['auc']:.3f}.",
        "",
        "## Quality vs share sent strong (test)",
        "",
        "| threshold | share strong | router quality (95% CI) | random at same share |",
        "|---|---|---|---|",
    ]
    for r in metrics["test"]["curve"]:
        lines.append(
            f"| {r['threshold']:.2f} | {r['share_strong']:.1%} | {r['quality']:.4f} "
            f"({r['ci_low']:.3f}-{r['ci_high']:.3f}) | {r['random_quality_at_share']:.4f} |"
        )
    lines += ["", "## Where the router sends traffic (test, by family)", ""]
    lines += [
        "| family | prompts | router sends cheap | cheap answer is win-or-tie |",
        "|---|---|---|---|",
    ]
    for fam, v in metrics["test"]["families"].items():
        lines.append(
            f"| {fam} | {v['n']} | {v['router_cheap_share']:.0%} | {v['cheap_win_or_tie_rate']:.0%} |"  # noqa: E501
        )
    lines += ["", "## Calibration (test): is a score of 0.8 really 80%?", ""]
    lines += ["| mean score | observed win-or-tie | n |", "|---|---|---|"]
    for c in metrics["test"]["calibration"]:
        lines.append(f"| {c['mean_score']:.3f} | {c['observed_win_or_tie']:.3f} | {c['n']} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot(metrics: dict[str, Any], path: Path) -> bool:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    curve = metrics["test"]["curve"]
    xs = [r["share_strong"] * 100 for r in curve]
    ax.plot(xs, [r["quality"] for r in curve], marker="o", ms=3, label="router (threshold sweep)")
    ax.fill_between(xs, [r["ci_low"] for r in curve], [r["ci_high"] for r in curve], alpha=0.2)
    rnd = metrics["test"]["baselines"]["random"]
    ax.plot([(1 - r["share_cheap"]) * 100 for r in rnd], [r["quality"] for r in rnd],
            ls="--", label="random")  # fmt: skip
    cheap_q = metrics["test"]["baselines"]["always_cheap"]["quality"]
    ax.scatter([0], [cheap_q], color="C3", zorder=5, label="always cheap")
    ax.scatter([100], [1.0], color="C2", zorder=5, label="always strong")
    ax.axhline(M4_FLOOR, color="grey", ls=":", lw=1)
    ax.set_xlabel("share of requests sent to the strong model (%)")
    ax.set_ylabel("policy quality (win-or-tie rate vs always strong)")
    ax.set_title(f"Router quality vs cost ({metrics['world']} world, test split)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return True


def main() -> int:
    parser = argparse.ArgumentParser(prog="python -m evals.router.evaluate", description=__doc__)
    parser.add_argument("--world", default="sim", choices=["sim", "real"])
    parser.add_argument("--check", action="store_true", help="compare with the committed metrics")
    args = parser.parse_args()
    ds = data_mod.load(args.world)
    artifact = load_artifact(artifact_path(args.world))
    metrics = evaluate_world(ds, artifact)
    folder = data_mod.HERE / args.world
    metrics_path = folder / "metrics.json"
    if args.check:
        if not metrics_path.exists():
            print(f"{metrics_path} is missing: run `make eval-router` and commit it")
            return 1
        old = headline_numbers(json.loads(metrics_path.read_text(encoding="utf-8")))
        new = headline_numbers(metrics)
        problems = compare(old, new)
        if problems:
            print(
                f"router eval ({args.world}) regressed vs the committed baseline: "
                + "; ".join(problems)
            )
            print("If intended, run `make eval-router` and commit the new metrics.json.")
            return 1
        print(f"router eval ({args.world}) OK: {new}")
        return 0
    data_mod.write_text(metrics_path, json.dumps(metrics, indent=2))
    write_report(metrics, folder / "report.md")
    plotted = plot(metrics, folder / "curve.png")
    print(json.dumps(headline_numbers(metrics), indent=2))
    print(f"wrote {metrics_path.name}, report.md" + (", curve.png" if plotted else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
