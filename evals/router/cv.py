"""Cross-validated AUC: is there *any* learnable signal in a small dataset?

    python -m evals.router.cv --world real

With only ~150 real prompts the held-out test split is about 27 prompts: far too few for an AUC
to mean anything. Cross-validation over *all* the rows says more about whether the cheap
model's win-or-tie outcome can be predicted from the prompt at all. Folds are made by cluster
(near-duplicate groups never straddle a fold), and the scalar scaler is re-fit inside each fold.

Reports, per regularization strength ``C``, the AUC of the pooled out-of-fold predictions with
a seeded bootstrap interval. An interval that includes 0.5 means "no detectable signal".
"""

import argparse
import json

import numpy as np
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold

from evals.router import data as data_mod
from evals.router.evaluate import auc
from evals.router.train import C_GRID, MAX_ITER, TOL, design_matrix

N_FOLDS = 5
N_BOOT = 1000


def oof_scores(ds: data_mod.Dataset, c: float) -> np.ndarray:
    y = ds.y_judge.astype(int)
    groups = np.array([r["cluster"] for r in ds.rows])
    out = np.zeros(len(y))
    for train_idx, test_idx in GroupKFold(n_splits=N_FOLDS).split(ds.embeddings, y, groups):
        mean = ds.scalars[train_idx].mean(axis=0)
        scale = ds.scalars[train_idx].std(axis=0)
        scale = np.where(scale < 1e-8, 1.0, scale)
        x = design_matrix(ds, mean, scale)
        model = LogisticRegression(C=c, solver="lbfgs", max_iter=MAX_ITER, tol=TOL)
        model.fit(x[train_idx], y[train_idx])
        out[test_idx] = model.predict_proba(x[test_idx])[:, 1]
    return out


def auc_ci(scores: np.ndarray, labels: np.ndarray) -> tuple[float, float, float]:
    rng = np.random.default_rng(0)
    point = auc(scores, labels)
    boots = []
    for _ in range(N_BOOT):
        idx = rng.integers(0, len(labels), len(labels))
        v = auc(scores[idx], labels[idx])
        if not np.isnan(v):
            boots.append(v)
    return point, float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m evals.router.cv", description=__doc__)
    parser.add_argument("--world", default="real", choices=["sim", "real"])
    args = parser.parse_args()
    ds = data_mod.load(args.world)
    y = ds.y_judge
    print(f"{len(y)} prompts, cheap-answer win-or-tie rate {y.mean():.3f}")
    rows = []
    for c in C_GRID:
        scores = oof_scores(ds, c)
        point, lo, hi = auc_ci(scores, y)
        rows.append(
            {"C": c, "auc": round(point, 4), "ci_low": round(lo, 4), "ci_high": round(hi, 4)}
        )
        print(f"C={c:<5} out-of-fold AUC {point:.3f}  (95% CI {lo:.3f}-{hi:.3f})")
    # One more baseline: the prompt's length alone, no embedding at all.
    length = ds.scalars[:, 0]
    ranks = rankdata(length)
    point, lo, hi = auc_ci(ranks, y)
    print(f"length only      AUC {point:.3f}  (95% CI {lo:.3f}-{hi:.3f})")
    out = data_mod.HERE / args.world / "cv.json"
    data_mod.write_text(
        out,
        json.dumps(
            {"folds": N_FOLDS, "n": len(y), "by_C": rows, "length_only_auc": round(point, 4)},
            indent=2,
        ),
    )
    print(f"wrote {out.name}")


if __name__ == "__main__":
    main()
