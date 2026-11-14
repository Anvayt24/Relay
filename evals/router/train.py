"""Train the router (PRD FR-7; ADR 0017). Offline, with scikit-learn.

    python -m evals.router.train                 # make train-router
    python -m evals.router.train --world real

What it does:
1. Standardize the scalar features using the **train** split's mean and standard deviation.
2. For each regularization strength ``C`` in a small fixed grid, fit an L2 logistic regression on
   the train split, using the **judge's** labels (the labels one actually has in production).
3. Keep the ``C`` with the lowest log-loss on the validation split.
4. Choose the routing threshold on validation (``evaluate.select_threshold``) and write the
   artifact: a plain JSON file of weights (``relay/routing/artifacts/<world>-world.json``).

sklearn is used here and only here: serving reads the JSON and does one dot product in numpy
(``relay/routing/router.py``), so the gateway image never contains sklearn.

The fit is deterministic for a given sklearn version (lbfgs, fixed tolerance, no randomness),
but sklearn's numerics can differ slightly across platforms and versions. That is why CI
gates on the *committed artifact*, not on re-training; ``tests/bench`` checks that re-training
here reproduces the artifact's predictions.
"""

import argparse
from dataclasses import replace
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

from evals.router import data as data_mod
from evals.router import evaluate
from evals.router.data import Dataset
from relay.routing.artifact import RouterArtifact
from relay.routing.features import N_SCALARS, feature_vector

C_GRID = (0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0)
MAX_ITER = 5000
TOL = 1e-8


def design_matrix(ds: Dataset, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
    return np.stack(
        [feature_vector(ds.embeddings[i], ds.scalars[i], mean, scale) for i in range(len(ds.rows))]
    )


def fit(ds: Dataset, c: float, x: np.ndarray, train: np.ndarray) -> LogisticRegression:
    model = LogisticRegression(C=c, solver="lbfgs", max_iter=MAX_ITER, tol=TOL)
    model.fit(x[train], ds.y_judge[train].astype(int))
    return model


def train(ds: Dataset, version: str | None = None) -> tuple[RouterArtifact, dict[str, Any]]:
    train_m, val_m = ds.mask("train"), ds.mask("val")
    mean = ds.scalars[train_m].mean(axis=0)
    scale = ds.scalars[train_m].std(axis=0)
    scale = np.where(
        scale < 1e-8, 1.0, scale
    )  # a constant column (e.g. tools) must not divide by 0
    x = design_matrix(ds, mean, scale)

    results = []
    for c in C_GRID:
        model = fit(ds, c, x, train_m)
        p = model.predict_proba(x[val_m])[:, 1]
        results.append((float(log_loss(ds.y_judge[val_m].astype(int), p, labels=[0, 1])), c, model))
    val_loss, best_c, best = min(results, key=lambda r: r[0])

    dims = ds.embeddings.shape[1]
    artifact = RouterArtifact(
        version=version or f"router-{ds.name}-v1",
        embedding_model=str(ds.manifest["embedding_model"]),
        dimensions=dims,
        weights=best.coef_[0].astype(float),
        bias=float(best.intercept_[0]),
        scaler_mean=mean,
        scaler_scale=scale,
        threshold=0.5,  # replaced below, once the scores exist
        trained_on={
            "dataset": ds.name,
            "dataset_sha256": ds.manifest.get("sha256"),
            "n_train": int(train_m.sum()),
            "labels": "judge (y_judge); validation chose C and the threshold",
            "sklearn": "LogisticRegression(l2, lbfgs)",
            "C": best_c,
        },
    )
    scores = artifact.score_matrix(ds.embeddings, ds.scalars)
    selection = evaluate.select_threshold(ds, scores)
    full = replace(
        artifact,
        threshold=float(selection["threshold"]),
        metrics={
            "validation_log_loss": round(val_loss, 6),
            "C_grid": list(C_GRID),
            "threshold_selection": selection,
            "n_features": dims + N_SCALARS,
        },
    )
    return full, {"C": best_c, "val_log_loss": val_loss, "selection": selection}


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m evals.router.train", description=__doc__)
    parser.add_argument("--world", default="sim", choices=["sim", "real"])
    parser.add_argument("--version", default=None)
    args = parser.parse_args()
    ds = data_mod.load(args.world)
    artifact, info = train(ds, args.version)
    out = evaluate.artifact_path(args.world)
    out.parent.mkdir(parents=True, exist_ok=True)
    artifact.save(out)
    print(
        f"trained {artifact.version}: C={info['C']}, validation log-loss {info['val_log_loss']:.4f}"
    )
    print(f"threshold (validation, M4): {info['selection']['threshold']}  -> {out}")


if __name__ == "__main__":
    main()
