"""The trained router, as a file (ADR 0017).

A logistic regression is just numbers: one weight per feature, a bias, and the scaling used on
the scalar features. So the artifact is a small **JSON file**, not a pickle:

- loading it can't execute code (a pickle can, which makes "load a model file" a security
  question; JSON makes it a validation question);
- it is diffable in git, so a retrained router shows up as a reviewable change;
- inference needs only numpy.

``load`` validates everything it can and raises ``ArtifactError`` otherwise: the router treats
*any* artifact problem as "route to strong", so a file that is half-right must be refused, not
used. The checks: format version, feature-spec version, array shapes, finite numbers, and a
sha256 over the numeric content (so a hand-edited or truncated file doesn't pass).
"""

import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from relay.routing.features import FEATURE_SPEC_VERSION, N_SCALARS, feature_vector

FORMAT_VERSION = 1


class ArtifactError(Exception):
    """The router artifact is missing or invalid. Never fatal: callers route to strong."""


@dataclass(frozen=True)
class RouterArtifact:
    version: str  # recorded as the request's policy_version when this router decides
    embedding_model: str
    dimensions: int  # embedding size; the weight vector is dimensions + N_SCALARS long
    weights: np.ndarray
    bias: float
    scaler_mean: np.ndarray  # over the scalar block only
    scaler_scale: np.ndarray
    threshold: float  # the validation-chosen default (the route config's value is what runs)
    trained_on: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)

    def logit(self, embedding: np.ndarray, scalars: np.ndarray) -> float:
        x = feature_vector(embedding, scalars, self.scaler_mean, self.scaler_scale)
        return float(x @ self.weights + self.bias)

    def score(self, embedding: np.ndarray, scalars: np.ndarray) -> float:
        """P(the cheap model's answer is win-or-tie vs the strong model's), in [0, 1]."""
        return sigmoid(self.logit(embedding, scalars))

    def score_matrix(self, embeddings: np.ndarray, scalars: np.ndarray) -> np.ndarray:
        """Vectorized ``score`` for evaluation: rows are prompts."""
        std = (scalars - self.scaler_mean) / self.scaler_scale
        z = np.concatenate([embeddings, std], axis=1) @ self.weights + self.bias
        probabilities: np.ndarray = 1.0 / (1.0 + np.exp(-np.clip(z, -60, 60)))
        return probabilities

    # ------------------------------------------------------------------ (de)serialization
    def _core(self) -> dict[str, Any]:
        return {
            "weights": [float(x) for x in self.weights],
            "bias": float(self.bias),
            "scaler_mean": [float(x) for x in self.scaler_mean],
            "scaler_scale": [float(x) for x in self.scaler_scale],
        }

    def sha256(self) -> str:
        return _digest(self._core())

    def dumps(self) -> str:
        doc = {
            "format_version": FORMAT_VERSION,
            "feature_spec_version": FEATURE_SPEC_VERSION,
            "version": self.version,
            "embedding_model": self.embedding_model,
            "dimensions": self.dimensions,
            "threshold": self.threshold,
            "trained_on": self.trained_on,
            "metrics": self.metrics,
            **self._core(),
            "sha256": self.sha256(),
        }
        return json.dumps(doc, indent=1, sort_keys=False) + "\n"

    def save(self, path: Path) -> None:
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            f.write(self.dumps())


def sigmoid(z: float) -> float:
    # Split on sign so exp() never overflows for large |z|.
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


def _digest(core: dict[str, Any]) -> str:
    blob = json.dumps(core, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _array(doc: dict[str, Any], key: str, length: int) -> np.ndarray:
    raw = doc.get(key)
    if not isinstance(raw, list) or len(raw) != length:
        raise ArtifactError(f"{key}: expected a list of {length} numbers")
    try:
        arr = np.asarray(raw, dtype=float)
    except (TypeError, ValueError) as e:
        raise ArtifactError(f"{key}: not numeric") from e
    if not np.all(np.isfinite(arr)):
        raise ArtifactError(f"{key}: contains NaN or infinity")
    return arr


def loads(text: str) -> RouterArtifact:
    try:
        doc = json.loads(text)
    except ValueError as e:
        raise ArtifactError(f"not valid JSON: {e}") from e
    if not isinstance(doc, dict):
        raise ArtifactError("not a JSON object")
    if doc.get("format_version") != FORMAT_VERSION:
        raise ArtifactError(f"unsupported format_version {doc.get('format_version')!r}")
    if doc.get("feature_spec_version") != FEATURE_SPEC_VERSION:
        raise ArtifactError(
            f"trained for feature spec {doc.get('feature_spec_version')!r}, "
            f"this build computes {FEATURE_SPEC_VERSION}"
        )
    dimensions = doc.get("dimensions")
    if not isinstance(dimensions, int) or dimensions <= 0:
        raise ArtifactError("dimensions must be a positive integer")
    weights = _array(doc, "weights", dimensions + N_SCALARS)
    mean = _array(doc, "scaler_mean", N_SCALARS)
    scale = _array(doc, "scaler_scale", N_SCALARS)
    if np.any(scale <= 0):
        raise ArtifactError("scaler_scale must be positive")
    bias = doc.get("bias")
    if not isinstance(bias, int | float) or not math.isfinite(bias):
        raise ArtifactError("bias must be a finite number")
    threshold = doc.get("threshold")
    if not isinstance(threshold, int | float) or not 0.0 <= threshold <= 1.0:
        raise ArtifactError("threshold must be in [0, 1]")
    version, model = doc.get("version"), doc.get("embedding_model")
    if not isinstance(version, str) or not version or not isinstance(model, str) or not model:
        raise ArtifactError("version and embedding_model must be non-empty strings")

    artifact = RouterArtifact(
        version=version,
        embedding_model=model,
        dimensions=dimensions,
        weights=weights,
        bias=float(bias),
        scaler_mean=mean,
        scaler_scale=scale,
        threshold=float(threshold),
        trained_on=doc.get("trained_on") or {},
        metrics=doc.get("metrics") or {},
    )
    if doc.get("sha256") != artifact.sha256():
        raise ArtifactError("sha256 mismatch: the file was edited or truncated")
    return artifact


def load(path: Path) -> RouterArtifact:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as e:
        raise ArtifactError(f"cannot read {path}: {e}") from e
    return loads(text)
