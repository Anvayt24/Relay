"""The router artifact: a safe, validated file format (ADR 0017)."""

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from relay.routing.artifact import ArtifactError, RouterArtifact, load, loads, sigmoid
from relay.routing.features import N_SCALARS

from .conftest import length_artifact


def doc_of(artifact: RouterArtifact) -> dict[str, Any]:
    return json.loads(artifact.dumps())  # type: ignore[no-any-return]


def test_round_trip_preserves_everything(tmp_path: Path) -> None:
    original = length_artifact()
    path = tmp_path / "router.json"
    original.save(path)
    back = load(path)
    assert back.version == original.version
    assert back.embedding_model == original.embedding_model
    assert back.sha256() == original.sha256()
    np.testing.assert_array_equal(back.weights, original.weights)
    assert back.score(np.zeros(384), np.zeros(N_SCALARS)) == original.score(
        np.zeros(384), np.zeros(N_SCALARS)
    )


def test_saved_file_has_lf_line_endings(tmp_path: Path) -> None:
    path = tmp_path / "router.json"
    length_artifact().save(path)
    assert b"\r\n" not in path.read_bytes()  # byte-stable on Windows and Linux (hash, git)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda d: d.update(format_version=99), "format_version"),
        (lambda d: d.update(feature_spec_version=99), "feature spec"),
        (lambda d: d.update(dimensions=-1), "dimensions"),
        (lambda d: d.update(weights=d["weights"][:-1]), "weights"),
        (lambda d: d.update(weights=[*d["weights"][:-1], "x"]), "numeric"),
        (lambda d: d.update(scaler_scale=[0.0] * N_SCALARS), "scaler_scale"),
        (lambda d: d.update(bias="high"), "bias"),
        (lambda d: d.update(threshold=1.5), "threshold"),
        (lambda d: d.update(version=""), "version"),
        (lambda d: d.update(embedding_model=None), "embedding_model"),
        (lambda d: d["weights"].__setitem__(0, 0.123), "sha256"),  # hand-edited weight
        (lambda d: d.pop("sha256"), "sha256"),
    ],
)
def test_invalid_artifacts_are_refused(mutate: Any, message: str) -> None:
    doc = doc_of(length_artifact())
    mutate(doc)
    with pytest.raises(ArtifactError, match=message):
        loads(json.dumps(doc))


def test_nan_and_infinity_are_refused() -> None:
    doc = doc_of(length_artifact())
    doc["weights"][3] = float("nan")
    with pytest.raises(ArtifactError, match="NaN"):
        loads(json.dumps(doc))  # json.dumps writes NaN; the loader must still reject it


@pytest.mark.parametrize("text", ["", "not json", "[1, 2]", "null", '{"format_version": 1'])
def test_garbage_is_refused(text: str) -> None:
    with pytest.raises(ArtifactError):
        loads(text)


def test_missing_file_is_an_artifact_error_not_an_oserror(tmp_path: Path) -> None:
    with pytest.raises(ArtifactError, match="cannot read"):
        load(tmp_path / "nope.json")


@given(z=st.floats(-1e6, 1e6, allow_nan=False))
def test_sigmoid_is_a_probability_and_never_overflows(z: float) -> None:
    p = sigmoid(z)
    assert 0.0 <= p <= 1.0


def test_score_is_monotone_in_the_logit() -> None:
    a = length_artifact()
    scalars = np.zeros(N_SCALARS)
    short, long_ = scalars.copy(), scalars.copy()
    short[0], long_[0] = 2.0, 6.0  # log_chars
    assert a.score(np.zeros(384), short) > a.score(np.zeros(384), long_)


def test_vectorized_scores_match_the_scalar_path() -> None:
    a = length_artifact()
    rng = np.random.default_rng(0)
    emb = rng.normal(size=(5, 384))
    sc = rng.normal(size=(5, N_SCALARS))
    vec = a.score_matrix(emb, sc)
    one_by_one = [a.score(emb[i], sc[i]) for i in range(5)]
    np.testing.assert_allclose(vec, one_by_one)
