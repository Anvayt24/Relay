import math

import pytest

from relay.config.loader import load_config
from relay.schemas.errors import ErrorClass
from relay.simulator.quality import QualitySpec, WinRate
from relay.simulator.scenario import (
    PRESETS,
    Capacity,
    ErrorMix,
    Incident,
    LatencyModel,
    ModelProfile,
    OutputLength,
    Scenario,
    lognormal_params,
)


def test_lognormal_params_from_median_and_p95() -> None:
    mu, sigma = lognormal_params(0.5, 1.5)
    assert math.exp(mu) == pytest.approx(0.5)
    assert math.exp(mu + 1.6448536 * sigma) == pytest.approx(1.5)


@pytest.mark.parametrize(("median", "p95"), [(0, 1), (-1, 1), (2, 1)])
def test_lognormal_params_rejects_bad_inputs(median: float, p95: float) -> None:
    with pytest.raises(ValueError):
        lognormal_params(median, p95)


def test_latency_and_length_validate() -> None:
    with pytest.raises(ValueError):
        LatencyModel(ttft_median_s=1.0, ttft_p95_s=0.5)
    with pytest.raises(ValueError):
        LatencyModel(tokens_per_s=0)
    with pytest.raises(ValueError):
        OutputLength(median_tokens=10, p95_tokens=5)
    LatencyModel()  # 0/0 = instant is allowed


def test_error_mix_validates_probabilities() -> None:
    with pytest.raises(ValueError, match="sum"):
        ErrorMix(rate_limited=0.6, server_error=0.5)
    with pytest.raises(ValueError):
        ErrorMix(timeout=-0.1)
    ErrorMix(overloaded=0.7, server_error=0.2, timeout=0.1)  # float sum 1.0000000000000002


def test_error_mix_pick_slices_the_unit_interval() -> None:
    mix = ErrorMix(rate_limited=0.1, server_error=0.2)
    assert mix.pick(0.05) is ErrorClass.RATE_LIMITED
    assert mix.pick(0.15) is ErrorClass.SERVER_ERROR
    assert mix.pick(0.29) is ErrorClass.SERVER_ERROR
    # Not 0.30: in floats 0.1 + 0.2 is 0.30000000000000004, so 0.30 is still inside.
    assert mix.pick(0.31) is None
    assert ErrorMix().pick(0.0) is None


def test_capacity_and_incident_validate() -> None:
    with pytest.raises(ValueError):
        Capacity(rpm=0, tpm=10)
    with pytest.raises(ValueError):
        Incident(start_s=5, end_s=5)
    with pytest.raises(ValueError):
        Incident(start_s=0, end_s=5, latency_multiplier=0)


def test_incident_applies_only_inside_its_window_and_to_its_models() -> None:
    incident = Incident(start_s=10, end_s=20, models=("a",))
    assert not incident.applies("a", 9.99)
    assert incident.applies("a", 10)
    assert not incident.applies("a", 20)  # end is exclusive
    assert not incident.applies("b", 15)
    assert Incident(start_s=0, end_s=1).applies("anything", 0.5)  # no models = all


def test_win_rate_is_solved_against_the_reference() -> None:
    scenario = Scenario(
        models={
            "strong": ModelProfile(quality=QualitySpec(base=0.9)),
            "cheap": ModelProfile(quality=QualitySpec(base=WinRate("strong", 0.93))),
        }
    )
    assert scenario.q("cheap", "default", 0) == pytest.approx(1 - 0.07 / 0.9)


def test_infeasible_win_rate_fails_at_construction() -> None:
    # The reference is right only 5% of the time, so the cheap model cannot lose 10%.
    with pytest.raises(ValueError, match="infeasible"):
        Scenario(
            models={
                "strong": ModelProfile(quality=QualitySpec(base=0.05)),
                "cheap": ModelProfile(quality=QualitySpec(base=WinRate("strong", 0.90))),
            }
        )


def test_infeasibility_after_a_shift_also_fails_at_construction() -> None:
    with pytest.raises(ValueError, match="infeasible"):
        Scenario(
            models={
                "strong": ModelProfile(quality=QualitySpec(base=0.9, shifts=[(100, 0.05)])),
                "cheap": ModelProfile(quality=QualitySpec(base=WinRate("strong", 0.90))),
            }
        )


def test_win_rate_reference_must_exist_and_not_cycle() -> None:
    with pytest.raises(ValueError, match="not a scenario model"):
        Scenario(models={"a": ModelProfile(quality=QualitySpec(base=WinRate("zzz", 0.9)))})
    with pytest.raises(ValueError, match="cycle"):
        Scenario(
            models={
                "a": ModelProfile(quality=QualitySpec(base=WinRate("b", 0.9))),
                "b": ModelProfile(quality=QualitySpec(base=WinRate("a", 0.9))),
            }
        )


def test_cohort_quality_overrides_the_default() -> None:
    scenario = Scenario(
        models={
            "m": ModelProfile(
                quality=QualitySpec(base=0.9), cohort_quality={"lang=hi": QualitySpec(base=0.6)}
            )
        }
    )
    assert scenario.q("m", "default", 0) == 0.9
    assert scenario.q("m", "lang=hi", 0) == 0.6
    assert scenario.q("m", "unknown-cohort", 0) == 0.9


def test_presets_cover_the_mock_models_in_the_registry() -> None:
    # Relay sends provider_model ids to the mock adapter. If a preset lacked one, every
    # request for that model would fail with model_not_found under RELAY_SIM_SCENARIO.
    shipped = load_config()  # relay/config/relay.yaml
    mock_models = {s.provider_model for s in shipped.models.values() if s.provider == "mock"}
    for name, scenario in PRESETS.items():
        assert mock_models <= set(scenario.models), name
