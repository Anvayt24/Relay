import os
import subprocess
import sys

from relay.simulator.seeding import rng_for, uniform_for


def _draws(seed: int, *keys: str | int) -> list[float]:
    rng = rng_for(seed, *keys)
    return [rng.random() for _ in range(5)]


def test_same_keys_give_the_same_sequence() -> None:
    assert _draws(7, 3, "mock-strong-v1", "latency") == _draws(7, 3, "mock-strong-v1", "latency")


def test_any_different_key_gives_a_different_sequence() -> None:
    base = _draws(7, 3, "mock-strong-v1", "latency")
    assert _draws(8, 3, "mock-strong-v1", "latency") != base  # seed
    assert _draws(7, 4, "mock-strong-v1", "latency") != base  # request index
    assert _draws(7, 3, "mock-cheap-v1", "latency") != base  # model
    assert _draws(7, 3, "mock-strong-v1", "error") != base  # purpose (substream)


def test_keys_are_not_concatenated_ambiguously() -> None:
    # ("ab", "c") and ("a", "bc") must be different streams.
    assert _draws(0, "ab", "c") != _draws(0, "a", "bc")
    # 1 and "1" too.
    assert _draws(0, 1) != _draws(0, "1")


def test_stable_across_processes_with_different_hash_seeds() -> None:
    # Python salts str hashes per process. If rng_for used hash(), this would fail.
    code = (
        "from relay.simulator.seeding import rng_for; "
        "print(rng_for(7, 3, 'mock-strong-v1', 'latency').random())"
    )
    outputs = set()
    for hash_seed in ("1", "2"):
        env = {**os.environ, "PYTHONHASHSEED": hash_seed}
        result = subprocess.run(
            [sys.executable, "-c", code], env=env, capture_output=True, text=True, check=True
        )
        outputs.add(result.stdout.strip())
    assert outputs == {repr(_draws(7, 3, "mock-strong-v1", "latency")[0])}


def test_uniform_for_is_deterministic_and_in_range() -> None:
    draws = [uniform_for(1, i, "quality") for i in range(10_000)]
    assert draws == [uniform_for(1, i, "quality") for i in range(10_000)]
    assert all(0.0 <= u < 1.0 for u in draws)
    # Roughly uniform: each tenth of [0, 1) gets about 1,000 of 10,000 draws.
    for decile in range(10):
        in_bin = sum(decile / 10 <= u < (decile + 1) / 10 for u in draws)
        assert 850 < in_bin < 1150
