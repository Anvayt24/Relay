import pytest

from relay.auth.keys import display_prefix, generate_key, hash_key, looks_like_key
from relay.auth.store import CachedKeyStore, KeyStoreUnavailable, MemoryKeyStore


def test_keys_are_random_and_well_formed() -> None:
    keys = {generate_key() for _ in range(100)}
    assert len(keys) == 100
    for key in keys:
        assert key.startswith("rk_")
        assert looks_like_key(key)
        assert len(key) == 3 + 43  # 32 bytes, url-safe base64 without padding
    assert display_prefix("rk_abcdefgh123") == "rk_abcdefgh"


def test_hash_depends_on_the_pepper() -> None:
    key = generate_key()
    assert hash_key(key, "p1") == hash_key(key, "p1")
    # A different pepper invalidates every stored hash: rotating it revokes all keys.
    assert hash_key(key, "p1") != hash_key(key, "p2")
    assert len(hash_key(key, "p1")) == 32  # SHA-256


def test_shape_check_rejects_garbage() -> None:
    assert not looks_like_key("Bearer x")
    assert not looks_like_key("sk-openai-style-key-123456")
    assert not looks_like_key("rk_short")


class Clock:
    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t


@pytest.fixture
def clock() -> Clock:
    return Clock()


async def test_fresh_entries_skip_the_store(clock: Clock) -> None:
    inner = MemoryKeyStore({b"h": "t1"})
    cached = CachedKeyStore(inner, positive_ttl_s=300, clock=clock)
    assert await cached.lookup(b"h") == "t1"
    assert await cached.lookup(b"h") == "t1"
    assert inner.lookups == 1
    clock.t = 301
    await cached.lookup(b"h")
    assert inner.lookups == 2  # expired: asked again (picks up a revocation)


async def test_negative_entries_are_cached_briefly(clock: Clock) -> None:
    inner = MemoryKeyStore({})
    cached = CachedKeyStore(inner, negative_ttl_s=10, clock=clock)
    for _ in range(5):
        assert await cached.lookup(b"bad") is None
    assert inner.lookups == 1  # a bad key can't hammer the database
    clock.t = 11
    await cached.lookup(b"bad")
    assert inner.lookups == 2


async def test_outage_serves_known_keys_and_refuses_unknown_ones(clock: Clock) -> None:
    inner = MemoryKeyStore({b"known": "t1"})
    cached = CachedKeyStore(inner, positive_ttl_s=300, stale_if_error_s=3600, clock=clock)
    await cached.lookup(b"known")
    inner.down = True
    clock.t = 1000  # past the fresh TTL, within stale-if-error
    assert await cached.lookup(b"known") == "t1"
    with pytest.raises(KeyStoreUnavailable):
        await cached.lookup(b"never-seen")  # can't tell, so don't guess
    clock.t = 5000  # past stale-if-error too
    with pytest.raises(KeyStoreUnavailable):
        await cached.lookup(b"known")


async def test_a_negative_entry_is_not_served_during_an_outage(clock: Clock) -> None:
    inner = MemoryKeyStore({})
    cached = CachedKeyStore(inner, negative_ttl_s=10, clock=clock)
    await cached.lookup(b"k")
    inner.down = True
    clock.t = 20
    with pytest.raises(KeyStoreUnavailable):
        await cached.lookup(b"k")
