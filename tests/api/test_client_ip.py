"""Which address a per-IP limit counts a request against.

The socket peer is the only address a client cannot choose. `X-Forwarded-For` is believed only
when the peer is a proxy the operator listed, and then only the part that proxy added.
"""

import ipaddress

from hypothesis import given
from hypothesis import strategies as st

from relay.api.client_ip import parse_trusted_proxies, resolve_client_ip

TRUSTED = parse_trusted_proxies("10.0.0.0/8, 192.168.1.5")


def test_without_trusted_proxies_the_peer_is_the_client() -> None:
    assert resolve_client_ip("203.0.113.9", "6.6.6.6", ()) == "203.0.113.9"


def test_a_trusted_proxy_vouches_for_the_address_it_saw() -> None:
    # "6.6.6.6" was typed by the client; "198.51.100.7" is what the proxy saw.
    assert resolve_client_ip("10.1.2.3", "6.6.6.6, 198.51.100.7", TRUSTED) == "198.51.100.7"


def test_a_chain_of_trusted_proxies_is_walked_from_the_right() -> None:
    header = "6.6.6.6, 198.51.100.7, 10.9.9.9"
    assert resolve_client_ip("192.168.1.5", header, TRUSTED) == "198.51.100.7"


def test_a_peer_that_is_not_trusted_cannot_name_its_own_address() -> None:
    assert resolve_client_ip("203.0.113.9", "198.51.100.7", TRUSTED) == "203.0.113.9"


def test_garbage_falls_back_to_the_peer_and_a_missing_peer_to_one_shared_key() -> None:
    assert resolve_client_ip("10.1.2.3", "not-an-ip", TRUSTED) == "10.1.2.3"
    assert resolve_client_ip("10.1.2.3", None, TRUSTED) == "10.1.2.3"
    assert resolve_client_ip(None, "198.51.100.7", TRUSTED) == "unknown"


def test_parsing_ignores_blanks_and_rejects_nonsense() -> None:
    assert parse_trusted_proxies("") == ()
    assert len(parse_trusted_proxies(" 10.0.0.0/8 , ,::1 ")) == 2
    try:
        parse_trusted_proxies("not-a-network")
    except ValueError as exc:
        assert "not-a-network" in str(exc)
    else:
        raise AssertionError("an invalid proxy list must not be silently dropped")


@given(
    peer=st.ip_addresses().map(str).filter(lambda p: not ipaddress.ip_address(p).is_private),
    forwarded=st.text(max_size=80),
)
def test_a_client_never_chooses_its_address_unless_a_trusted_proxy_forwards_it(
    peer: str, forwarded: str
) -> None:
    assert resolve_client_ip(peer, forwarded, TRUSTED) == peer
