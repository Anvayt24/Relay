"""The client address a per-IP limit counts against.

The socket peer is the one address a client cannot choose. ``X-Forwarded-For`` is a header
the client can set to anything, so it is read only when the peer is a proxy the operator
listed in ``RELAY_TRUSTED_PROXIES``, and then only the part that proxy appended: the
right-most entry that is not itself a trusted proxy. Everything to its left was typed by
someone else.
"""

import ipaddress
from collections.abc import Sequence

type Network = ipaddress.IPv4Network | ipaddress.IPv6Network

UNKNOWN = "unknown"  # no usable address: all such requests share one limit, none are exempt


def parse_trusted_proxies(raw: str) -> tuple[Network, ...]:
    """Comma-separated addresses or CIDR blocks. A typo raises: a proxy list that silently
    drops an entry would make every client behind that proxy look like one address."""
    networks: list[Network] = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        try:
            networks.append(ipaddress.ip_network(item, strict=False))
        except ValueError as exc:
            raise ValueError(f"RELAY_TRUSTED_PROXIES: {item!r} is not an address or CIDR") from exc
    return tuple(networks)


def _address(text: str) -> ipaddress.IPv4Address | ipaddress.IPv6Address | None:
    try:
        return ipaddress.ip_address(text.strip())
    except ValueError:
        return None


def _trusted(
    address: ipaddress.IPv4Address | ipaddress.IPv6Address, nets: Sequence[Network]
) -> bool:
    return any(address in net for net in nets)


def resolve_client_ip(
    peer: str | None, forwarded_for: str | None, trusted: Sequence[Network]
) -> str:
    peer_address = _address(peer) if peer else None
    if peer_address is None:
        return UNKNOWN
    if not trusted or not _trusted(peer_address, trusted) or not forwarded_for:
        return str(peer_address)
    for entry in reversed(forwarded_for.split(",")):
        address = _address(entry)
        if address is None:
            return str(peer_address)  # a malformed entry: believe nothing it was beside
        if not _trusted(address, trusted):
            return str(address)
    return str(peer_address)
