"""Turns an ``Authorization`` header into a ``Tenant``, or into the right error.

    missing / malformed header   -> 401 invalid_api_key
    unknown or revoked key       -> 401 invalid_api_key
    key store down, key unknown  -> 503 auth_unavailable (the key may be fine; see store.py)
    valid key                    -> the tenant from config

Both 401 cases share one message, so a caller can't learn whether a key exists, is revoked,
or was never issued.
"""

import logging

from relay.auth.keys import hash_key, looks_like_key
from relay.auth.store import KeyStore, KeyStoreUnavailable
from relay.config.schema import RelayConfig
from relay.config.tenants import Tenant
from relay.schemas.errors import ErrorClass, RelayError

logger = logging.getLogger(__name__)

INVALID = "Invalid API key. Pass it as 'Authorization: Bearer rk_...'."


class Authenticator:
    def __init__(self, store: KeyStore, pepper: str, config: RelayConfig) -> None:
        self.store = store
        self._pepper = pepper
        self._config = config

    async def authenticate(self, authorization: str | None) -> Tenant:
        scheme, _, key = (authorization or "").partition(" ")
        key = key.strip()
        if scheme.lower() != "bearer" or not looks_like_key(key):
            raise RelayError(401, ErrorClass.INVALID_API_KEY, INVALID)
        try:
            tenant_id = await self.store.lookup(hash_key(key, self._pepper))
        except KeyStoreUnavailable:
            raise RelayError(
                503,
                ErrorClass.AUTH_UNAVAILABLE,
                "Authentication is temporarily unavailable; retry shortly.",
                retry_after_s=5,
            ) from None
        if tenant_id is None:
            raise RelayError(401, ErrorClass.INVALID_API_KEY, INVALID)
        tenant = self._config.tenants.get(tenant_id)
        if tenant is None:
            # A key whose tenant was removed from the config. An operator problem, but to the
            # caller it is just a key that no longer works.
            logger.warning("API key maps to unknown tenant %r", tenant_id)
            raise RelayError(401, ErrorClass.INVALID_API_KEY, INVALID)
        return tenant
