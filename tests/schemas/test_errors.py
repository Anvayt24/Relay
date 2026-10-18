import pytest

from relay.schemas.errors import (
    UPSTREAM_STATUS,
    ErrorClass,
    ProviderError,
    ProviderSignal,
    RelayError,
)

# Every provider-side class must have an HTTP status, or from_provider() would KeyError at
# request time. Gateway-side classes are constructed with an explicit status instead.
PROVIDER_CLASSES = [
    ErrorClass.RATE_LIMITED,
    ErrorClass.QUOTA_EXHAUSTED,
    ErrorClass.OVERLOADED,
    ErrorClass.TIMEOUT,
    ErrorClass.SERVER_ERROR,
    ErrorClass.BAD_REQUEST,
    ErrorClass.CONTEXT_LENGTH_EXCEEDED,
    ErrorClass.MODEL_NOT_FOUND,
    ErrorClass.AUTH,
    ErrorClass.CONTENT_FILTERED,
    ErrorClass.STREAM_BROKEN,
]


def test_every_provider_class_has_a_status() -> None:
    assert set(UPSTREAM_STATUS) == set(PROVIDER_CLASSES)


@pytest.mark.parametrize(
    ("error_class", "status"),
    [
        # Relay's credentials to the provider are broken, not the client's: never 401.
        (ErrorClass.AUTH, 502),
        # The provider dropped a model Relay is configured to use: Relay's problem, not 404.
        (ErrorClass.MODEL_NOT_FOUND, 502),
        (ErrorClass.SERVER_ERROR, 502),
        (ErrorClass.OVERLOADED, 503),
        (ErrorClass.TIMEOUT, 504),
        (ErrorClass.RATE_LIMITED, 429),
        (ErrorClass.QUOTA_EXHAUSTED, 429),
        (ErrorClass.CONTEXT_LENGTH_EXCEEDED, 400),
    ],
)
def test_upstream_status_mapping(error_class: ErrorClass, status: int) -> None:
    assert UPSTREAM_STATUS[error_class] == status


def test_from_provider_carries_class_and_retry_after() -> None:
    err = ProviderError(
        ErrorClass.RATE_LIMITED,
        "slow down",
        ProviderSignal(retry_after_s=2.5, limit_scope="model"),
    )
    relay_err = RelayError.from_provider(err)
    assert relay_err.status == 429
    assert relay_err.error_class is ErrorClass.RATE_LIMITED
    assert relay_err.retry_after_s == 2.5
    assert "slow down" in relay_err.message


def test_from_provider_without_signal() -> None:
    relay_err = RelayError.from_provider(ProviderError(ErrorClass.SERVER_ERROR, "boom"))
    assert relay_err.status == 502
    assert relay_err.retry_after_s is None
