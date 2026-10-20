"""What ReservationStage held for one request, so its finalize can settle exactly that."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Reservation:
    res_id: str
    budget: str  # "tenant:YYYYMMDD": the store key of today's budget
    tpm_key: str
    tokens: int  # reserved against tenant TPM: estimated prompt + output cap
    prompt_estimate: int
    nano: int  # reserved against the daily budget (0 when nothing was reserved)
    held: bool  # False when no money was held (free model, or soft tenant over budget)
