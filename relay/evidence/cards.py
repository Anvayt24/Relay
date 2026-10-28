"""Route cards: what the evidence says about one comparison, per cohort (PRD §10.8).

A card answers "is the candidate good enough?" for one comparison on one route:

- ``cheap_vs_strong``: answers the router sent cheap, against the strong model;
- ``fallback:<src>-><dst>``: a fallback edge's target, against a healthy primary;
- ``band:<lo>-<hi>``: the cheap model on the strong-routed requests whose router score is in
  the band a lower threshold would move (the only evidence for a promotion, §10.2 item 5);
- ``canary:<version>``: answers served under a canary bundle (Phase 13).

**How a card is computed** (``compute_cards``):

1. One judge only: pairs from other judge models or prompt versions are excluded (ADR 0014).
2. Judge calibration: the M7 gate (kappa >= 0.6 and loss recall >= 0.7 on >= 50 human labels,
   ``calibrate.py``) and n_human >= 50 for this comparison, else the card is ``uncalibrated``.
   The rectifier is ``stale`` when its newest human label is too old (ADR 0021).
3. Looks: the decision uses the first ``look * every`` judged pairs in time order, and a
   decided state changes only when a new look is reached (ADR 0019).
4. Estimate: PPI++ per cohort, with the **pooled** rectifier (all human labels of the
   comparison) and that cohort's judge-only pairs (ADR 0021). Small cohorts merge into
   ``other``; the ``all`` card covers everything.
5. Decision at the look's alpha: Holm across cohorts for ``safe``, Benjamini-Hochberg for
   ``unsafe`` (ADR 0020). The ``all`` card is tested on its own.
6. Action (§10.7): disable an unsafe edge, roll back an unsafe canary, recommend promoting a
   band only when every cohort is safe. Nothing automatic when uncalibrated or stale.

Labels here are "win or tie" = 1 (``w`` is the win-or-tie rate), unlike the calibration
module's loss-positive convention.
"""

import math
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timedelta
from typing import Any, Literal

import numpy as np
from scipy.stats import norm

from relay.evidence import stats
from relay.evidence.calibrate import KAPPA_MIN, LOSS_RECALL_MIN, MIN_N
from relay.evidence.looks import LookSchedule

State = Literal["safe", "unsafe", "undecided", "uncalibrated", "stale"]
Interval = Literal["wald", "bootstrap"]
ALL = "all"
OTHER = "other"


@dataclass(frozen=True)
class JudgedPair:
    """What the engine needs from one ``eval_pairs`` row, joined with its first human label."""

    pair_id: str
    route: str
    comparison: str
    cohort: str
    judge_label: str | None  # win | tie | loss | None (abstained)
    judge_model: str
    judge_prompt_version: str
    ts: datetime
    policy_version: str | None = None
    human_label: str | None = None  # the first (non-relabel) human label, if any
    human_ts: datetime | None = None

    @property
    def judge_ok(self) -> bool | None:
        return None if self.judge_label is None else self.judge_label != "loss"

    @property
    def human_ok(self) -> bool | None:
        return None if self.human_label is None else self.human_label != "loss"


@dataclass(frozen=True)
class CardSpec:
    floor: float = 0.90
    looks: LookSchedule = field(default_factory=LookSchedule)
    min_cohort_n: int = 200
    rectifier_max_age: timedelta = timedelta(days=30)
    interval: Interval = "wald"
    n_boot: int = 2000


@dataclass(frozen=True)
class JudgeCalibration:
    judge: str  # "model@prompt_version"
    n: int
    kappa: float
    loss_recall: float

    @property
    def calibrated(self) -> bool:
        return (
            self.n >= MIN_N
            and not math.isnan(self.kappa)
            and not math.isnan(self.loss_recall)
            and self.kappa >= KAPPA_MIN
            and self.loss_recall >= LOSS_RECALL_MIN
        )


@dataclass(frozen=True)
class RouteCard:
    route: str
    cohort: str
    comparison: str
    estimate: float | None
    ci: tuple[float, float] | None  # one-sided bounds at the look's alpha, each side
    floor: float
    n_judge: int  # judged pairs available (the decision may use fewer: see look)
    n_human: int
    judge: str
    judge_kappa: float | None
    state: State
    recommended_action: str
    look: int
    max_looks: int
    policy_version: str | None
    updated_at: datetime
    metric: str = "win_or_tie_rate"
    rectifier_scope: str = "pooled"
    interval: Interval = "wald"
    reason: str | None = None
    samples_needed: int | None = None
    lam: float | None = None
    ess_gain: float | None = None
    n_excluded: int = 0  # pairs from another judge, or abstentions
    w_self: float | None = None
    cost_per_1k_requests: dict[str, float] | None = None

    @property
    def key(self) -> tuple[str, str, str]:
        return (self.route, self.comparison, self.cohort)

    def as_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["ci"] = list(self.ci) if self.ci else None
        out["updated_at"] = self.updated_at.isoformat()
        return out


def calibration_by_judge(pairs: Iterable[JudgedPair]) -> dict[str, JudgeCalibration]:
    """The M7 numbers per judge, from every human-labeled pair it judged (any comparison:
    calibration is a property of the judge, not of one route)."""
    human: dict[str, list[bool]] = {}
    judge: dict[str, list[bool]] = {}
    for p in pairs:
        if p.human_label is None or p.judge_label is None:
            continue
        key = judge_key(p)
        human.setdefault(key, []).append(p.human_label == "loss")
        judge.setdefault(key, []).append(p.judge_label == "loss")
    out = {}
    for key, h in human.items():
        c = stats.confusion(h, judge[key])
        out[key] = JudgeCalibration(key, c.n, stats.cohen_kappa(c), stats.loss_recall(c))
    return out


def judge_key(p: JudgedPair) -> str:
    return f"{p.judge_model}@{p.judge_prompt_version}"


def compute_cards(
    pairs: Sequence[JudgedPair],
    spec: CardSpec,
    *,
    calibration: Mapping[str, JudgeCalibration],
    now: datetime,
    previous: Mapping[tuple[str, str, str], RouteCard] | None = None,
) -> list[RouteCard]:
    """Cards for every (route, comparison) in ``pairs``. ``previous`` holds the cards from the
    last run: a decided state only changes when a new look is reached."""
    groups: dict[tuple[str, str], list[JudgedPair]] = {}
    for p in pairs:
        groups.setdefault((p.route, p.comparison), []).append(p)
    cards: list[RouteCard] = []
    for (route, comparison), group in sorted(groups.items()):
        cards += _comparison_cards(route, comparison, group, spec, calibration, now, previous or {})
    return cards


# ------------------------------------------------------------------ one comparison
def _comparison_cards(
    route: str,
    comparison: str,
    pairs: list[JudgedPair],
    spec: CardSpec,
    calibration: Mapping[str, JudgeCalibration],
    now: datetime,
    previous: Mapping[tuple[str, str, str], RouteCard],
) -> list[RouteCard]:
    judges = Counter(judge_key(p) for p in pairs)
    judge, _ = judges.most_common(1)[0]
    mine = [p for p in pairs if judge_key(p) == judge]
    judged = sorted((p for p in mine if p.judge_label is not None), key=lambda p: p.ts)
    excluded = len(pairs) - len(judged)

    labeled = [p for p in judged if p.human_label is not None]
    unlabeled = [p for p in judged if p.human_label is None]
    look = spec.looks.look_for(len(unlabeled))
    decision_data = unlabeled[: spec.looks.sample_size(look)]
    policy = Counter(p.policy_version for p in decision_data or unlabeled).most_common(1)
    base = {
        "route": route,
        "comparison": comparison,
        "floor": spec.floor,
        "n_judge": len(unlabeled),
        "n_human": len(labeled),
        "judge": judge,
        "look": look,
        "max_looks": spec.looks.max_looks,
        "policy_version": policy[0][0] if policy else None,
        "updated_at": now,
        "interval": spec.interval,
        "n_excluded": excluded,
    }
    cal = calibration.get(judge)
    base["judge_kappa"] = None if cal is None or math.isnan(cal.kappa) else cal.kappa

    # Non-decision states first: they hold whatever the data says, at any time.
    blocked: tuple[State, str] | None = None
    if cal is None or not cal.calibrated:
        blocked = ("uncalibrated", "judge has not passed the M7 gate")
    elif len(labeled) < stats.MIN_PPI_LABELED:
        blocked = ("uncalibrated", f"needs {stats.MIN_PPI_LABELED} human labels")
    else:
        newest = max(p.human_ts or p.ts for p in labeled)
        if now - newest > spec.rectifier_max_age:
            blocked = ("stale", "newest human label is older than the rectifier max age")
    if blocked is not None:
        return [_judge_only_card(base, unlabeled, blocked)]

    if look == 0:
        # No look reached yet: report progress, decide nothing.
        return [_progress_card(base, unlabeled, spec)]

    y = np.array([p.human_ok for p in labeled], dtype=bool)
    f = np.array([p.judge_ok for p in labeled], dtype=bool)
    alpha = spec.looks.alpha_per_look
    cohort_groups = _cohorts(decision_data, spec.min_cohort_n)
    estimates = {
        name: _estimate(np.array([p.judge_ok for p in ps], dtype=bool), y, f, spec, alpha)
        for name, ps in cohort_groups.items()
    }
    all_est = _estimate(
        np.array([p.judge_ok for p in decision_data], dtype=bool), y, f, spec, alpha
    )
    pooling_failed = pooling_failures(labeled, alpha)

    names = list(estimates)
    safe = stats.holm([estimates[n]["p_safe"] for n in names], alpha)
    unsafe = stats.benjamini_hochberg([estimates[n]["p_unsafe"] for n in names], alpha)
    states: dict[str, State] = {}
    reasons: dict[str, str | None] = {}
    for name, is_safe, is_unsafe in zip(names, safe, unsafe, strict=True):
        states[name] = "unsafe" if is_unsafe else "safe" if is_safe else "undecided"
        reasons[name] = "multiplicity-adjusted (Holm / BH)" if len(names) > 1 else None
        if name in pooling_failed and states[name] == "safe":
            states[name], reasons[name] = "undecided", "rectifier_pooling_failed"
    all_state: State = (
        "unsafe"
        if all_est["p_unsafe"] <= alpha
        else "safe"
        if all_est["p_safe"] <= alpha
        else "undecided"
    )

    every_cohort_safe = all(s == "safe" for s in states.values())
    cards = [
        _decided_card(
            base,
            ALL,
            all_est,
            all_state,
            None,
            len(decision_data),
            spec,
            previous,
            every_cohort_safe,
        )
    ]
    for name in names:
        cards.append(
            _decided_card(
                base,
                name,
                estimates[name],
                states[name],
                reasons[name],
                len(cohort_groups[name]),
                spec,
                previous,
                every_cohort_safe,
            )
        )
    return cards


def _estimate(
    f_unlabeled: np.ndarray, y: np.ndarray, f: np.ndarray, spec: CardSpec, alpha: float
) -> dict[str, Any]:
    # Two-sided at 2*alpha: each side is a one-sided bound at the look's alpha.
    ppi = stats.ppi_mean_ci(f_unlabeled, y, f, alpha=2 * alpha)
    out: dict[str, Any] = {
        "estimate": ppi.estimate,
        "se": ppi.se,
        "lam": ppi.lam,
        "ess_gain": ppi.ess_gain,
        "n": len(f_unlabeled),
    }
    if spec.interval == "bootstrap":
        lo, hi, reps = stats.ppi_bootstrap_ci(
            f_unlabeled, y, f, alpha=2 * alpha, n_boot=spec.n_boot, seed=0
        )
        out["ci"] = (lo, hi)
        out["p_safe"] = (1 + int(np.sum(reps <= spec.floor))) / (len(reps) + 1)
        out["p_unsafe"] = (1 + int(np.sum(reps >= spec.floor))) / (len(reps) + 1)
    else:
        out["ci"] = (ppi.ci_low, ppi.ci_high)
        out["p_safe"] = stats.p_safe(ppi.estimate, ppi.se, spec.floor)
        out["p_unsafe"] = stats.p_unsafe(ppi.estimate, ppi.se, spec.floor)
    return out


def _cohorts(pairs: Sequence[JudgedPair], min_n: int) -> dict[str, list[JudgedPair]]:
    """Cohort -> pairs, with every cohort below ``min_n`` merged into ``other`` (§10.6)."""
    by: dict[str, list[JudgedPair]] = {}
    for p in pairs:
        by.setdefault(p.cohort, []).append(p)
    out: dict[str, list[JudgedPair]] = {}
    for name, ps in sorted(by.items()):
        out.setdefault(name if len(ps) >= min_n else OTHER, []).extend(ps)
    return out


def pooling_failures(labeled: Sequence[JudgedPair], alpha: float) -> set[str]:
    """Cohorts whose judge bias differs from the rest (ADR 0021): a two-sample z-test of the
    mean residual y - f, only where a cohort has >= 50 human labels, Holm-adjusted."""
    residual = {
        p.pair_id: float(p.human_label != "loss") - float(p.judge_label != "loss") for p in labeled
    }
    by: dict[str, list[float]] = {}
    for p in labeled:
        by.setdefault(p.cohort, []).append(residual[p.pair_id])
    tested, pvalues = [], []
    for name, inside in by.items():
        outside = [residual[p.pair_id] for p in labeled if p.cohort != name]
        if len(inside) < stats.MIN_PPI_LABELED or len(outside) < 2:
            continue
        a, b = np.array(inside), np.array(outside)
        se = math.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
        if se == 0:
            continue
        z = (a.mean() - b.mean()) / se
        tested.append(name)
        pvalues.append(float(2 * norm.sf(abs(z))))
    return {name for name, bad in zip(tested, stats.holm(pvalues, alpha), strict=True) if bad}


# ------------------------------------------------------------------ building cards
def _judge_only_card(
    base: dict[str, Any], unlabeled: Sequence[JudgedPair], blocked: tuple[State, str]
) -> RouteCard:
    # Shown for information only: a judge-only rate is biased by the judge's errors, which is
    # exactly why it may not drive anything.
    est = float(np.mean([p.judge_ok for p in unlabeled])) if unlabeled else None
    state, reason = blocked
    return RouteCard(
        **base,
        cohort=ALL,
        estimate=est,
        ci=None,
        state=state,
        reason=f"{reason}; estimate is judge-only",
        recommended_action="none",
    )


def _progress_card(
    base: dict[str, Any], unlabeled: Sequence[JudgedPair], spec: CardSpec
) -> RouteCard:
    need = spec.looks.every - len(unlabeled)
    return RouteCard(
        **base,
        cohort=ALL,
        estimate=None,
        ci=None,
        state="undecided",
        reason=f"first look at {spec.looks.every} judged pairs",
        recommended_action="collect_more",
        samples_needed=max(0, need),
    )


def _decided_card(
    base: dict[str, Any],
    cohort: str,
    est: dict[str, Any],
    state: State,
    reason: str | None,
    n: int,
    spec: CardSpec,
    previous: Mapping[tuple[str, str, str], RouteCard],
    every_cohort_safe: bool,
) -> RouteCard:
    prev = previous.get((base["route"], base["comparison"], cohort))
    if (
        prev is not None
        and prev.look == base["look"]
        and prev.state
        in (
            "safe",
            "unsafe",
            "undecided",
        )
    ):
        # Same look as last time: the decision was already taken on this data. New human
        # labels must not turn one look into two (ADR 0019).
        state, reason = prev.state, prev.reason
    needed = None
    if state == "undecided":
        if spec.looks.is_final(base["look"]):
            reason = "no decision after the last pre-registered look"
        else:
            needed = stats.samples_needed(
                est["estimate"], est["se"], n, spec.floor, spec.looks.alpha_per_look
            )
    return RouteCard(
        **base,
        cohort=cohort,
        estimate=est["estimate"],
        ci=est["ci"],
        state=state,
        reason=reason,
        recommended_action=action(base["comparison"], state, every_cohort_safe and cohort == ALL),
        samples_needed=needed,
        lam=est["lam"],
        ess_gain=est["ess_gain"],
    )


def action(comparison: str, state: State, promotable: bool) -> str:
    """§10.7, as a pure function of the comparison kind and the state."""
    kind = comparison.split(":", 1)[0]
    if state in ("uncalibrated", "stale"):
        return "none"
    if state == "undecided":
        return "collect_more"
    if state == "unsafe":
        return {
            "fallback": "disable_edge",
            "canary": "rollback",
            "band": "do_not_promote",
            "cheap_vs_strong": "raise_threshold",
        }.get(kind, "alert")
    # safe
    if kind == "band":
        return "promote" if promotable else "collect_more"
    if kind == "canary":
        return "promote" if promotable else "collect_more"
    return "none"


def with_cost(card: RouteCard, cost: Mapping[str, float] | None) -> RouteCard:
    return replace(card, cost_per_1k_requests=dict(cost) if cost else None)


def render_markdown(cards: Sequence[RouteCard]) -> str:
    lines = [
        "| route | comparison | cohort | state | estimate | ci | floor | n judge | n human "
        "| look | action |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for c in cards:
        est = f"{c.estimate:.3f}" if c.estimate is not None else "-"
        ci = f"[{c.ci[0]:.3f}, {c.ci[1]:.3f}]" if c.ci else "-"
        act = c.recommended_action
        if c.samples_needed:
            act += f" (~{c.samples_needed} more)"
        lines.append(
            f"| {c.route} | {c.comparison} | {c.cohort} | {c.state} | {est} | {ci} | {c.floor} "
            f"| {c.n_judge} | {c.n_human} | {c.look}/{c.max_looks} | {act} |"
        )
    return "\n".join(lines)
