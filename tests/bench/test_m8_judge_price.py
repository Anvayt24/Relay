"""M8 (what evidence costs) with a paid judge: Claude Sonnet 5.5 instead of the cheap model.

The simulator runs the traffic; only the judge's price changes. Sonnet's list price is $2 in /
$10 out per million tokens, and a judge's reason (rubric v2) is output the mock judge does not
produce, so it is added at a stated token count and priced at the output rate.
"""

from evals.evidence import cost

SONNET = {"price_in": 2.0, "price_out": 10.0}


def test_a_paid_judge_makes_evidence_cost_more_than_a_cheap_one() -> None:
    cheap = cost.run(n_requests=100)["undecided_rate"]
    paid = cost.measure_priced_judge(**SONNET, extra_output_tokens=0, sample_rate=0.05,
                                     n_requests=100, seed=0)  # fmt: skip
    assert paid["shadow_pairs"] == cheap["shadow_pairs"]  # same traffic, same sampling
    assert paid["routed_list_nano_usd"] == cheap["routed_list_nano_usd"]
    assert paid["evidence_list_nano_usd"] > cheap["evidence_list_nano_usd"]
    assert paid["share"] > cheap["share"]


def test_the_judges_written_reason_is_priced_at_the_output_rate() -> None:
    base = cost.measure_priced_judge(**SONNET, extra_output_tokens=0, sample_rate=0.05,
                                     n_requests=100, seed=0)  # fmt: skip
    with_reason = cost.measure_priced_judge(**SONNET, extra_output_tokens=80, sample_rate=0.05,
                                            n_requests=100, seed=0)  # fmt: skip
    # nano-USD per token = USD per million tokens x 1000.
    extra = with_reason["evidence_list_nano_usd"] - base["evidence_list_nano_usd"]
    assert extra == base["shadow_pairs"] * 80 * round(SONNET["price_out"] * 1000)


def test_the_run_reports_the_paid_judge_next_to_the_cheap_one() -> None:
    m8 = cost.run(n_requests=100)
    assert m8["judge"] == "mock/cheap"  # the gated figure is unchanged
    sonnet = m8["sonnet_judge"]
    assert {"undecided_rate", "decided_rate", "met", "price_in", "price_out"} <= set(sonnet)
    assert sonnet["extra_output_tokens"] > 0
