import assert from "node:assert/strict";
import { test } from "node:test";
import {
  decisionFromHeaders,
  duration,
  percent,
  routerSentence,
  savedFraction,
  chartMax,
  linePoints,
  usd,
  withInterval,
} from "../dist/format.js";

test("percent and duration say what they measure, with units", () => {
  assert.equal(percent(0.9312), "93.1%");
  assert.equal(percent(null), "—");
  assert.equal(duration(0.0421), "42 ms");
  assert.equal(duration(1.5), "1.50 s");
  assert.equal(duration(undefined), "—");
});

test("money keeps the precision a single request needs", () => {
  assert.equal(usd(0), "$0");
  assert.equal(usd(0.0000123), "$0.000012");
  assert.equal(usd(0.0123), "$0.0123");
  assert.equal(usd(null), "—");
});

test("a statistic is written with its interval", () => {
  assert.equal(withInterval({ value: 0.612, ci_low: 0.539, ci_high: 0.688 }), "0.612 [0.539, 0.688]");
});

test("the decision is read from the response headers, and a missing one stays missing", () => {
  const headers = new Headers({
    "x-relay-cache": "miss",
    "x-relay-router-score": "0.8765",
    "x-relay-model": "mock/cheap",
    "x-relay-fallback-hops": "1",
    "x-relay-latency-ms": "42",
    "x-relay-cost-usd": "0.000010",
    "x-relay-strong-cost-usd": "0.000050",
  });
  const d = decisionFromHeaders(headers, 0.82);
  assert.equal(d.cache, "miss");
  assert.equal(d.routerScore, 0.8765);
  assert.equal(d.fallbackHops, 1);
  assert.equal(d.provider, null);
  assert.equal(decisionFromHeaders(new Headers(), null).routerScore, null);
});

test("savings are relative to always using the strong model, and never negative", () => {
  assert.equal(savedFraction({ costUsd: 0.00001, strongCostUsd: 0.00005 }), 0.8);
  assert.equal(savedFraction({ costUsd: 0.0001, strongCostUsd: 0.00005 }), 0);
  assert.equal(savedFraction({ costUsd: null, strongCostUsd: 0.00005 }), null);
  assert.equal(savedFraction({ costUsd: 0.1, strongCostUsd: 0 }), null);
});

test("the router sentence puts the score against its threshold", () => {
  assert.equal(
    routerSentence({ routerScore: 0.9, routerThreshold: 0.82 }),
    "0.9000 ≥ 0.82 → cheap model",
  );
  assert.equal(
    routerSentence({ routerScore: 0.5, routerThreshold: 0.82 }),
    "0.5000 < 0.82 → strong model",
  );
  assert.equal(routerSentence({ routerScore: null, routerThreshold: 0.82 }), "no router on this route");
});

test("a line needs two points, skips gaps, and keeps the extremes off the frame", () => {
  assert.equal(linePoints([null, 1], 100, 20, 10), "");
  assert.equal(linePoints([10, 0, 10], 100, 20, 10), "0.0,4.0 50.0,16.0 100.0,4.0");
  assert.equal(linePoints([1, null, 1], 100, 20, 1), "0.0,4.0 100.0,4.0");
  assert.equal(linePoints([5, 50], 100, 20, 10), "0.0,10.0 100.0,4.0"); // above the top is clamped
});

test("a chart is as tall as its tallest series, rounded up, and never below the floor", () => {
  assert.equal(chartMax([[0.4, null], [0.9]]), 1);
  assert.equal(chartMax([[2.5, 1], [14.2]]), 15);
  assert.equal(chartMax([[null], []], 3), 3);
});

test("a router that could not decide says why, instead of looking like no router", () => {
  assert.equal(
    routerSentence({ routerScore: null, routerThreshold: 0.82, routerReason: "router_fallback:embedder_not_ready" }),
    "strong model: the router could not decide (embedder_not_ready)",
  );
  const d = decisionFromHeaders(new Headers({ "x-relay-router-reason": "router" }), null);
  assert.equal(d.routerReason, "router");
});
