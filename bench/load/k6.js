// Constant-arrival-rate load: RATE requests/second for DURATION, whether or not earlier
// requests have finished (open model, like the chaos generator). TARGET is the base URL:
// the simulator directly, or Relay behind the load balancer.
import http from "k6/http";
import { check } from "k6";

const target = __ENV.TARGET;
const rate = parseInt(__ENV.RATE || "200");

export const options = {
  scenarios: {
    steady: {
      executor: "constant-arrival-rate",
      rate: rate,
      timeUnit: "1s",
      duration: __ENV.DURATION || "60s",
      preAllocatedVUs: 200,
      maxVUs: 2000,
    },
  },
  summaryTrendStats: ["avg", "min", "med", "p(90)", "p(95)", "p(99)", "max"],
};

// VARY=1 makes every prompt unique (and temperature 0, so it is cache-eligible): each request
// misses the exact cache, pays for an embedding and a vector lookup, then the provider call and
// the write-back. That is the heaviest path the cache adds (M1(b)).
const vary = __ENV.VARY === "1";
function body() {
  return JSON.stringify({
    model: __ENV.MODEL || "relay/auto",
    messages: [{ role: "user", content: vary ? `explain topic ${__VU}-${__ITER}` : "hello" }],
    max_tokens: 16,
    ...(vary ? { temperature: 0 } : {}),
  });
}
const params = { headers: { "Content-Type": "application/json", Authorization: "Bearer load" } };

export default function () {
  const res = http.post(`${target}/v1/chat/completions`, body(), params);
  check(res, { "status 200": (r) => r.status === 200 });
  // The first few failures, with their bodies: a load test that fails must say why.
  if (res.status !== 200 && __ITER < 3) {
    console.error(`status ${res.status} (${res.error_code}): ${res.error} ${String(res.body).slice(0, 200)}`);
  }
}
