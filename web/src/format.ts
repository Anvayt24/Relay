// Plain logic, no DOM: how numbers, money and decisions are written on the page. Node tests it.

/** A fraction as a percentage with one decimal ("93.0%"), or an em dash when there is none. */
export function percent(fraction: number | null | undefined): string {
  return fraction === null || fraction === undefined ? "—" : `${(fraction * 100).toFixed(1)}%`;
}

/** Seconds as the unit a person reads: milliseconds under a second, seconds above. */
export function duration(seconds: number | null | undefined): string {
  if (seconds === null || seconds === undefined) return "—";
  return seconds < 1 ? `${Math.round(seconds * 1000)} ms` : `${seconds.toFixed(2)} s`;
}

/** Dollars written with the precision a single request needs (a request costs fractions of a cent). */
export function usd(dollars: number | null | undefined): string {
  if (dollars === null || dollars === undefined) return "—";
  if (dollars === 0) return "$0";
  return dollars < 0.01 ? `$${dollars.toFixed(6)}` : `$${dollars.toFixed(4)}`;
}

/** A statistic with its confidence interval: "0.612 [0.539, 0.688]". */
export function withInterval(v: { value: number; ci_low: number; ci_high: number }): string {
  return `${v.value.toFixed(3)} [${v.ci_low.toFixed(3)}, ${v.ci_high.toFixed(3)}]`;
}

export interface Decision {
  cache: string | null;
  routerScore: number | null;
  routerReason: string | null;
  routerThreshold: number | null;
  provider: string | null;
  model: string | null;
  fallbackHops: number | null;
  latencyMs: number | null;
  costUsd: number | null;
  strongCostUsd: number | null;
}

/** What Relay decided, read from the X-Relay-* response headers. */
export function decisionFromHeaders(headers: Headers, routerThreshold: number | null): Decision {
  const num = (name: string): number | null => {
    const raw = headers.get(name);
    if (raw === null || raw === "") return null;
    const value = Number(raw);
    return Number.isFinite(value) ? value : null;
  };
  return {
    cache: headers.get("x-relay-cache"),
    routerScore: num("x-relay-router-score"),
    routerReason: headers.get("x-relay-router-reason"),
    routerThreshold,
    provider: headers.get("x-relay-provider"),
    model: headers.get("x-relay-model"),
    fallbackHops: num("x-relay-fallback-hops"),
    latencyMs: num("x-relay-latency-ms"),
    costUsd: num("x-relay-cost-usd"),
    strongCostUsd: num("x-relay-strong-cost-usd"),
  };
}

/** How much cheaper than always using the strong model, as a fraction (null when unknown). */
export function savedFraction(d: Pick<Decision, "costUsd" | "strongCostUsd">): number | null {
  if (d.costUsd === null || d.strongCostUsd === null || d.strongCostUsd <= 0) return null;
  return Math.max(0, 1 - d.costUsd / d.strongCostUsd);
}

/** The router's choice, in words: the score against the threshold it was held to. */
export function routerSentence(
  d: Pick<Decision, "routerScore" | "routerThreshold"> & { routerReason?: string | null },
): string {
  if (d.routerScore === null) {
    const reason = d.routerReason ?? null;
    if (reason !== null && reason.startsWith("router_fallback:")) {
      return `strong model: the router could not decide (${reason.slice("router_fallback:".length)})`;
    }
    return "no router on this route";
  }
  const score = d.routerScore.toFixed(4);
  if (d.routerThreshold === null) return score;
  const cheap = d.routerScore >= d.routerThreshold;
  return `${score} ${cheap ? "≥" : "<"} ${d.routerThreshold} → ${cheap ? "cheap" : "strong"} model`;
}

/**
 * SVG polyline points for a series (seconds, say), scaled so `max` is the top. `pad` keeps a
 * line that sits at the top or the bottom off the frame, where it would be hidden. A null leaves
 * a gap; fewer than two points draw nothing.
 */
export function linePoints(
  points: (number | null)[],
  width: number,
  height: number,
  max: number,
  pad = 4,
): string {
  const usable = points.map((p, i) => [i, p] as const).filter(([, p]) => p !== null);
  if (usable.length < 2 || max <= 0) return "";
  const last = Math.max(points.length - 1, 1);
  const inner = height - 2 * pad;
  return usable
    .map(([i, p]) => {
      const clamped = Math.min(Math.max(p as number, 0), max);
      return `${((i / last) * width).toFixed(1)},${(pad + (1 - clamped / max) * inner).toFixed(1)}`;
    })
    .join(" ");
}

/** The top of a chart for these series: the largest value, never below `floor`, rounded up. */
export function chartMax(series: (number | null)[][], floor = 1): number {
  const peak = Math.max(floor, ...series.flat().map((v) => v ?? 0));
  return Math.ceil(peak);
}
