// The page's only contact with the server: /v1/demo. Same origin, so no CORS.

export interface Info {
  chat: { real: boolean; sim: boolean };
  chaos: boolean;
  router_threshold: number | null;
}

export interface ClientStats {
  n: number;
  success_rate: number | null;
  p50_s: number | null;
  p95_s: number | null;
}

export interface ChaosStats {
  relay: ClientStats;
  naive: ClientStats;
  enabled: boolean;
}

export interface Interval {
  value: number;
  ci_low: number;
  ci_high: number;
}

export interface JudgeSummary {
  rubric: string[];
  n: number;
  abstained: number;
  status: string;
  kappa: Interval;
  loss_recall: Interval;
  accuracy: Interval;
}

export interface Card {
  route: string;
  comparison: string;
  estimate: number | null;
  floor: number;
  n_judge: number;
  n_human: number;
  judge: string;
  state: string;
  reason: string;
}

export interface Evidence {
  cards: Card[];
  live_cards: Card[];
  policy: { active: string; candidate: string | null; mode: string; canary_pct: number };
  calibration: {
    judges: Record<string, JudgeSummary>;
    gate: { kappa_min: number; loss_recall_min: number; n_min: number };
    human_agreement_mtbench: { n: number; kappa: number; accuracy: number };
    caveat: string;
  };
  simulated: {
    canary: {
      runs: number;
      floor: number;
      bad: { w: number; rolled_back: number; promote_recommended: number; undecided: number };
      good: { w: number; rolled_back: number; promote_recommended: number; undecided: number };
    };
    m8: { cheap_judge: number; sonnet_judge: number | null };
  };
}

export type Mode = "real" | "sim" | "key";

export class ApiError extends Error {
  constructor(
    readonly status: number,
    readonly code: string,
    message: string,
  ) {
    super(message);
  }
}

async function failure(response: Response): Promise<ApiError> {
  try {
    const body = (await response.json()) as { error?: { code?: string; message?: string } };
    return new ApiError(response.status, body.error?.code ?? "error", body.error?.message ?? response.statusText);
  } catch {
    return new ApiError(response.status, "error", response.statusText);
  }
}

async function json<T>(response: Response): Promise<T> {
  if (!response.ok) throw await failure(response);
  return (await response.json()) as T;
}

export const getInfo = async (): Promise<Info> => json(await fetch("/v1/demo/info"));
export const getEvidence = async (): Promise<Evidence> => json(await fetch("/v1/demo/evidence"));
export const getChaosStats = async (): Promise<ChaosStats> =>
  json(await fetch("/v1/demo/chaos/stats"));

export async function postChaos(action: string): Promise<void> {
  const response = await fetch("/v1/demo/chaos", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ action, duration_s: 30 }),
  });
  if (!response.ok) throw await failure(response);
}

export interface ChatResult {
  text: string;
  headers: Headers;
}

/** `key` mode is the ordinary endpoint with the visitor's own API key, so the page is useful on
 *  any gateway, not only one with a public demo tenant. The key is kept in memory only. */
export async function postChat(mode: Mode, prompt: string, apiKey = ""): Promise<ChatResult> {
  const url = mode === "key" ? "/v1/chat/completions" : `/v1/demo/chat?mode=${mode}`;
  const headers: Record<string, string> = { "content-type": "application/json" };
  if (mode === "key") headers["authorization"] = `Bearer ${apiKey}`;
  const response = await fetch(url, {
    method: "POST",
    headers,
    body: JSON.stringify({
      model: "relay/auto",
      messages: [{ role: "user", content: prompt }],
      // Deterministic, so asking the same thing twice is a cache hit (an unset temperature is
      // not cached: it is not repeatable).
      temperature: 0,
    }),
  });
  if (!response.ok) throw await failure(response);
  const body = (await response.json()) as { choices: { message: { content: string } }[] };
  return { text: body.choices[0]?.message.content ?? "", headers: response.headers };
}
