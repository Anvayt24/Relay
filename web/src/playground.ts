// Playground: send a prompt, then show what Relay decided and why.

import { ApiError, type Info, type Mode, postChat } from "./api.js";
import { clear, el, h, term } from "./dom.js";
import {
  decisionFromHeaders,
  duration,
  percent,
  routerSentence,
  savedFraction,
  usd,
} from "./format.js";

export function initPlayground(info: Info): void {
  const form = el<HTMLFormElement>("chat-form");
  const prompt = el<HTMLTextAreaElement>("prompt");
  const send = el<HTMLButtonElement>("send");
  const answer = el<HTMLElement>("answer");
  const decision = el<HTMLElement>("decision");
  const mode = el<HTMLSelectElement>("mode");

  // Only the modes this server has turned on.
  const modes: [Mode, string][] = [];
  if (info.chat.real) modes.push(["real", "Free models (Groq)"]);
  if (info.chat.sim) modes.push(["sim", "Simulator (shows the router)"]);
  modes.push(["key", "This gateway, with your API key"]); // always: it works on any deployment
  for (const [value, label] of modes) mode.append(h("option", { value }, label));
  if (modes.length === 1) mode.disabled = true;
  const keyRow = el<HTMLElement>("key-row");
  const key = el<HTMLInputElement>("api-key");
  const showKey = (): void => {
    keyRow.hidden = mode.value !== "key";
  };
  mode.addEventListener("change", showKey);
  showKey();

  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    const text = prompt.value.trim();
    if (text === "") return;
    send.disabled = true;
    clear(answer);
    clear(decision);
    answer.textContent = "Waiting for the answer…";
    try {
      const chosen = mode.value as Mode;
      const result = await postChat(chosen, text, key.value.trim());
      const d = decisionFromHeaders(result.headers, chosen === "real" ? null : info.router_threshold);
      answer.textContent = result.text; // text, never HTML
      const saved = savedFraction(d);
      decision.append(
        h("h3", {}, "What Relay decided"),
        h(
          "dl",
          { class: "facts" },
          ...term("Cache", d.cache ?? "—", d.cache === "hit" ? "served from the cache, no model call" : ""),
          ...term("Router", routerSentence(d)),
          ...term("Answered by", `${d.model ?? "—"} (${d.provider ?? "—"})`),
          ...term("Fallback hops", d.fallbackHops === null ? "—" : String(d.fallbackHops), d.fallbackHops === 0 ? "the primary answered" : ""),
          ...term("Latency", d.latencyMs === null ? "—" : duration(d.latencyMs / 1000)),
          ...term("Cost", usd(d.costUsd), "list price"),
          ...term("Always-strong cost", usd(d.strongCostUsd), saved === null ? "" : `${percent(saved)} saved`),
        ),
      );
    } catch (error) {
      answer.textContent = "";
      const message =
        error instanceof ApiError ? `${error.message} (${error.code})` : "The request could not be sent.";
      answer.append(h("p", { class: "error", role: "alert" }, message));
    } finally {
      send.disabled = false;
    }
  });
}
