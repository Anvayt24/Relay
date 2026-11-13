// Chaos: break the simulated primary and watch Relay and a naive retry client side by side.

import { ApiError, type ChaosStats, type ClientStats, getChaosStats, type Info, postChaos } from "./api.js";
import { clear, el, h } from "./dom.js";
import { chartMax, duration, linePoints, percent } from "./format.js";

const POLL_MS = 1500;
const HISTORY = 60; // points kept for the chart: 90 seconds

export interface Chaos {
  /** Start or stop polling: it only runs while its tab is open. */
  setActive(active: boolean): void;
}

export function initChaos(info: Info): Chaos {
  if (!info.chaos) {
    el("chaos-off").hidden = false;
    el("chaos-live").hidden = true;
    return { setActive: () => undefined };
  }

  // The chart shows p95 latency over time: success rates are both near 100% (the naive client
  // gets there in the end), so the difference between the two is how long it takes.
  const history = { relay: [] as (number | null)[], naive: [] as (number | null)[] };
  let timer: number | undefined;

  const draw = (stats: ChaosStats): void => {
    show("relay", stats.relay);
    show("naive", stats.naive);
    for (const name of ["relay", "naive"] as const) {
      history[name].push(stats[name].p95_s);
      if (history[name].length > HISTORY) history[name].shift();
    }
    const top = chartMax([history.relay, history.naive]);
    const line = (name: "relay" | "naive"): string => linePoints(history[name], 300, 80, top);
    el("line-relay").setAttribute("points", line("relay"));
    el("line-naive").setAttribute("points", line("naive"));
    el("chart-top").textContent = `${top} s`;
  };

  const poll = async (): Promise<void> => {
    try {
      draw(await getChaosStats());
      el("chaos-error").textContent = "";
    } catch (error) {
      el("chaos-error").textContent = error instanceof ApiError ? error.message : "Lost contact with the server.";
    }
  };

  for (const button of document.querySelectorAll<HTMLButtonElement>("[data-chaos]")) {
    button.addEventListener("click", async () => {
      button.disabled = true;
      try {
        await postChaos(button.dataset["chaos"] ?? "clear");
        await poll();
      } catch (error) {
        el("chaos-error").textContent = error instanceof ApiError ? error.message : "The action failed.";
      } finally {
        button.disabled = false;
      }
    });
  }

  return {
    setActive(active: boolean): void {
      if (active && timer === undefined) {
        void poll();
        timer = window.setInterval(() => void poll(), POLL_MS);
      } else if (!active && timer !== undefined) {
        window.clearInterval(timer);
        timer = undefined;
      }
    },
  };
}

function show(name: "relay" | "naive", stats: ClientStats): void {
  const box = el(`stats-${name}`);
  clear(box);
  box.append(
    h("div", { class: "figure" }, stats.success_rate === null ? "—" : percent(stats.success_rate)),
    h("div", { class: "label" }, "of requests succeeded"),
    h("div", { class: "figure small" }, duration(stats.p95_s)),
    h("div", { class: "label" }, "p95 latency"),
    h("div", { class: "label" }, `${stats.n} requests in the last 30 s`),
  );
}
