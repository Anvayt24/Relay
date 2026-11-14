// Evidence: how Relay measures its own decisions. Read-only.

import { type Card, type Evidence, getEvidence, type JudgeSummary } from "./api.js";
import { clear, el, h, table } from "./dom.js";
import { percent, withInterval } from "./format.js";

export async function loadEvidence(): Promise<void> {
  const root = el("evidence-body");
  try {
    render(root, await getEvidence());
  } catch {
    clear(root);
    root.append(h("p", { class: "error", role: "alert" }, "The evidence could not be loaded."));
  }
}

function render(root: HTMLElement, data: Evidence): void {
  clear(root);
  const gate = data.calibration.gate;

  root.append(
    h("h3", {}, "Is the judge right?"),
    h(
      "p",
      {},
      "Relay asks an LLM judge whether a cheap model's answer is materially worse than a strong model's. " +
        "Before trusting it, the judge is compared with human labels on public data. " +
        `Relay uses a judge only if Cohen's κ ≥ ${gate.kappa_min} and loss recall ≥ ${gate.loss_recall_min}, on at least ${gate.n_min} pairs.`,
    ),
    table(
      ["Judge", "Rubric", "Pairs", "κ [95% CI]", "Loss recall [95% CI]", "Result"],
      Object.entries(data.calibration.judges).map(([name, j]: [string, JudgeSummary]) => [
        name,
        j.rubric.join(", "),
        String(j.n),
        withInterval(j.kappa),
        withInterval(j.loss_recall),
        j.status,
      ]),
    ),
    h(
      "p",
      { class: "note" },
      `Two humans labelling the same MT-Bench pairs agree at κ = ${data.calibration.human_agreement_mtbench.kappa.toFixed(3)} ` +
        `(n = ${data.calibration.human_agreement_mtbench.n}), the ceiling a judge can be expected to reach there. ` +
        data.calibration.caveat,
    ),
  );

  root.append(
    h("h3", {}, "Route cards"),
    h(
      "p",
      {},
      "One card per cheap-vs-strong comparison: the measured rate at which the cheap answer is as good or better, " +
        "its state, and why. These come from a real run on free Groq models (29 judged pairs).",
    ),
    cardsTable(data.cards),
  );
  if (data.live_cards.length > 0) {
    root.append(h("h4", {}, "This gateway's own cards, right now"), cardsTable(data.live_cards));
  }

  const sim = data.simulated;
  root.append(
    h("h3", {}, "Simulated results"),
    h("p", { class: "note" }, "From the simulator, not from real traffic."),
    table(
      ["Canary policy", "True win-or-tie rate", "Rolled back", "Recommended for promotion", "Undecided"],
      [
        ["Bad (below the floor)", percent(sim.canary.bad.w), percent(sim.canary.bad.rolled_back),
          percent(sim.canary.bad.promote_recommended), percent(sim.canary.bad.undecided)],
        ["Good (above the floor)", percent(sim.canary.good.w), percent(sim.canary.good.rolled_back),
          percent(sim.canary.good.promote_recommended), percent(sim.canary.good.undecided)],
      ],
    ),
    h(
      "p",
      { class: "note" },
      `${sim.canary.runs} simulated canaries each, floor ${sim.canary.floor}. A bad canary is never promoted. ` +
        `Evidence cost, as a share of routed spend: ` +
        `${percent(sim.m8.cheap_judge)} with a cheap judge` +
        (sim.m8.sonnet_judge === null ? "." : `, ${percent(sim.m8.sonnet_judge)} with the calibrated Sonnet judge at the default sampling rate.`),
    ),
  );
}

function cardsTable(cards: Card[]): HTMLTableElement {
  return table(
    ["Route", "Comparison", "Win-or-tie rate", "Floor", "Judged / human", "State", "Why"],
    cards.map((c) => [
      c.route,
      c.comparison,
      percent(c.estimate),
      percent(c.floor),
      `${c.n_judge} / ${c.n_human}`,
      c.state,
      c.reason,
    ]),
  );
}
