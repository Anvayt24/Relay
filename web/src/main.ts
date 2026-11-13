import { getInfo } from "./api.js";
import { initChaos } from "./chaos.js";
import { el } from "./dom.js";
import { loadEvidence } from "./evidence.js";
import { initPlayground } from "./playground.js";

const TABS = ["playground", "chaos", "evidence"] as const;
type Tab = (typeof TABS)[number];

async function main(): Promise<void> {
  const info = await getInfo();
  initPlayground(info);
  const chaos = initChaos(info);
  let evidenceLoaded = false;

  const open = (tab: Tab): void => {
    for (const name of TABS) {
      const selected = name === tab;
      el(`tab-${name}`).setAttribute("aria-selected", String(selected));
      (el(`tab-${name}`) as HTMLButtonElement).tabIndex = selected ? 0 : -1;
      el(`panel-${name}`).hidden = !selected;
    }
    chaos.setActive(tab === "chaos");
    if (tab === "evidence" && !evidenceLoaded) {
      evidenceLoaded = true;
      void loadEvidence();
    }
    history.replaceState(null, "", `#${tab}`);
  };

  for (const name of TABS) {
    el(`tab-${name}`).addEventListener("click", () => open(name));
    // Arrow keys move between tabs, as a tab list should.
    el(`tab-${name}`).addEventListener("keydown", (event) => {
      const key = (event as KeyboardEvent).key;
      if (key !== "ArrowRight" && key !== "ArrowLeft") return;
      const step = key === "ArrowRight" ? 1 : -1;
      const next = TABS[(TABS.indexOf(name) + step + TABS.length) % TABS.length] as Tab;
      open(next);
      el(`tab-${next}`).focus();
    });
  }

  // Stop polling when the page is hidden: nobody is watching, so nothing should be sent.
  document.addEventListener("visibilitychange", () => {
    const current = TABS.find((name) => !el(`panel-${name}`).hidden);
    chaos.setActive(current === "chaos" && document.visibilityState === "visible");
  });

  const fromHash = location.hash.slice(1) as Tab;
  open(TABS.includes(fromHash) ? fromHash : "playground");
}

main().catch(() => {
  el("fatal").hidden = false;
});
