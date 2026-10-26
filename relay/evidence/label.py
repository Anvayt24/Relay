"""The human labeling tool: a small local web page for judging pairs, blind.

    python -m relay.evidence.label --pairs pairs.jsonl --labels labels.jsonl

Opens on http://127.0.0.1:8200. Bound to localhost only: it serves prompts and model answers.

What makes the labels usable as ground truth for calibrating the judge (PRD §10.3):
- **Blind.** The page never shows model names, and which response is on the left is random
  per pair (fixed by a seed, so a refresh doesn't reshuffle). You judge the answers, not the
  brand.
- **The same question as the judge:** is one response *materially worse* than the other?
  Three buttons: "A is materially worse", "B is materially worse", "Neither".
- **Stratified.** The queue interleaves cohorts, so stopping early still leaves a spread
  instead of all of one kind.
- **A blind 10% re-label.** Some pairs come back later without any sign that you've seen
  them. Your agreement with yourself is the ceiling on how well anyone, a judge included, can
  agree with you (PRD: "with a single annotator, re-label a blind 10% subset").
- **Resumable.** Labels are appended to a JSONL file as you go; restarting continues where
  you stopped. Nothing is lost on a crash.

Keys: 1 = A worse, 2 = B worse, 3 = neither.
"""

import argparse
import asyncio
import json
import random
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from relay.evidence.pairs import EvalPair, read_jsonl

RELABEL_FRACTION = 0.10
MIN_RELABEL_GAP = 15  # a re-label comes back at least this many items later


@dataclass(frozen=True)
class QueueItem:
    pair_id: str
    shown_swapped: bool  # True = the candidate is on the right (response B)
    is_relabel: bool


def build_queue(pairs: list[EvalPair], *, seed: int = 0) -> list[QueueItem]:
    """Stratified order, then the blind re-labels slotted in later."""
    rng = random.Random(seed)
    by_cohort: dict[str, list[EvalPair]] = {}
    for p in pairs:
        by_cohort.setdefault(p.cohort, []).append(p)
    for group in by_cohort.values():
        rng.shuffle(group)
    ordered: list[EvalPair] = []
    while any(by_cohort.values()):  # round-robin across cohorts
        for cohort in sorted(by_cohort):
            if by_cohort[cohort]:
                ordered.append(by_cohort[cohort].pop())
    queue = [QueueItem(p.id, rng.random() < 0.5, False) for p in ordered]

    n_relabel = max(1, round(len(queue) * RELABEL_FRACTION)) if len(queue) >= 10 else 0
    for original_pos in sorted(rng.sample(range(len(queue)), n_relabel), reverse=True):
        item = queue[original_pos]
        # Back in with a fresh random side, so it isn't recognized by position.
        at = rng.randint(min(len(queue), original_pos + MIN_RELABEL_GAP), len(queue))
        queue.insert(at, QueueItem(item.pair_id, rng.random() < 0.5, True))
    return queue


def candidate_label(choice: str, *, shown_swapped: bool) -> str:
    """The button the human pressed, as a label from the candidate's point of view."""
    if choice == "neither":
        return "tie"
    a_worse = choice == "a_worse"
    candidate_worse = a_worse != shown_swapped  # candidate is A unless swapped
    return "loss" if candidate_worse else "win"


class Choice(BaseModel):
    pair_id: str
    choice: str  # a_worse | b_worse | neither
    is_relabel: bool = False


def _append_line(path: Path, line: str) -> None:
    # Appended and closed per label: a crash loses at most the label being written.
    with open(path, "a", encoding="utf-8", newline="\n") as f:
        f.write(line + "\n")


def create_app(pairs_path: Path, labels_path: Path, *, annotator: str, seed: int = 0) -> FastAPI:
    pairs = {p.id: p for p in read_jsonl(pairs_path)}
    queue = build_queue(list(pairs.values()), seed=seed)
    done: set[tuple[str, bool]] = set()
    if labels_path.exists():
        for line in labels_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                done.add((row["pair_id"], bool(row["is_relabel"])))
    app = FastAPI(title="Relay labeling", openapi_url=None)

    def pending() -> list[QueueItem]:
        return [q for q in queue if (q.pair_id, q.is_relabel) not in done]

    @app.get("/", response_class=HTMLResponse)
    async def page() -> str:
        return PAGE

    @app.get("/api/next")
    async def next_item() -> dict[str, Any]:
        todo = pending()
        progress = {"done": len(queue) - len(todo), "total": len(queue)}
        if not todo:
            return {"finished": True, "progress": progress}
        item = todo[0]
        p = pairs[item.pair_id]
        cand, ref = p.candidate_response, p.reference_response
        a, b = (ref, cand) if item.shown_swapped else (cand, ref)
        return {
            "pair_id": item.pair_id,
            "is_relabel": item.is_relabel,
            "prompt": p.prompt,
            "a": a,
            "b": b,
            "progress": progress,
        }

    @app.post("/api/label")
    async def label(body: Choice) -> dict[str, str]:
        key = (body.pair_id, body.is_relabel)
        item = next((q for q in queue if (q.pair_id, q.is_relabel) == key), None)
        if item is None or body.choice not in ("a_worse", "b_worse", "neither"):
            raise HTTPException(status_code=400, detail="unknown item or choice")
        row = {
            "pair_id": body.pair_id,
            "label": candidate_label(body.choice, shown_swapped=item.shown_swapped),
            "annotator": annotator,
            "is_relabel": body.is_relabel,
            "shown_swapped": item.shown_swapped,
            "ts": datetime.now(UTC).isoformat(),
        }
        await asyncio.to_thread(_append_line, labels_path, json.dumps(row))
        done.add(key)
        return {"status": "ok"}

    return app


PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>Relay labeling</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
  :root { color-scheme: light dark; --bg:#fafafa; --fg:#1b1b1b; --card:#fff; --line:#d8d8d8;
          --accent:#0b5fff; --muted:#666; }
  @media (prefers-color-scheme: dark) {
    :root { --bg:#14161a; --fg:#e8e8e8; --card:#1d2026; --line:#333844; --muted:#9aa; } }
  body { margin:0; font:15px/1.5 system-ui, sans-serif; background:var(--bg); color:var(--fg); }
  main { max-width:1100px; margin:0 auto; padding:16px; }
  header { display:flex; justify-content:space-between; color:var(--muted); }
  .prompt, .resp { background:var(--card); border:1px solid var(--line); border-radius:8px;
                   padding:12px 14px; white-space:pre-wrap; overflow-wrap:anywhere; }
  .cols { display:grid; grid-template-columns:1fr 1fr; gap:12px; margin-top:12px; }
  @media (max-width:760px) { .cols { grid-template-columns:1fr; } }
  h2 { font-size:13px; text-transform:uppercase; letter-spacing:.06em; color:var(--muted); }
  .buttons { display:flex; gap:10px; margin-top:16px; flex-wrap:wrap; }
  button { font:inherit; padding:10px 16px; border-radius:8px; border:1px solid var(--line);
           background:var(--card); color:var(--fg); cursor:pointer; }
  button:hover { border-color:var(--accent); }
  kbd { border:1px solid var(--line); border-radius:4px; padding:0 5px; font-size:12px; }
</style></head><body><main>
<header><div>Is one response <b>materially worse</b> than the other?</div>
  <div id="progress"></div></header>
<h2>Request</h2><div class="prompt" id="prompt"></div>
<div class="cols"><div><h2>Response A</h2><div class="resp" id="a"></div></div>
  <div><h2>Response B</h2><div class="resp" id="b"></div></div></div>
<div class="buttons">
  <button data-c="a_worse"><kbd>1</kbd> A is materially worse</button>
  <button data-c="b_worse"><kbd>2</kbd> B is materially worse</button>
  <button data-c="neither"><kbd>3</kbd> Neither (equivalent or minor difference)</button>
</div>
<p id="done" hidden>All done. You can close this tab.</p>
<script>
let current = null;
async function load() {
  const r = await (await fetch("/api/next")).json();
  const p = r.progress;
  document.getElementById("progress").textContent = `${p.done} / ${p.total}`;
  if (r.finished) {
    document.querySelector(".buttons").hidden = true;
    document.getElementById("done").hidden = false;
    return;
  }
  current = r;
  for (const id of ["prompt", "a", "b"]) document.getElementById(id).textContent = r[id];
}
async function send(choice) {
  if (!current) return;
  await fetch("/api/label", { method: "POST", headers: {"Content-Type":"application/json"},
    body: JSON.stringify({pair_id: current.pair_id, choice, is_relabel: current.is_relabel}) });
  current = null;
  load();
}
document.querySelectorAll("button").forEach(b => b.onclick = () => send(b.dataset.c));
addEventListener("keydown", e => {
  const m = {"1":"a_worse","2":"b_worse","3":"neither"}[e.key]; if (m) send(m);
});
load();
</script></main></body></html>
"""


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m relay.evidence.label", description=__doc__)
    parser.add_argument("--pairs", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--annotator", default="annotator-1")
    parser.add_argument("--port", type=int, default=8200)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    app = create_app(Path(args.pairs), Path(args.labels), annotator=args.annotator, seed=args.seed)
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
