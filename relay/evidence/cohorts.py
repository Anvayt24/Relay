"""Cohorts: slices of traffic that the evidence is reported per (PRD §10.1, ADR 0013).

A route can be safe overall and unsafe for one kind of request: a cheap model that is fine in
English but not in Hindi, or fine for chat but not for code. Averaging hides that, so every
judged pair carries a cohort, and cards are computed per cohort.

The tags are **cheap, deterministic heuristics** over the request, with a small fixed set of
values each, because a cohort with unbounded values would split the evidence into pieces too
small to say anything (and would be an unbounded metric label):

- ``lang``: the dominant script (``latin``, ``devanagari``, ``cjk``, ``cyrillic``, ``arabic``,
  ``other``). It is a script check, not language identification: English and French are both
  ``latin``. Honest and free; a real language ID model is a possible later refinement.
- ``len``: ``short`` / ``medium`` / ``long`` by prompt characters.
- ``code``: the prompt contains a code fence.
- ``tools``: the request uses tools (always false until tool pass-through, FR-1 P1).

The client's own ``X-Relay-Cohort`` tag (already limited to the tenant's allowlist) is kept
separate, as a free extra dimension.
"""

import unicodedata
from dataclasses import dataclass

SHORT_MAX_CHARS = 200
MEDIUM_MAX_CHARS = 1200

_SCRIPTS = (
    ("devanagari", "DEVANAGARI"),
    ("cjk", "CJK"),
    ("cjk", "HIRAGANA"),
    ("cjk", "KATAKANA"),
    ("cjk", "HANGUL"),
    ("cyrillic", "CYRILLIC"),
    ("arabic", "ARABIC"),
    ("latin", "LATIN"),
)


@dataclass(frozen=True)
class Cohort:
    lang: str
    length: str
    code: bool
    tools: bool = False
    tag: str | None = None  # the client's X-Relay-Cohort, if any

    @property
    def key(self) -> str:
        parts = [f"lang={self.lang}", f"len={self.length}", f"code={str(self.code).lower()}"]
        parts.append(f"tools={str(self.tools).lower()}")
        return ",".join(parts)


def script_of(text: str) -> str:
    """The most common script among the letters of ``text`` (``other`` if there are none)."""
    counts: dict[str, int] = {}
    for ch in text:
        if not ch.isalpha():
            continue
        name = unicodedata.name(ch, "")
        for script, marker in _SCRIPTS:
            if marker in name:
                counts[script] = counts.get(script, 0) + 1
                break
        else:
            counts["other"] = counts.get("other", 0) + 1
    return max(counts, key=lambda s: counts[s]) if counts else "other"


def cohort_of(prompt: str, *, tools: bool = False, tag: str | None = None) -> Cohort:
    n = len(prompt)
    length = "short" if n <= SHORT_MAX_CHARS else "medium" if n <= MEDIUM_MAX_CHARS else "long"
    return Cohort(lang=script_of(prompt), length=length, code="```" in prompt, tools=tools, tag=tag)
