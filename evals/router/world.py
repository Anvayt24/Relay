"""The simulated world for the router evaluation (ADR 0018).

The simulator's quality is a per-request coin flip, identical for every prompt, so no
classifier could beat random on it. A router can only be evaluated where some prompts are
genuinely harder than others. This module builds such a world:

- every prompt belongs to a **family** (trivia, arithmetic, code, ...) with a base probability
  ``w`` that the cheap model's answer is win-or-tie vs the strong model's;
- ``w`` shifts with **visible** attributes of the prompt (how many digits, how many
  sub-questions, the task's tier), so an embedding or length feature can learn it;
- and with **invisible** per-instance noise that nothing can see, so the best possible router is
  still wrong sometimes (irreducible error, like the real world).

**The numbers below are the ones fixed in ADR 0018 before any router was trained. They are not
tuned afterwards.** The honest framing: a router trained here learns the difficulty structure
built into this world, so results measure the *machinery* (training, thresholding, bootstrap,
PPI, the CI gate), not real-world savings.

Truth for a prompt reuses the simulator's own definitions: the cheap and strong models are each
"correct" or not (``q_for_w`` solves the cheap model's accuracy from ``w``), and
``outcome(cand_correct, ref_correct)`` says win / tie / loss. The judge is ``MockJudge``.
"""

import hashlib
import math
import random
from dataclasses import dataclass

from relay.simulator.judge import JudgeParams, MockJudge
from relay.simulator.quality import outcome, q_for_w
from relay.simulator.seeding import uniform_for

Q_REF = 0.95  # the strong model is right 95% of the time
JUDGE = JudgeParams(sensitivity=0.90, false_loss_rate=0.03)
NOISE_SCALE = 1.6  # per-instance logit shift is (u - 0.5) * NOISE_SCALE


@dataclass(frozen=True)
class Item:
    text: str
    family: str
    instance: str  # paraphrases of the same question share an instance (and its hidden w)
    w: float  # hidden: P(cheap answer is win-or-tie vs strong), before the invisible noise


def _clamp(w: float) -> float:
    return min(0.995, max(0.20, w))


def logit(p: float) -> float:
    return math.log(p / (1 - p))


def sigmoid(z: float) -> float:
    return 1 / (1 + math.exp(-z))


def hidden_w(seed: int, item: Item) -> float:
    """The cheap model's true win-or-tie probability for this prompt, after the invisible
    per-instance noise (so paraphrases of one question share it)."""
    u = uniform_for(seed, item.instance, "difficulty-noise")
    return sigmoid(logit(_clamp(item.w)) + (u - 0.5) * NOISE_SCALE)


def text_id(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


# ---------------------------------------------------------------- prompt families
COUNTRIES = [
    "France",
    "Germany",
    "Italy",
    "Spain",
    "Portugal",
    "Poland",
    "Sweden",
    "Norway",
    "Finland",
    "Denmark",
    "Greece",
    "Turkey",
    "Egypt",
    "Kenya",
    "Nigeria",
    "Ghana",
    "Morocco",
    "Ethiopia",
    "Brazil",
    "Argentina",
    "Chile",
    "Peru",
    "Colombia",
    "Mexico",
    "Canada",
    "Japan",
    "China",
    "India",
    "Pakistan",
    "Thailand",
    "Vietnam",
    "Indonesia",
    "Australia",
    "Ireland",
    "Austria",
    "Switzerland",
    "Belgium",
    "Hungary",
    "Romania",
    "Ukraine",
    "Cuba",
    "Iran",
    "Iraq",
    "Israel",
    "Jordan",
    "Nepal",
    "Mongolia",
    "Kazakhstan",
    "Uruguay",
    "Ecuador",
    "Bolivia",
    "Tunisia",
    "Algeria",
    "Senegal",
    "Zambia",
    "Uganda",
    "Tanzania",
]
WORKS = [
    "Hamlet",
    "Pride and Prejudice",
    "Moby-Dick",
    "Don Quixote",
    "The Odyssey",
    "War and Peace",
    "Ulysses",
    "Great Expectations",
    "Jane Eyre",
    "Frankenstein",
    "Dracula",
    "The Great Gatsby",
    "1984",
    "Brave New World",
    "The Catcher in the Rye",
    "Crime and Punishment",
    "The Divine Comedy",
    "Les Miserables",
    "Middlemarch",
    "Wuthering Heights",
    "The Hobbit",
    "Beloved",
    "Lolita",
    "Candide",
    "Faust",
    "Anna Karenina",
    "The Trial",
    "Heart of Darkness",
    "Emma",
    "Persuasion",
]
EVENTS = [
    "the Berlin Wall fell",
    "the first moon landing happened",
    "World War II ended",
    "the French Revolution began",
    "the Titanic sank",
    "the printing press was invented",
    "Columbus reached the Americas",
    "the Roman Empire fell",
    "the Magna Carta was signed",
    "the Soviet Union dissolved",
    "the Industrial Revolution began",
    "the Declaration of Independence was signed",
    "the Black Death reached Europe",
    "the Wright brothers first flew",
    "the Suez Canal opened",
]
TERMS = [
    "photosynthesis",
    "inflation",
    "entropy",
    "recursion",
    "latency",
    "idempotence",
    "mitosis",
    "osmosis",
    "gravity",
    "democracy",
    "opportunity cost",
    "compound interest",
    "blockchain",
    "encryption",
    "an algorithm",
    "polymorphism",
    "concurrency",
    "a deadlock",
    "caching",
    "normalization",
    "overfitting",
    "regression",
    "an ecosystem",
    "biodiversity",
    "a monopoly",
    "a recession",
    "federalism",
    "sovereignty",
    "a metaphor",
    "irony",
    "satire",
    "alliteration",
    "syntax",
    "a phoneme",
    "an isotope",
    "a catalyst",
    "an enzyme",
    "a genome",
    "an antibody",
    "a vaccine",
    "a pandemic",
    "machine learning",
    "a neural network",
    "gradient descent",
    "a hash table",
    "binary search",
    "big-O notation",
    "supply and demand",
    "a derivative",
    "an integral",
    "a prime number",
    "a vector",
    "a matrix",
    "a eigenvalue",
    "probability",
    "variance",
    "a p-value",
]
PHRASES = [
    "good morning",
    "thank you very much",
    "where is the train station",
    "how much does this cost",
    "I would like a glass of water",
    "what time is it",
    "nice to meet you",
    "I do not understand",
    "please speak more slowly",
    "where is the bathroom",
    "have a great day",
    "I am lost",
    "the food is delicious",
    "can you help me",
    "see you tomorrow",
]
LANGS = ("Spanish", "French", "German", "Italian", "Hindi")
UNITS = (
    ("kilometers", "miles"),
    ("miles", "kilometers"),
    ("kilograms", "pounds"),
    ("pounds", "kilograms"),
    ("liters", "gallons"),
    ("gallons", "liters"),
)
CODE_TASKS = {
    "simple": (
        "reverse a string|sum the numbers in a list|count the vowels in a string|check if a number "
        "is even|find the maximum of a list|convert Celsius to Fahrenheit|check if a string is a "
        "palindrome|remove duplicates from a list|count the words in a sentence|swap two variables"
    ),
    "medium": (
        "merge overlapping intervals|parse a CSV line with quoted fields|implement binary search "
        "on a sorted list|group a list of words into anagram sets|validate balanced parentheses "
        "in an expression|flatten an arbitrarily nested list|compute the longest common "
        "subsequence of two strings|implement an LRU cache|find the k most frequent elements in a "
        "list|rotate a matrix 90 degrees"
    ),
    "hard": (
        "implement a thread-safe token-bucket rate limiter|implement a trie with prefix search "
        "and deletion|find the shortest path in a weighted graph with negative edges|implement a "
        "lock-free single-producer single-consumer queue|solve the N-queens problem for N up to "
        "12 efficiently|implement a regular expression matcher supporting . and *|build a "
        "consistent-hash ring with virtual nodes|implement the Raft leader-election protocol|"
        "write a tokenizer and recursive-descent parser for arithmetic expressions|detect "
        "cycles in a distributed wait-for graph"
    ),
}
CODE_W = {"simple": 0.85, "medium": 0.68, "hard": 0.50}
LANG_NAMES = ("Python", "JavaScript", "Go", "Rust")
STEMS = [
    "explain why the sky is blue",
    "compare the causes of the two world wars",
    "describe how a vaccine trains the immune system",
    "outline the pros and cons of nuclear power",
    "explain how interest rates affect inflation",
    "summarize the plot of a tragedy of your choice",
    "explain how a compiler differs from an interpreter",
    "describe the water cycle",
    "argue for and against a four-day work week",
    "explain what causes the seasons",
    "describe how public-key encryption works",
    "compare capitalism and socialism",
    "explain how neural networks learn",
    "describe the causes of the 2008 financial crisis",
    "explain how GPS determines position",
    "outline how a bill becomes law",
    "explain natural selection",
    "describe the structure of DNA",
    "compare TCP and UDP",
    "explain what a black hole is",
    "describe how a car engine works",
    "discuss the ethics of gene editing",
    "explain supply chain resilience",
    "describe how the internet routes packets",
    "explain the greenhouse effect",
    "compare renewable and fossil energy costs",
    "describe how memory works in the brain",
    "explain the central limit theorem",
    "outline the stages of grief",
    "explain how credit scores are calculated",
]
SITUATIONS = [
    "just got a job offer in another city",
    "keep procrastinating on my thesis",
    "had an argument with my best friend",
    "want to start running but hate it",
    "feel overwhelmed at work",
    "am thinking about adopting a dog",
    "cannot decide between two universities",
    "want to learn a new language but have no time",
    "am nervous about a presentation tomorrow",
    "need to ask my boss for a raise",
    "am moving in with my partner",
    "feel lonely after relocating",
    "want to change careers at forty",
    "keep forgetting to save money",
    "am torn about lending money to a relative",
    "want to be more creative",
    "am burnt out but cannot take time off",
    "need to give hard feedback to a colleague",
    "am planning a first solo trip",
    "struggle to wake up early",
]


def _digits(n: int) -> int:
    return len(str(abs(n)))


def generate(seed: int = 0) -> list[Item]:
    """About 3,500 prompts across 8 families. Deterministic for a seed."""
    rng = random.Random(seed)
    items: list[Item] = []

    def add(text: str, family: str, instance: str, w: float) -> None:
        items.append(Item(text, family, f"{family}:{instance}", w))

    # trivia: easy for a small model
    for c in COUNTRIES:
        for t in ("What is the capital of {x}?", "Which city is the capital of {x}?",
                  "Name the capital city of {x}.", "The capital of {x} is which city?"):  # fmt: skip  # noqa: E501
            add(t.format(x=c), "trivia", f"capital-{c}", 0.985)
    for work in WORKS:
        for t in ("Who wrote {x}?", "Who is the author of {x}?", "Name the author of {x}."):
            add(t.format(x=work), "trivia", f"author-{work}", 0.985)
    for ev in EVENTS:
        for t in ("In what year did {x}?", "When was it that {x}? Give the year."):
            add(t.format(x=ev), "trivia", f"year-{ev}", 0.985)

    for term in TERMS:
        for t in ("What is {x}?", "Define {x}.", "Can you explain what {x} means?",
                  "Give me a short definition of {x}."):  # fmt: skip
            add(t.format(x=term), "definition", term, 0.96)

    for phrase in PHRASES:
        for lang in LANGS:
            for t in ("Translate '{x}' into {l}.", "How do you say '{x}' in {l}?",
                      "What is '{x}' in {l}?", "Give me the {l} translation of '{x}'."):  # fmt: skip  # noqa: E501
                add(t.format(x=phrase, l=lang), "translation", f"{phrase}-{lang}", 0.96)

    for i in range(130):
        digits = rng.choice([1, 2, 2, 3, 3, 4, 5, 6])
        value = rng.randint(10 ** (digits - 1), 10**digits - 1)
        a, b = rng.choice(UNITS)
        w = 0.93 - 0.04 * max(0, digits - 2)
        for t in ("Convert {v} {a} to {b}.", "How many {b} are there in {v} {a}?",
                  "{v} {a} is how many {b}?"):  # fmt: skip
            add(t.format(v=value, a=a, b=b), "conversion", f"{i}-{value}-{a}", w)

    for i in range(230):
        ops = rng.choice([1, 1, 2, 2, 3])
        digits = rng.choice([2, 2, 3, 3, 4, 5, 6])
        n = [rng.randint(10 ** (digits - 1), 10**digits - 1) for _ in range(4)]
        if ops == 1:
            p = rng.choice([5, 10, 12, 15, 20, 25, 30, 35, 40, 60, 75])
            texts = (f"What is {p}% of {n[0]}?", f"Calculate {p} percent of {n[0]}.",
                     f"{p}% of {n[0]} equals what?")  # fmt: skip
        elif ops == 2:
            texts = (f"Compute {n[0]} * {n[1]} + {n[2]}.", f"What is {n[0]} times {n[1]}, plus {n[2]}?",  # noqa: E501
                     f"Evaluate {n[0]}*{n[1]}+{n[2]}.")  # fmt: skip
        else:
            texts = (f"Evaluate ({n[0]} + {n[1]}) * {n[2]} - {n[3]}.",
                     f"What is ({n[0]} + {n[1]}) times {n[2]}, minus {n[3]}?",
                     f"Work out ({n[0]}+{n[1]})*{n[2]}-{n[3]}.")  # fmt: skip
        w = 0.88 - 0.07 * max(0, digits - 2) - 0.05 * (ops - 1)
        for t in texts:
            add(t, "arithmetic", f"{i}-{ops}-{digits}-{n[0]}", w)

    for tier, tasks in CODE_TASKS.items():
        for task in tasks.split("|"):
            for lang in LANG_NAMES[:3]:
                for t in ("Write a {l} function that can {x}.", "Show me {l} code to {x}.",
                          "How do I {x} in {l}?"):  # fmt: skip
                    add(t.format(l=lang, x=task), "code", f"{tier}-{task}-{lang}", CODE_W[tier])

    for i in range(230):
        k = rng.choice([2, 2, 3, 3, 4, 5, 6])
        picked = rng.sample(STEMS, k)
        ordinals = ("First", "Second", "Third", "Fourth", "Fifth", "Sixth")
        text = " ".join(f"{ordinals[j]}, {s}." for j, s in enumerate(picked))
        add(text, "reasoning", f"{i}-{k}", 0.70 - 0.07 * k)
        # a second wording of the same question: different framing, same instance
        add(
            "Please address all of the following. " + text, "reasoning", f"{i}-{k}", 0.70 - 0.07 * k
        )

    for s in SITUATIONS:
        for t in ("I {x}. What should I do?", "I {x}, any advice?", "What would you suggest if I {x}?",  # noqa: E501
                  "Help me think through this: I {x}."):  # fmt: skip
            add(t.format(x=s), "advice", s, 0.62)
    for i in range(80):
        s1, s2 = rng.sample(SITUATIONS, 2)
        add(f"I {s1}, and I also {s2}. How should I prioritize?", "advice", f"combo-{i}", 0.60)

    # exact duplicates carry no new information
    seen: set[str] = set()
    unique = []
    for it in items:
        if it.text not in seen:
            seen.add(it.text)
            unique.append(it)
    return unique


# ---------------------------------------------------------------- truth and judge labels
def truth(seed: int, item: Item) -> tuple[bool, bool, float]:
    """(cheap model's answer correct?, strong model's correct?, the hidden w used)."""
    w = hidden_w(seed, item)
    q_cheap = q_for_w(w, Q_REF)
    tid = text_id(item.text)
    cheap = uniform_for(seed, tid, "cheap-correct") < q_cheap
    ref = uniform_for(seed, tid, "strong-correct") < Q_REF
    return cheap, ref, w


def labels(seed: int, item: Item, index: int) -> tuple[bool, bool, float]:
    """(y_true, y_judge, w): win-or-tie as the truth says, and as the noisy judge says."""
    cheap, ref, w = truth(seed, item)
    y_true = outcome(cheap, ref) != "loss"
    judge = MockJudge(seed, JUDGE)
    y_judge = judge.verdict(cand_correct=cheap, ref_correct=ref, index=index, candidate="cheap")
    return y_true, y_judge == "win_or_tie", w
