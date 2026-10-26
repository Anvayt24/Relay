"""Prompt families for the cache evaluation: where the labeled pairs come from.

The question the cache must answer correctly is: *is the cached answer to prompt A an
acceptable answer to prompt B?* Two kinds of pairs test it:

- **Positives** (label 1): the *same* question worded differently. "What is the capital of
  France?" / "Name the capital city of France." The cached answer is right for both.
- **Hard negatives** (label 0): the *same wording* with one fact changed. "What is 15% of 200?"
  / "What is 15% of 300?" Embeddings call these nearly identical (only one token differs) and
  the answers differ. These are the pairs that make a semantic cache dangerous, so they are
  built on purpose, in every family, sharing as much of the prompt as possible.
- **Easy negatives** (label 0): unrelated prompts from different families.

Each family is a set of paraphrase templates and a set of fact values. A prompt is
``template.format(**value)``; an *instance* is one value (the "fact" being asked about).
Generated, not scraped: no license questions, and any reader can regenerate the data.

Label definition (PRD FR-6): a pair is positive only if the cached response is an acceptable
answer to the new prompt. For these families that is exactly "same instance". The labels are
therefore *constructed*, not human-judged, and the report says so.
"""

from dataclasses import dataclass
from itertools import product


@dataclass(frozen=True)
class Family:
    name: str
    templates: tuple[str, ...]
    instances: tuple[dict[str, str], ...]


def _v(**kw: object) -> dict[str, str]:
    return {k: str(v) for k, v in kw.items()}


PERCENT = Family(
    "percent",
    (
        "What is {p}% of {n}?",
        "Calculate {p} percent of {n}.",
        "{p}% of {n} equals what?",
        "Can you tell me what {p} percent of {n} is?",
    ),
    tuple(
        _v(p=p, n=n)
        for p, n in product(
            (5, 10, 12, 15, 20, 25, 30, 35, 40, 45, 50, 60, 75, 80),
            (80, 120, 150, 200, 240, 300, 360, 450, 500, 800, 1000, 1200),
        )
    ),
)

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
    "Bhutan",
    "Mongolia",
    "Kazakhstan",
    "Uruguay",
    "Ecuador",
    "Bolivia",
    "Venezuela",
    "Tunisia",
    "Algeria",
    "Senegal",
    "Zambia",
    "Uganda",
    "Tanzania",
]

CAPITAL = Family(
    "capital",
    (
        "What is the capital of {x}?",
        "Which city is the capital of {x}?",
        "The capital city of {x} is called what?",
        "Name the capital city of {x}.",
    ),
    tuple(_v(x=c) for c in COUNTRIES),
)

UNITS = (
    ("kilometers", "miles"),
    ("miles", "kilometers"),
    ("kilograms", "pounds"),
    ("pounds", "kilograms"),
    ("liters", "gallons"),
    ("gallons", "liters"),
)

CONVERT = Family(
    "convert",
    (
        "Convert {v} {a} to {b}.",
        "How many {b} are there in {v} {a}?",
        "{v} {a} is how many {b}?",
        "Express {v} {a} in {b}.",
    ),
    tuple(_v(v=v, a=a, b=b) for (a, b), v in product(UNITS, (2, 5, 10, 20, 50, 100))),
)

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
    "supply-and-demand",
    "opportunity-cost",
    "compound-interest",
    "blockchain",
    "encryption",
    "algorithm",
    "polymorphism",
    "concurrency",
    "deadlock",
    "caching",
    "normalization",
    "overfitting",
    "regression",
    "ecosystem",
    "biodiversity",
    "fossil-fuel",
    "renewable-energy",
    "supply-chain",
    "monopoly",
    "recession",
    "federalism",
    "sovereignty",
    "metaphor",
    "irony",
    "satire",
    "alliteration",
    "syntax",
    "semantics",
    "phoneme",
    "isotope",
    "catalyst",
    "enzyme",
    "genome",
    "antibody",
    "vaccine",
    "pandemic",
    "quantum-computing",
    "machine-learning",
    "neural-network",
    "gradient-descent",
    "hash-table",
    "binary-search",
    "big-O-notation",
]

DEFINE = Family(
    "define",
    (
        "What does {x} mean?",
        "Define the term {x}.",
        "Can you explain what {x} is?",
        "Give me a definition of {x}.",
    ),
    tuple(_v(x=t.replace("-", " ")) for t in TERMS),
)

TASKS = [
    "reverse a string",
    "reverse a list",
    "sort a list in ascending order",
    "sort a list in descending order",
    "check if a number is prime",
    "check if a number is even",
    "compute the factorial of a number",
    "compute the nth Fibonacci number",
    "count the vowels in a string",
    "count the words in a string",
    "find the maximum of a list",
    "find the minimum of a list",
    "remove duplicates from a list",
    "merge two sorted lists",
    "check if a string is a palindrome",
    "convert a string to uppercase",
    "convert a string to lowercase",
    "read a file line by line",
    "write text to a file",
    "sum the digits of a number",
    "compute the average of a list",
    "flatten a nested list",
    "count the frequency of each character",
    "swap two variables",
    "find the second largest number",
    "check if two strings are anagrams",
    "generate a random password",
    "compute the greatest common divisor",
    "convert Celsius to Fahrenheit",
    "convert Fahrenheit to Celsius",
]

CODE = Family(
    "code",
    (
        "Write a Python function that can {x}.",
        "Show me Python code to {x}.",
        "How do I {x} in Python?",
        "Give a Python function which will {x}.",
    ),
    tuple(_v(x=t) for t in TASKS),
)

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

TRANSLATE = Family(
    "translate",
    (
        "Translate '{x}' into {l}.",
        "How do you say '{x}' in {l}?",
        "What is '{x}' in {l}?",
        "Give me the {l} translation of '{x}'.",
    ),
    tuple(_v(x=x, l=lang) for x, lang in product(PHRASES, LANGS)),
)

FAMILIES = (PERCENT, CAPITAL, CONVERT, DEFINE, CODE, TRANSLATE)
