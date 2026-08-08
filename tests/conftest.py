"""Shared fixtures: a tiny self-contained corpus + a deterministic fake LLM."""

from __future__ import annotations

import pytest

from nrag._types import Document

# (doc_id, title, body) — distinct topics, a typo target, near-duplicate algorithms.
CORPUS = [
    ("python", "Python Language",
     "Python is a high-level interpreted programming language famous for readable syntax "
     "and a huge standard library used in data science and web development."),
    ("recursion", "Recursion",
     "Recursion is a technique where a function calls itself. A base case stops the "
     "recursion to avoid infinite loops. Tail recursion can be optimized by compilers."),
    ("refund", "Refund Policy",
     "Our refund policy lets customers request their money back within 30 days. "
     "Reimbursements are returned to the original payment method after approval."),
    ("photosynthesis", "Photosynthesis",
     "Photosynthesis is the process plants use to convert sunlight, water and carbon "
     "dioxide into glucose and oxygen inside chloroplasts."),
    ("fox", "Pangram",
     "The quick brown fox jumps over the lazy dog. This pangram contains every letter of "
     "the English alphabet at least once."),
    ("binary_search", "Binary Search",
     "Binary search finds an item in a sorted array by repeatedly halving the search "
     "interval, running in logarithmic time."),
    ("quicksort", "Quicksort",
     "Quicksort is a divide and conquer sorting algorithm that partitions an array around "
     "a pivot element and recurses on the partitions."),
    ("http", "HTTP Protocol",
     "HTTP is the protocol of the web, a stateless request response protocol exchanged "
     "between clients and servers over TCP."),
    ("git", "Git",
     "Git is a distributed version control system that tracks changes in source code "
     "during software development and supports branching and merging."),
    ("neural_net", "Neural Networks",
     "A neural network is a model of connected nodes loosely inspired by the neurons in a "
     "biological brain, trained by adjusting weights."),
]

# (query, relevant_doc_id) — all have lexical overlap so pure-lexical scores high.
QUERIES = [
    ("programming language with readable syntax", "python"),
    ("a function that calls itself", "recursion"),
    ("recurssion typo tolerance", "recursion"),          # typo -> ngram signal
    ("get my money back from a purchase", "refund"),
    ("how plants convert sunlight into energy", "photosynthesis"),
    ("divide and conquer sorting with a pivot", "quicksort"),
    ("search a sorted array in logarithmic time", "binary_search"),
    ("distributed version control for source code", "git"),
    ("stateless web request response protocol", "http"),
]


@pytest.fixture
def corpus_docs():
    return [
        Document(doc_id=d, text=f"# {title}\n\n{body}", source=f"{d}.md",
                 metadata={"content_type": "markdown", "source": f"{d}.md", "title": title})
        for (d, title, body) in CORPUS
    ]


@pytest.fixture
def queries():
    return list(QUERIES)


class FakeLLM:
    """Deterministic stand-in that routes by prompt content and counts calls."""

    def __init__(self):
        self.calls = {"contextual": 0, "expand": 0, "answer": 0}
        self.model_name = "fake-test-model"

    def complete(self, prompt, *, max_tokens=None, temperature=0.0, stop=None, system=None):
        sys = system or ""
        if "context blurb" in sys or "situate this chunk" in prompt:
            self.calls["contextual"] += 1
            return "This chunk is part of a reference document on the topic."
        if "Passage:" in prompt and "search query" in prompt.lower():
            self.calls["expand"] += 1
            return "Refund reimbursement money back returns policy purchase customer."
        if "Keywords:" in prompt:
            self.calls["expand"] += 1
            return "Reasoning here.\nKeywords: refund, money back, reimbursement, return"
        if "Answer (cite sources" in prompt or prompt.startswith("Context:"):
            self.calls["answer"] += 1
            return "Based on the context, here is the answer [1]."
        return "ok"


@pytest.fixture
def fake_llm():
    return FakeLLM()
