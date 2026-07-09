"""Tests for the rule classifier."""

from __future__ import annotations

import pytest

from llmrouter.classifier import RuleClassifier
from llmrouter.config import RouterConfig
from llmrouter.registry import Tier


@pytest.fixture(scope="module")
def classifier() -> RuleClassifier:
    cfg = RouterConfig()
    registry = cfg.load_registry()
    policy = cfg.load_policy()
    return RuleClassifier(policy.rules, registry)


@pytest.mark.parametrize(
    "query",
    [
        "Classify this email as spam or not spam.",
        "Extract all dates from the document.",
        "Translate this into German.",
        "Reformat these numbers as currency.",
    ],
)
def test_cheap_keywords_route_cheap(classifier: RuleClassifier, query: str) -> None:
    d = classifier.classify(query)
    assert d is not None
    assert d.tier is Tier.CHEAP
    assert d.classifier_used == "rules"


@pytest.mark.parametrize(
    "query",
    [
        "Prove this theorem by induction.",
        "Debug this segfault in the parser.",
        "Design a fault-tolerant message queue.",
        "Refactor this to remove the circular import.",
    ],
)
def test_frontier_keywords_route_frontier(
    classifier: RuleClassifier, query: str
) -> None:
    d = classifier.classify(query)
    assert d is not None
    assert d.tier is Tier.FRONTIER


def test_frontier_beats_cheap_on_conflict(classifier: RuleClassifier) -> None:
    # Contains both "extract" (cheap) and "prove" (frontier).
    d = classifier.classify("Extract the axioms, then prove the lemma.")
    assert d is not None
    assert d.tier is Tier.FRONTIER


def test_no_keyword_returns_none(classifier: RuleClassifier) -> None:
    assert classifier.classify("Tell me about the weather today.") is None


def test_word_boundary_avoids_false_prefix_match(
    classifier: RuleClassifier,
) -> None:
    # "classifier" should still match the "classify" family (\w* suffix),
    # but an unrelated word must not trigger a cheap match.
    assert classifier.classify("What is your favourite colour?") is None


def test_explicit_task_type_override(classifier: RuleClassifier) -> None:
    d = classifier.classify("anything at all", metadata={"task_type": "frontier"})
    assert d is not None
    assert d.tier is Tier.FRONTIER
    assert "override" in d.reason


def test_invalid_task_type_raises(classifier: RuleClassifier) -> None:
    with pytest.raises(ValueError):
        classifier.classify("x", metadata={"task_type": "supercomputer"})


def test_large_context_forces_context_capable_model(
    classifier: RuleClassifier,
) -> None:
    d = classifier.classify("short text", metadata={"token_count": 350_000})
    assert d is not None
    assert d.model.max_context >= 350_000
    assert "large-context" in d.reason


def test_token_estimate_from_query_length(classifier: RuleClassifier) -> None:
    # ~4 chars/token; 1.2M chars -> ~300k tokens, over the 200k threshold.
    huge = "word " * 300_000  # 1.5M chars
    d = classifier.classify(huge)
    assert d is not None
    assert "large-context" in d.reason
