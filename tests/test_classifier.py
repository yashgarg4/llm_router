"""Tests for the embedding complexity classifier and the full cascade.

These load the real bge-small model (downloaded/cached on first run) — the
classifier's whole value is its actual embeddings, so mocking them would test
nothing. Kept module-scoped so the model loads once.
"""

from __future__ import annotations

import pytest

from llmrouter import Router, RouterConfig, Tier
from llmrouter.classifier import EmbeddingClassifier


@pytest.fixture(scope="module")
def classifier() -> EmbeddingClassifier:
    cfg = RouterConfig()
    pol = cfg.load_policy()
    return EmbeddingClassifier(
        exemplars=pol.exemplars,
        thresholds=pol.thresholds,
        model_name=cfg.embedding_model,
        temperature=cfg.embedding_temperature,
    )


@pytest.mark.parametrize(
    "query",
    [
        "What is the capital of France?",
        "Convert this JSON object to a CSV row.",
        "What is the capital of Japan?",
    ],
)
def test_simple_queries_score_low(
    classifier: EmbeddingClassifier, query: str
) -> None:
    r = classifier.classify(query)
    assert r.tier is Tier.CHEAP
    assert r.score < 0.40


@pytest.mark.parametrize(
    "query",
    [
        "Prove the halting problem is undecidable.",
        "Design a rate limiter for a distributed API gateway.",
        "Derive the worst-case complexity of this recursive routine.",
    ],
)
def test_complex_queries_score_high(
    classifier: EmbeddingClassifier, query: str
) -> None:
    r = classifier.classify(query)
    assert r.tier is Tier.FRONTIER
    assert r.score >= 0.80


def test_score_is_monotone_across_complexity(
    classifier: EmbeddingClassifier,
) -> None:
    simple = classifier.classify("What is the capital of France?").score
    moderate = classifier.classify("Explain what a hash map is to a beginner.").score
    hard = classifier.classify(
        "Design a fault-tolerant distributed consensus protocol."
    ).score
    assert simple < moderate < hard


def test_score_bounds(classifier: EmbeddingClassifier) -> None:
    r = classifier.classify("anything")
    assert 0.0 <= r.score <= 1.0
    assert 0.0 <= r.margin


def test_requires_axis_endpoints() -> None:
    cfg = RouterConfig()
    pol = cfg.load_policy()
    with pytest.raises(ValueError):
        EmbeddingClassifier(
            exemplars={Tier.MEDIUM: pol.exemplars[Tier.MEDIUM]},
            thresholds=pol.thresholds,
        )


# -- full cascade through the Router -----------------------------------------


@pytest.fixture(scope="module")
def router() -> Router:
    return Router.from_config(RouterConfig())


def test_rule_hit_short_circuits_before_embedding(router: Router) -> None:
    d = router.route("Classify this ticket by urgency.")
    assert d.classifier_used == "rules"
    assert d.score is None  # rules produce no complexity score


def test_rule_miss_falls_through_to_embedding(router: Router) -> None:
    d = router.route("What is the capital of France?")
    assert d.classifier_used == "embedding"
    assert d.tier is Tier.CHEAP
    assert d.score is not None


def test_embedding_tiers_a_hard_ambiguous_prompt_up(router: Router) -> None:
    # No frontier keyword, so rules miss; the scorer must recognise the
    # reasoning complexity and route to frontier.
    d = router.route(
        "Walk me through the trade-offs of building your own consensus "
        "layer versus using an off-the-shelf coordination service, and when "
        "each is justified."
    )
    assert d.classifier_used == "embedding"
    assert d.tier is Tier.FRONTIER


def test_llm_classifier_off_by_default(router: Router) -> None:
    assert router.config.use_llm_classifier is False


def test_llm_classifier_fires_only_when_enabled_and_ambiguous() -> None:
    """With the flag on and an injected fake LLM, an ambiguous prompt (small
    centroid margin) routes via the LLM stage; the default path never does."""

    class FakeLLM:
        def classify(self, query: str):
            from llmrouter.classifier.llm import LLMScore

            return LLMScore(score=0.95, tier=Tier.FRONTIER, reason="fake=hard")

    cfg = RouterConfig(use_llm_classifier=True, ambiguity_margin=1.0)  # always ambiguous
    router = Router.from_config(cfg)
    router._llm = FakeLLM()  # inject, skip real provider call

    d = router.route("Tell me something interesting about the ocean.")
    assert d.classifier_used == "llm"
    assert d.tier is Tier.FRONTIER
    assert d.score == 0.95
