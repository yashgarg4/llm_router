"""OPTIONAL LLM-based complexity classifier — the third cascade stage.

⚠️  OFF BY DEFAULT. This is the only classifier that makes a network call.
It adds a full model round-trip (≈50-100ms + network) and a per-query cost to
the routing hot path, and couples routing availability to a model provider's
uptime. It runs *only* when:

    config.use_llm_classifier is True   AND   the embedding stage was ambiguous
    (top-two tier centroids too close to call)

That narrow trigger is the whole point: pay for an LLM opinion only on the
handful of prompts that genuinely sit between two tiers, never on every
request. See INTERNAL_NOTES §2 "why not an LLM classifier by default".

It calls a *cheap* model (the classifier must never cost more than the
routing it informs) and requests a structured 0..1 complexity score.
"""

from __future__ import annotations

import logging

from pydantic import BaseModel, ConfigDict, Field

from ..config import Thresholds
from ..registry import ModelRegistry, Tier

logger = logging.getLogger("llmrouter.classifier.llm")


class ComplexityAssessment(BaseModel):
    """Structured output the classifier model is asked to return."""

    score: float = Field(
        ge=0, le=1,
        description="0 = trivial (classify/extract/format), "
        "1 = hard reasoning/code/multi-step. 0.5 = moderate (summarise/draft).",
    )
    rationale: str = Field(description="One short sentence explaining the score.")


class LLMScore(BaseModel):
    """Result of an LLM complexity assessment, shaped like EmbeddingScore."""

    model_config = ConfigDict(frozen=True)

    score: float
    tier: Tier
    reason: str


_SYSTEM_PROMPT = (
    "You are a request-complexity classifier for an LLM router. Rate how much "
    "reasoning capability the user's request needs, as a score from 0 to 1:\n"
    "  0.0-0.4  simple: classify, extract, translate, format, factual lookup\n"
    "  0.4-0.8  moderate: summarise, draft, rewrite, explain simply\n"
    "  0.8-1.0  hard: multi-step reasoning, proofs, debugging, system design\n"
    "Return only the structured score and a one-sentence rationale. Do NOT "
    "answer the request itself."
)


class LLMClassifier:
    """Flag-gated LLM complexity classifier. Constructs its model lazily."""

    def __init__(
        self,
        registry: ModelRegistry,
        thresholds: Thresholds,
        model_name: str = "gemini-3.1-flash-lite",
        client: object | None = None,
    ) -> None:
        # Validate the model is registered and cheap-tier-ish; fail loud early.
        self._model_spec = registry.get(model_name)
        self._thresholds = thresholds
        self._model_name = model_name
        self._client = client  # injectable for tests; built lazily otherwise

    def _get_client(self):
        if self._client is None:
            # Lazy import: langchain-google-genai is only needed when the flag
            # is on, so the default install/path never requires it.
            from langchain_google_genai import ChatGoogleGenerativeAI

            llm = ChatGoogleGenerativeAI(model=self._model_name, temperature=0)
            self._client = llm.with_structured_output(ComplexityAssessment)
        return self._client

    def classify(self, query: str) -> LLMScore:
        """One cheap-model call → structured complexity score → tier."""
        client = self._get_client()
        assessment: ComplexityAssessment = client.invoke(  # type: ignore[union-attr]
            [("system", _SYSTEM_PROMPT), ("human", query)]
        )
        score = min(1.0, max(0.0, float(assessment.score)))
        tier = self._thresholds.tier_for_score(score)
        logger.info(
            "llm classifier: score=%.2f tier=%s rationale=%s",
            score, tier.value, assessment.rationale,
        )
        return LLMScore(
            score=score,
            tier=tier,
            reason=(
                f"llm classifier score {score:.2f} -> {tier.value} "
                f"({assessment.rationale})"
            ),
        )
