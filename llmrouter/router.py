"""The Router: turns a query into a concrete model choice.

The full cascade (Phase 2):

    rules  ──match──▶ decision
      │ miss
      ▼
    embedding scorer ──confident──▶ decision
      │ ambiguous (margin < config.ambiguity_margin)  AND  use_llm_classifier
      ▼
    LLM classifier ──▶ decision              (opt-in; the only networked stage)

If the embedding stage cannot be built (e.g. offline, no model cache) the
router degrades gracefully to the policy's default tier and logs why — it
never fails a route just because the scorer is unavailable.

Routing latency is deliberately tiny (a regex, then one local embedding). The
router is not the bottleneck — the model call that follows is orders of
magnitude slower. See INTERNAL_NOTES §3 "why router latency is a misframing".
"""

from __future__ import annotations

import logging

from .classifier import EmbeddingClassifier, RuleClassifier
from .config import PolicyConfig, RouterConfig
from .registry import ModelRegistry, RouteDecision

logger = logging.getLogger("llmrouter.router")


class Router:
    """Cost- and quality-aware router (rules → embeddings → optional LLM)."""

    def __init__(
        self,
        registry: ModelRegistry,
        policy: PolicyConfig,
        config: RouterConfig | None = None,
    ) -> None:
        self.registry = registry
        self.policy = policy
        self.config = config or RouterConfig()
        self._rules = RuleClassifier(policy.rules, registry)
        # The embedding scorer and LLM classifier are built lazily on first
        # non-rule route so rule-only workloads pay no model-load cost.
        self._embedding: EmbeddingClassifier | None = None
        self._embedding_failed = False
        self._llm = None  # built lazily only if use_llm_classifier

    @classmethod
    def from_config(cls, config: RouterConfig | None = None) -> "Router":
        config = config or RouterConfig.from_env()
        registry = config.load_registry()
        policy = config.load_policy()
        return cls(registry=registry, policy=policy, config=config)

    # -- lazy classifier builders ------------------------------------------

    def _get_embedding(self) -> EmbeddingClassifier | None:
        if self._embedding is None and not self._embedding_failed:
            try:
                self._embedding = EmbeddingClassifier(
                    exemplars=self.policy.exemplars,
                    thresholds=self.policy.thresholds,
                    model_name=self.config.embedding_model,
                    temperature=self.config.embedding_temperature,
                )
            except Exception:  # missing model cache, offline, bad exemplars…
                self._embedding_failed = True
                logger.warning(
                    "embedding classifier unavailable; falling back to default "
                    "tier for rule misses", exc_info=True,
                )
        return self._embedding

    def _get_llm(self):
        if self._llm is None:
            from .classifier.llm import LLMClassifier

            self._llm = LLMClassifier(
                registry=self.registry,
                thresholds=self.policy.thresholds,
                model_name=self.config.llm_classifier_model,
            )
        return self._llm

    # -- routing ------------------------------------------------------------

    def route(self, query: str, metadata: dict | None = None) -> RouteDecision:
        """Choose a model for ``query`` via the cascade."""
        # Stage 1 — rules (sub-1ms, deterministic).
        decision = self._rules.classify(query, metadata)
        if decision is not None:
            return decision

        # Stage 2 — embedding scorer.
        embedder = self._get_embedding()
        if embedder is None:
            return self._default_decision(
                "no rule matched; embedding scorer unavailable -> default tier"
            )

        score = embedder.classify(query)

        # Stage 3 — optional LLM classifier, only when enabled AND the
        # embedding decision was ambiguous (query sits between two tiers).
        if (
            self.config.use_llm_classifier
            and score.margin < self.config.ambiguity_margin
        ):
            try:
                llm_score = self._get_llm().classify(query)
                model = self.registry.cheapest_in_tier(llm_score.tier)
                return RouteDecision(
                    model=model,
                    tier=llm_score.tier,
                    reason=(
                        f"embedding ambiguous (margin {score.margin:.2f} < "
                        f"{self.config.ambiguity_margin}); {llm_score.reason}"
                    ),
                    classifier_used="llm",
                    score=llm_score.score,
                )
            except Exception:
                # Provider hiccup must not fail the route — use the embedding
                # decision we already have.
                logger.warning(
                    "LLM classifier failed; using embedding decision",
                    exc_info=True,
                )

        model = self.registry.cheapest_in_tier(score.tier)
        return RouteDecision(
            model=model,
            tier=score.tier,
            reason=score.reason,
            classifier_used="embedding",
            score=score.score,
        )

    def _default_decision(self, reason: str) -> RouteDecision:
        tier = self.policy.default_tier
        return RouteDecision(
            model=self.registry.cheapest_in_tier(tier),
            tier=tier,
            reason=reason,
            classifier_used="default",
        )
