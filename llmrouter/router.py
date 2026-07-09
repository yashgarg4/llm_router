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
import time

from pydantic import BaseModel, ConfigDict

from .classifier import EmbeddingClassifier, RuleClassifier
from .client import ModelCallError, ModelClient, ModelResponse
from .config import PolicyConfig, RouterConfig
from .fallback import FallbackPolicy
from .registry import ModelRegistry, RouteDecision, Tier
from .verify import verify as verify_output

logger = logging.getLogger("llmrouter.router")


class EscalationRecord(BaseModel):
    """One escalation step, for logging and metrics."""

    model_config = ConfigDict(frozen=True)

    from_tier: Tier
    to_tier: Tier
    from_model: str
    to_model: str
    trigger: str  # "error" | "verification"
    reason: str
    depth: int


class RouteResult(BaseModel):
    """The outcome of :meth:`Router.acomplete`: what came back, which model
    ultimately produced it, every escalation along the way, and the cost /
    latency of the whole attempt (summed across retries)."""

    model_config = ConfigDict(frozen=True)

    success: bool
    response: ModelResponse | None
    final_decision: RouteDecision
    escalations: list[EscalationRecord]
    total_cost: float
    latency_ms: float
    attempts: int
    error: str | None = None

    @property
    def escalated(self) -> bool:
        return bool(self.escalations)


class Router:
    """Cost- and quality-aware router (rules → embeddings → optional LLM)."""

    def __init__(
        self,
        registry: ModelRegistry,
        policy: PolicyConfig,
        config: RouterConfig | None = None,
        model_client: ModelClient | None = None,
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
        # Escalation is always capped (asserted inside FallbackPolicy).
        self.fallback = FallbackPolicy(registry, self.config.max_escalation_depth)
        # Model client is injectable (fakes in tests/demos); the real Gemini
        # client is built lazily on first acomplete if none is supplied.
        self._client = model_client

    @classmethod
    def from_config(
        cls,
        config: RouterConfig | None = None,
        model_client: ModelClient | None = None,
    ) -> "Router":
        config = config or RouterConfig.from_env()
        registry = config.load_registry()
        policy = config.load_policy()
        return cls(
            registry=registry, policy=policy, config=config,
            model_client=model_client,
        )

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

    # -- completion with capped, logged escalation -------------------------

    def _get_client(self) -> ModelClient:
        if self._client is None:
            from .client import GeminiClient

            self._client = GeminiClient()
        return self._client

    async def acomplete(
        self,
        query: str,
        metadata: dict | None = None,
        verify_spec: dict | None = None,
    ) -> RouteResult:
        """Route ``query``, call the chosen model, and on a transient failure
        or a failed verification escalate one tier — capped and logged — until
        success, exhaustion, or the cap.

        Returns a :class:`RouteResult` with the response (if any), the final
        model, every escalation, and the total cost + latency of the attempt.
        """
        client = self._get_client()
        start = time.perf_counter()

        decision = self.route(query, metadata)
        escalations: list[EscalationRecord] = []
        total_cost = 0.0
        depth = 0
        attempts = 0
        last_error: str | None = None

        while True:
            attempts += 1
            try:
                response = await client.acomplete(decision.model, query, metadata)
            except ModelCallError as exc:
                last_error = str(exc)
                if not exc.transient:
                    # 4xx / bad request: escalating wastes budget. Fail fast.
                    logger.error(
                        "non-transient model error on %s; not escalating: %s",
                        decision.model.name, exc,
                    )
                    return self._result(
                        False, None, decision, escalations, total_cost, start,
                        attempts, error=last_error,
                    )
                nxt = self.fallback.on_failure(decision, exc, depth)
                if nxt is None:
                    return self._result(
                        False, None, decision, escalations, total_cost, start,
                        attempts, error=last_error,
                    )
                escalations.append(
                    self._record(decision, nxt, "error", nxt.reason, depth + 1)
                )
                decision, depth = nxt, depth + 1
                continue

            # A response came back — count its cost (prices from the registry).
            total_cost += decision.model.cost_for(
                response.input_tokens, response.output_tokens
            )

            # Verification: a failed check escalates via the SAME path as a 5xx.
            # This is the silent cost-explosion vector — log it distinctly.
            if verify_spec and not verify_output(response, verify_spec):
                last_error = f"verification failed: {verify_spec.get('check')}"
                nxt = self.fallback.on_failure(decision, last_error, depth)
                logger.warning(
                    "VERIFICATION-TRIGGERED ESCALATION on %s (tier %s): %s "
                    "-- this is the cascade-explosion vector; watch the "
                    "fallback rate for this route",
                    decision.model.name, decision.tier.value, last_error,
                )
                if nxt is None:
                    return self._result(
                        False, response, decision, escalations, total_cost,
                        start, attempts, error=last_error,
                    )
                escalations.append(
                    self._record(decision, nxt, "verification", nxt.reason, depth + 1)
                )
                decision, depth = nxt, depth + 1
                continue

            # Success.
            return self._result(
                True, response, decision, escalations, total_cost, start, attempts,
            )

    @staticmethod
    def _record(
        frm: RouteDecision, to: RouteDecision, trigger: str, reason: str, depth: int
    ) -> EscalationRecord:
        return EscalationRecord(
            from_tier=frm.tier,
            to_tier=to.tier,
            from_model=frm.model.name,
            to_model=to.model.name,
            trigger=trigger,
            reason=reason,
            depth=depth,
        )

    @staticmethod
    def _result(
        success: bool,
        response: ModelResponse | None,
        decision: RouteDecision,
        escalations: list[EscalationRecord],
        total_cost: float,
        start: float,
        attempts: int,
        error: str | None = None,
    ) -> RouteResult:
        return RouteResult(
            success=success,
            response=response,
            final_decision=decision,
            escalations=escalations,
            total_cost=total_cost,
            latency_ms=(time.perf_counter() - start) * 1000.0,
            attempts=attempts,
            error=error,
        )
