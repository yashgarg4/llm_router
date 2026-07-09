"""The Router: turns a query into a concrete model choice.

Phase 1 is rules-only. The cascade is:

    rules -> (no match) -> default tier

Phase 2 inserts the embedding scorer (and optional LLM classifier) between
the rule miss and the default. The ``route`` signature and the returned
:class:`RouteDecision` are designed so that extension is additive.

Routing latency is deliberately tiny here (a regex scan). That is the point:
the router is *not* the bottleneck — the model call that follows is orders of
magnitude slower. Spending 5-100ms to pick a cheaper/faster model is almost
always a net win. See INTERNAL_NOTES "why router latency is a misframing".
"""

from __future__ import annotations

from .classifier import RuleClassifier
from .config import PolicyConfig, RouterConfig
from .registry import ModelRegistry, RouteDecision


class Router:
    """Cost- and quality-aware router (Phase 1: rule-based)."""

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

    @classmethod
    def from_config(cls, config: RouterConfig | None = None) -> "Router":
        """Build a router by loading the registry + policy from disk."""
        config = config or RouterConfig.from_env()
        registry = config.load_registry()
        policy = config.load_policy()
        return cls(registry=registry, policy=policy, config=config)

    def route(self, query: str, metadata: dict | None = None) -> RouteDecision:
        """Choose a model for ``query``.

        Stage 1 — rules. A rule match short-circuits and returns immediately.
        Stage N — (Phase 2) embedding scorer / optional LLM classifier.
        Fallthrough — default tier from policy.
        """
        decision = self._rules.classify(query, metadata)
        if decision is not None:
            return decision

        # No rule matched. Phase 1 defaults to the policy's default tier;
        # Phase 2 replaces this branch with the embedding scorer.
        tier = self.policy.default_tier
        model = self.registry.cheapest_in_tier(tier)
        return RouteDecision(
            model=model,
            tier=tier,
            reason=f"no rule matched; default tier -> {tier.value}",
            classifier_used="default",
        )
