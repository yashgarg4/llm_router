"""Escalation policy: what to do when a routed call fails.

The whole project exists because of *uncapped* escalation (INTERNAL_NOTES §1):
a systemic cheap-tier failure that silently escalates ~90% of traffic to the
frontier model. So escalation here is bounded on two independent axes:

  1. **The tier ladder is finite** — cheap → medium → frontier. Once at
     frontier there is nowhere to escalate, so `on_failure` returns None.
  2. **A depth cap** (`max_escalation_depth`, default 2) — a hard limit on how
     many escalations a single request may perform, asserted at construction.

Either bound alone prevents infinite escalation; both are enforced. Every
escalation produces a reason string; the router logs and counts them.
"""

from __future__ import annotations

import logging

from .registry import ModelRegistry, RouteDecision

logger = logging.getLogger("llmrouter.fallback")


class FallbackPolicy:
    """Decides the next model to try after a failure — one tier up, capped."""

    def __init__(self, registry: ModelRegistry, max_escalation_depth: int = 2) -> None:
        # The cap is non-negotiable: a negative or non-integer cap could defeat
        # the very protection this class provides. Assert it loudly.
        assert isinstance(max_escalation_depth, int), "escalation cap must be int"
        assert max_escalation_depth >= 0, "escalation cap must be >= 0"
        self._registry = registry
        self.max_escalation_depth = max_escalation_depth

    def on_failure(
        self, decision: RouteDecision, error: object, depth: int
    ) -> RouteDecision | None:
        """Return the next :class:`RouteDecision` (one tier up) or ``None`` if
        escalation is exhausted.

        ``depth`` is how many escalations have already happened for this
        request. Returns None when the depth cap is reached or the current tier
        is already the top of the ladder — never loops, never exceeds the cap.
        """
        if depth >= self.max_escalation_depth:
            logger.warning(
                "escalation cap reached (depth %d >= max %d); giving up on tier %s",
                depth, self.max_escalation_depth, decision.tier.value,
            )
            return None

        next_tier = decision.tier.next_up()
        if next_tier is None:
            logger.warning(
                "already at frontier tier (%s); cannot escalate further",
                decision.tier.value,
            )
            return None

        model = self._registry.cheapest_in_tier(next_tier)
        reason = (
            f"escalate {decision.tier.value} -> {next_tier.value} "
            f"(escalation {depth + 1}/{self.max_escalation_depth}) after: {error}"
        )
        logger.warning("fallback: %s", reason)
        return RouteDecision(
            model=model,
            tier=next_tier,
            reason=reason,
            classifier_used="fallback",
            score=decision.score,
        )
