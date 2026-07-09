"""Rule-based classifier — the first (and cheapest) stage of the cascade.

Rules are deterministic, sub-1ms, and never call an embedding model or an
LLM. They exist to catch the *obvious* cases so the more expensive stages
only run on genuinely ambiguous prompts.

``classify`` returns a :class:`RouteDecision`-shaped *tier* decision when a
rule fires, or ``None`` to defer to the next stage. It resolves the tier and
the reason; the router turns that tier into a concrete model via the
registry. (The rule classifier stays registry-agnostic except for the
large-context rule, which needs to force a context-capable model.)
"""

from __future__ import annotations

import re

from ..config import RuleConfig
from ..registry import ModelRegistry, ModelSpec, RouteDecision, Tier


class RuleClassifier:
    """Applies keyword, token-count, and explicit-override rules."""

    def __init__(self, rules: RuleConfig, registry: ModelRegistry) -> None:
        self._rules = rules
        self._registry = registry
        # Pre-compile word-boundary patterns once. Matching on \b avoids
        # "classifier" matching the "classify" keyword by accident... which
        # it would (classify is a prefix), so we match whole words only.
        self._cheap_re = self._compile(rules.keywords.cheap)
        self._frontier_re = self._compile(rules.keywords.frontier)

    @staticmethod
    def _compile(keywords: list[str]) -> re.Pattern[str] | None:
        if not keywords:
            return None
        alt = "|".join(re.escape(k) for k in keywords)
        return re.compile(rf"\b({alt})\w*", re.IGNORECASE)

    def estimate_tokens(self, text: str) -> int:
        """Cheap character-based token estimate. A real tokenizer is
        unnecessary for a routing threshold — order of magnitude is enough."""
        return max(1, len(text) // self._rules.chars_per_token)

    def classify(
        self, query: str, metadata: dict | None = None
    ) -> RouteDecision | None:
        """Return a rule decision, or ``None`` if no rule matches.

        Precedence (first match wins):
          1. explicit ``task_type`` override in metadata
          2. large-context rule (prompt too big -> force a big-context model)
          3. keyword rules (frontier beats cheap on conflict — under-routing a
             hard prompt is the costly mistake, so we bias toward capability)
        """
        metadata = metadata or {}

        # 1. Explicit override: caller already knows the tier.
        override = metadata.get("task_type")
        if override is not None:
            tier = self._coerce_tier(override)
            model = self._registry.cheapest_in_tier(tier)
            return RouteDecision(
                model=model,
                tier=tier,
                reason=f"explicit task_type override -> {tier.value}",
                classifier_used="rules",
            )

        # 2. Large-context rule: force a model that can hold the prompt.
        tokens = int(metadata.get("token_count") or self.estimate_tokens(query))
        if tokens > self._rules.large_context_token_threshold:
            return self._large_context_decision(tokens)

        # 3. Keyword rules. Frontier signals win over cheap signals: a prompt
        # that says both "extract" and "prove" is a reasoning task.
        frontier_hit = self._frontier_re and self._frontier_re.search(query)
        if frontier_hit:
            model = self._registry.cheapest_in_tier(Tier.FRONTIER)
            return RouteDecision(
                model=model,
                tier=Tier.FRONTIER,
                reason=f"frontier keyword matched: {frontier_hit.group(0)!r}",
                classifier_used="rules",
            )

        cheap_hit = self._cheap_re and self._cheap_re.search(query)
        if cheap_hit:
            model = self._registry.cheapest_in_tier(Tier.CHEAP)
            return RouteDecision(
                model=model,
                tier=Tier.CHEAP,
                reason=f"cheap keyword matched: {cheap_hit.group(0)!r}",
                classifier_used="rules",
            )

        # No rule matched — defer to the next classifier stage.
        return None

    # -- helpers ------------------------------------------------------------

    def _coerce_tier(self, value: object) -> Tier:
        if isinstance(value, Tier):
            return value
        try:
            return Tier(str(value).lower())
        except ValueError as exc:
            raise ValueError(
                f"invalid task_type override {value!r}; "
                f"expected one of {[t.value for t in Tier]}"
            ) from exc

    def _large_context_decision(self, tokens: int) -> RouteDecision:
        """Pick the cheapest model across all tiers that can hold ``tokens``.

        A huge prompt is not necessarily *complex*, so we do not force the
        frontier tier — we start cheap and climb only until the context fits.
        """
        for tier in (Tier.CHEAP, Tier.MEDIUM, Tier.FRONTIER):
            model = self._registry.cheapest_in_tier_with_context(tier, tokens)
            if model is not None:
                return RouteDecision(
                    model=model,
                    tier=tier,
                    reason=(
                        f"large-context rule: ~{tokens} tokens exceeds "
                        f"threshold; cheapest fitting model is {model.name} "
                        f"(context {model.max_context})"
                    ),
                    classifier_used="rules",
                )
        # Nothing in the registry can hold it — fall back to the largest.
        biggest: ModelSpec = max(self._registry.models, key=lambda m: m.max_context)
        return RouteDecision(
            model=biggest,
            tier=biggest.tier,
            reason=(
                f"large-context rule: ~{tokens} tokens exceeds every model's "
                f"context; using largest available ({biggest.name})"
            ),
            classifier_used="rules",
        )
