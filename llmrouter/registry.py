"""Model registry: the single source of truth for what models exist, which
tier each belongs to, what it costs, and what it can do.

Prices are loaded from ``config/models.yaml`` and are never hardcoded in
routing logic. A rate change is a YAML edit, not a code change.

This module also defines the shared routing types (``Tier`` and
``RouteDecision``) because they are tightly coupled to a ``ModelSpec``: a
decision *is* the selection of a registered model, so co-locating them keeps
the dependency graph acyclic (classifier and router both import from here).
"""

from __future__ import annotations

import enum
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator


class Tier(str, enum.Enum):
    """Cost/capability tiers, cheapest first. Order matters: escalation walks
    up this ladder, so the integer rank is meaningful."""

    CHEAP = "cheap"
    MEDIUM = "medium"
    FRONTIER = "frontier"

    @property
    def rank(self) -> int:
        """0 = cheapest. Used to compare tiers and to escalate one step up."""
        return {Tier.CHEAP: 0, Tier.MEDIUM: 1, Tier.FRONTIER: 2}[self]

    def next_up(self) -> "Tier | None":
        """The next more-capable tier, or None if already at the top."""
        order = [Tier.CHEAP, Tier.MEDIUM, Tier.FRONTIER]
        idx = order.index(self)
        return order[idx + 1] if idx + 1 < len(order) else None


class ModelSpec(BaseModel):
    """One registered model. Immutable once loaded."""

    model_config = ConfigDict(frozen=True)

    name: str
    tier: Tier
    provider: str
    input_cost_per_1m: float = Field(ge=0)
    output_cost_per_1m: float = Field(ge=0)
    max_context: int = Field(gt=0)
    supports_tools: bool = False

    def cost_for(self, input_tokens: int, output_tokens: int) -> float:
        """USD cost for a call with the given token counts. Split pricing:
        input and output are billed at different rates."""
        return (
            input_tokens / 1_000_000 * self.input_cost_per_1m
            + output_tokens / 1_000_000 * self.output_cost_per_1m
        )


class RouteDecision(BaseModel):
    """The output of routing a single query: which model, which tier, why,
    and which classifier stage decided. ``score`` is the 0..1 complexity
    score when an embedding/LLM classifier produced one (None for pure rule
    or default decisions)."""

    model_config = ConfigDict(frozen=True)

    model: ModelSpec
    tier: Tier
    reason: str
    classifier_used: str  # "rules" | "embedding" | "llm" | "default"
    score: float | None = None


class ModelRegistry(BaseModel):
    """Validated collection of registered models with tier-aware lookups."""

    models: list[ModelSpec]

    @field_validator("models")
    @classmethod
    def _non_empty_and_unique(cls, v: list[ModelSpec]) -> list[ModelSpec]:
        if not v:
            raise ValueError("model registry is empty")
        names = [m.name for m in v]
        dupes = {n for n in names if names.count(n) > 1}
        if dupes:
            raise ValueError(f"duplicate model names in registry: {sorted(dupes)}")
        return v

    @classmethod
    def load(cls, path: str | Path) -> "ModelRegistry":
        """Load and validate the registry YAML at ``path``."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"model registry not found: {path}")
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if "models" not in data:
            raise ValueError(f"{path} missing top-level 'models' key")
        return cls.model_validate(data)

    def get(self, name: str) -> ModelSpec:
        """Look up a model by exact name."""
        for m in self.models:
            if m.name == name:
                return m
        raise KeyError(f"no model named {name!r} in registry")

    def models_by_tier(self, tier: Tier) -> list[ModelSpec]:
        """All models in a tier, sorted cheapest-first by input cost then
        output cost (deterministic ordering)."""
        return sorted(
            (m for m in self.models if m.tier == tier),
            key=lambda m: (m.input_cost_per_1m, m.output_cost_per_1m, m.name),
        )

    def cheapest_in_tier(self, tier: Tier) -> ModelSpec:
        """The cheapest model in a tier. Raises if the tier is empty."""
        candidates = self.models_by_tier(tier)
        if not candidates:
            raise KeyError(f"no models registered in tier {tier.value!r}")
        return candidates[0]

    def cheapest_in_tier_with_context(
        self, tier: Tier, min_context: int
    ) -> ModelSpec | None:
        """Cheapest model in a tier whose ``max_context`` can hold
        ``min_context`` tokens. None if none qualifies."""
        for m in self.models_by_tier(tier):
            if m.max_context >= min_context:
                return m
        return None
