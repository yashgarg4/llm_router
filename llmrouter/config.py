"""Configuration objects for the router.

Two concerns, two objects:
  * ``PolicyConfig`` — the routing *policy* loaded from ``policy.yaml``
    (tier thresholds, rule keywords, exemplars, default tier).
  * ``RouterConfig`` — runtime knobs (which config files to load, feature
    flags, fallback caps). It knows how to build a registry + policy.

Forward-looking flags (``use_llm_classifier``, ``max_escalation_depth``) are
declared here in Phase 1 with safe defaults so later phases extend this file
rather than rewrite it. The default classifier path never calls an LLM.
"""

from __future__ import annotations

import os
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field

from .registry import ModelRegistry, Tier


class KeywordRules(BaseModel):
    """Keyword -> tier lists for the rule classifier."""

    model_config = ConfigDict(frozen=True)

    cheap: list[str] = Field(default_factory=list)
    frontier: list[str] = Field(default_factory=list)


class RuleConfig(BaseModel):
    """Rule-classifier settings from ``policy.yaml`` under ``rules``."""

    model_config = ConfigDict(frozen=True)

    keywords: KeywordRules = Field(default_factory=KeywordRules)
    large_context_token_threshold: int = Field(default=200_000, gt=0)
    chars_per_token: int = Field(default=4, gt=0)


class Thresholds(BaseModel):
    """Score cut points mapping a 0..1 complexity score to a tier."""

    model_config = ConfigDict(frozen=True)

    cheap_max: float = Field(default=0.40, ge=0, le=1)
    medium_max: float = Field(default=0.80, ge=0, le=1)

    def tier_for_score(self, score: float) -> Tier:
        """Map a 0..1 complexity score onto a tier."""
        if score < self.cheap_max:
            return Tier.CHEAP
        if score < self.medium_max:
            return Tier.MEDIUM
        return Tier.FRONTIER


class PolicyConfig(BaseModel):
    """Full routing policy loaded from ``policy.yaml``."""

    model_config = ConfigDict(frozen=True)

    thresholds: Thresholds = Field(default_factory=Thresholds)
    rules: RuleConfig = Field(default_factory=RuleConfig)
    exemplars: dict[Tier, list[str]] = Field(default_factory=dict)
    default_tier: Tier = Tier.MEDIUM

    @classmethod
    def load(cls, path: str | Path) -> "PolicyConfig":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"policy config not found: {path}")
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        return cls.model_validate(data)


class RouterConfig(BaseModel):
    """Runtime configuration for a :class:`~llmrouter.router.Router`."""

    model_config = ConfigDict(frozen=True)

    models_config: Path = Path("config/models.yaml")
    policy_config: Path = Path("config/policy.yaml")

    # Phase 2: opt-in LLM classifier. Default OFF — the default path is
    # rules -> embeddings only, never an LLM.
    use_llm_classifier: bool = False

    # Phase 3: escalation is always capped so a systemic cheap-tier failure
    # cannot route 100% of traffic to the most expensive model.
    max_escalation_depth: int = Field(default=2, ge=0)

    @classmethod
    def from_env(cls) -> "RouterConfig":
        """Build config, honouring optional path overrides from the env."""
        kwargs: dict[str, Path] = {}
        if p := os.getenv("LLMROUTER_MODELS_CONFIG"):
            kwargs["models_config"] = Path(p)
        if p := os.getenv("LLMROUTER_POLICY_CONFIG"):
            kwargs["policy_config"] = Path(p)
        return cls(**kwargs)

    def load_registry(self) -> ModelRegistry:
        return ModelRegistry.load(self.models_config)

    def load_policy(self) -> PolicyConfig:
        return PolicyConfig.load(self.policy_config)
