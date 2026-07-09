"""llmrouter — a cost- and quality-aware LLM router with per-route observability.

Public API (grows by phase):
  * Router            — route(query, metadata) -> RouteDecision   [Phase 1]
  * RouterConfig      — runtime knobs / feature flags             [Phase 1]
  * ModelRegistry     — YAML-backed model registry                [Phase 1]
  * RouteDecision     — the routing result                        [Phase 1]
  * Tier              — cheap / medium / frontier                 [Phase 1]

``Router.acomplete`` (call the chosen model with capped, logged escalation)
arrives in Phase 3.
"""

from .config import PolicyConfig, RouterConfig
from .registry import ModelRegistry, ModelSpec, RouteDecision, Tier
from .router import Router

__all__ = [
    "Router",
    "RouterConfig",
    "PolicyConfig",
    "ModelRegistry",
    "ModelSpec",
    "RouteDecision",
    "Tier",
]

__version__ = "0.1.0"
