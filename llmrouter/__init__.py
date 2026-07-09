"""llmrouter — a cost- and quality-aware LLM router with per-route observability.

Public API (grows by phase):
  * Router            — route(query, metadata) -> RouteDecision   [Phase 1]
  * RouterConfig      — runtime knobs / feature flags             [Phase 1]
  * ModelRegistry     — YAML-backed model registry                [Phase 1]
  * RouteDecision     — the routing result                        [Phase 1]
  * Tier              — cheap / medium / frontier                 [Phase 1]

  * Router.acomplete    — call chosen model, capped+logged escalation [Phase 3]
  * RouteResult         — response + escalations + cost + latency      [Phase 3]
  * FallbackPolicy      — one-tier-up, capped escalation               [Phase 3]
  * verify / ModelClient — verification hooks, model-call abstraction  [Phase 3]
"""

from .client import GeminiClient, ModelCallError, ModelClient, ModelResponse
from .config import PolicyConfig, RouterConfig
from .fallback import FallbackPolicy
from .registry import ModelRegistry, ModelSpec, RouteDecision, Tier
from .router import EscalationRecord, RouteResult, Router
from .verify import verify

__all__ = [
    "Router",
    "RouterConfig",
    "PolicyConfig",
    "ModelRegistry",
    "ModelSpec",
    "RouteDecision",
    "Tier",
    "RouteResult",
    "EscalationRecord",
    "FallbackPolicy",
    "verify",
    "ModelClient",
    "ModelResponse",
    "ModelCallError",
    "GeminiClient",
]

__version__ = "0.1.0"
