"""Phase 3 demo: force failures and watch the router escalate — capped, logged.

Run:
    python examples/demo_fallback.py

Uses a fake model client (no network, no API key) whose failure behaviour we
control per model, so we can demonstrate exactly:

  1. a transient 5xx on the cheap model  -> escalate ONE tier -> success
  2. a failed output verification         -> escalate via the same path
  3. a systemic failure                   -> escalation stops at the cap
     (this is the scenario that, uncapped, tripled a real bill)

Every escalation is logged by the router at WARNING; we also print a summary.
"""

from __future__ import annotations

import asyncio
import logging

from llmrouter import Router, RouterConfig
from llmrouter.client import ModelCallError, ModelResponse
from llmrouter.registry import ModelSpec

logging.basicConfig(level=logging.WARNING, format="  [log] %(name)s: %(message)s")


class ScriptedClient:
    """Fake client. ``fail_tiers`` maps a tier name to the behaviour for models
    in that tier: '5xx' raises transient, 'bad-json' returns unverifiable text,
    anything else returns a normal response."""

    def __init__(self, fail_tiers: dict[str, str]) -> None:
        self.fail_tiers = fail_tiers
        self.calls: list[str] = []

    async def acomplete(
        self, model: ModelSpec, query: str, metadata: dict | None = None
    ) -> ModelResponse:
        self.calls.append(model.name)
        behaviour = self.fail_tiers.get(model.tier.value)
        if behaviour == "5xx":
            raise ModelCallError(
                f"{model.name} unavailable", status_code=503
            )
        if behaviour == "bad-json":
            return ModelResponse(text="Sure! Here you go:", input_tokens=20, output_tokens=8)
        return ModelResponse(
            text='{"answer": "42"}', input_tokens=20, output_tokens=6
        )


def summarise(title: str, result) -> None:
    print(f"\n=== {title} ===")
    print(f"  success      : {result.success}")
    print(f"  final model  : {result.final_decision.model.name} "
          f"({result.final_decision.tier.value})")
    print(f"  attempts     : {result.attempts}")
    print(f"  escalations  : {len(result.escalations)}")
    for e in result.escalations:
        print(f"     - [{e.trigger}] {e.from_tier.value}->{e.to_tier.value} "
              f"({e.from_model} -> {e.to_model})")
    print(f"  total cost   : ${result.total_cost:.6f}")
    print(f"  latency      : {result.latency_ms:.1f} ms")
    if result.error:
        print(f"  error        : {result.error}")


async def main() -> None:
    cfg = RouterConfig(max_escalation_depth=2)
    q = "Extract the answer as JSON."  # 'extract' -> rules route to CHEAP
    schema = {"check": "json_schema", "required": ["answer"]}

    # 1. cheap tier 5xx -> escalate one tier -> medium succeeds
    r1 = Router.from_config(cfg, ScriptedClient({"cheap": "5xx"}))
    summarise("1. Transient 5xx on cheap tier", await r1.acomplete(q))

    # 2. cheap returns unverifiable output -> escalate on failed verification
    r2 = Router.from_config(cfg, ScriptedClient({"cheap": "bad-json"}))
    summarise(
        "2. Failed verification on cheap tier",
        await r2.acomplete(q, verify_spec=schema),
    )

    # 3. every tier 5xx -> escalate up to the cap, then give up (no infinite loop)
    r3 = Router.from_config(cfg, ScriptedClient({"cheap": "5xx", "medium": "5xx", "frontier": "5xx"}))
    res3 = await r3.acomplete(q)
    summarise("3. Systemic failure - stops at the cap", res3)
    print(f"\n  (cap = {cfg.max_escalation_depth}; models actually called: "
          f"{r3._client.calls})")


if __name__ == "__main__":
    asyncio.run(main())
