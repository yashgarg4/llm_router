"""Tests for escalation (fallback), verification, and the acomplete loop.

Coroutines are driven with ``asyncio.run`` inside sync tests so we need no
pytest-asyncio dependency. Model calls use a scripted fake client — the
escalation logic is what's under test, not any real provider.
"""

from __future__ import annotations

import asyncio

import pytest

from llmrouter import (
    FallbackPolicy,
    ModelCallError,
    ModelResponse,
    Router,
    RouterConfig,
    Tier,
    verify,
)
from llmrouter.registry import ModelSpec


# -- scripted fake client ----------------------------------------------------


class ScriptedClient:
    def __init__(self, behaviour: dict[str, str]) -> None:
        # behaviour: tier -> "5xx" | "4xx" | "bad-json" | "ok"
        self.behaviour = behaviour
        self.calls: list[str] = []

    async def acomplete(
        self, model: ModelSpec, query: str, metadata: dict | None = None
    ) -> ModelResponse:
        self.calls.append(model.name)
        b = self.behaviour.get(model.tier.value, "ok")
        if b == "5xx":
            raise ModelCallError(f"{model.name} down", status_code=503)
        if b == "4xx":
            raise ModelCallError(f"{model.name} bad request", status_code=400)
        if b == "bad-json":
            return ModelResponse(text="not json at all", input_tokens=5, output_tokens=3)
        return ModelResponse(text='{"answer": "ok"}', input_tokens=5, output_tokens=3)


def run(coro):
    return asyncio.run(coro)


def make_router(behaviour: dict[str, str], **cfg_kwargs) -> Router:
    cfg = RouterConfig(**cfg_kwargs)
    return Router.from_config(cfg, ScriptedClient(behaviour))


# -- FallbackPolicy unit tests ----------------------------------------------


def test_cap_must_be_non_negative_int() -> None:
    reg = RouterConfig().load_registry()
    with pytest.raises(AssertionError):
        FallbackPolicy(reg, max_escalation_depth=-1)


def test_on_failure_escalates_one_tier() -> None:
    reg = RouterConfig().load_registry()
    fp = FallbackPolicy(reg, max_escalation_depth=2)
    from llmrouter.registry import RouteDecision

    start = RouteDecision(
        model=reg.cheapest_in_tier(Tier.CHEAP), tier=Tier.CHEAP,
        reason="x", classifier_used="rules",
    )
    nxt = fp.on_failure(start, "boom", depth=0)
    assert nxt is not None and nxt.tier is Tier.MEDIUM
    assert nxt.classifier_used == "fallback"


def test_on_failure_returns_none_at_frontier() -> None:
    reg = RouterConfig().load_registry()
    fp = FallbackPolicy(reg, max_escalation_depth=5)
    from llmrouter.registry import RouteDecision

    start = RouteDecision(
        model=reg.cheapest_in_tier(Tier.FRONTIER), tier=Tier.FRONTIER,
        reason="x", classifier_used="rules",
    )
    assert fp.on_failure(start, "boom", depth=0) is None


def test_on_failure_returns_none_at_cap() -> None:
    reg = RouterConfig().load_registry()
    fp = FallbackPolicy(reg, max_escalation_depth=1)
    from llmrouter.registry import RouteDecision

    start = RouteDecision(
        model=reg.cheapest_in_tier(Tier.CHEAP), tier=Tier.CHEAP,
        reason="x", classifier_used="rules",
    )
    assert fp.on_failure(start, "boom", depth=1) is None  # depth == cap


# -- acomplete escalation behaviour -----------------------------------------


def test_transient_5xx_escalates_one_tier_to_success() -> None:
    r = make_router({"cheap": "5xx"})
    res = run(r.acomplete("Extract the answer as JSON."))  # 'extract' -> cheap
    assert res.success
    assert res.final_decision.tier is Tier.MEDIUM
    assert len(res.escalations) == 1
    assert res.escalations[0].trigger == "error"
    assert res.attempts == 2


def test_escalation_stops_at_cap_no_infinite_loop() -> None:
    r = make_router(
        {"cheap": "5xx", "medium": "5xx", "frontier": "5xx"},
        max_escalation_depth=2,
    )
    res = run(r.acomplete("Extract the answer as JSON."))
    assert res.success is False
    assert len(res.escalations) == 2  # capped
    # Exactly three models tried (initial + 2 escalations), never more.
    assert r._client.calls == [
        "gemini-3.1-flash-lite", "gemini-3-flash", "gemini-3.5-flash",
    ]


def test_frontier_failure_returns_error_not_loop() -> None:
    # Force routing to frontier (keyword 'design'); frontier 5xx -> give up.
    r = make_router({"frontier": "5xx"})
    res = run(r.acomplete("Design a distributed lock."))
    assert res.success is False
    assert res.escalations == []  # nowhere above frontier to escalate
    assert res.error is not None


def test_non_transient_4xx_does_not_escalate() -> None:
    r = make_router({"cheap": "4xx"})
    res = run(r.acomplete("Extract the answer as JSON."))
    assert res.success is False
    assert res.escalations == []  # 4xx must not escalate
    assert r._client.calls == ["gemini-3.1-flash-lite"]


def test_verification_failure_escalates_and_is_recorded(caplog) -> None:
    import logging

    r = make_router({"cheap": "bad-json"})  # cheap returns unverifiable text
    with caplog.at_level(logging.WARNING, logger="llmrouter.router"):
        res = run(
            r.acomplete(
                "Extract the answer as JSON.",
                verify_spec={"check": "json_schema", "required": ["answer"]},
            )
        )
    assert res.success
    assert res.final_decision.tier is Tier.MEDIUM
    assert len(res.escalations) == 1
    assert res.escalations[0].trigger == "verification"
    # The verification-triggered escalation must be logged (cascade vector).
    assert any("VERIFICATION-TRIGGERED ESCALATION" in m for m in caplog.messages)


def test_success_records_cost_from_registry_prices() -> None:
    r = make_router({})  # everything ok
    res = run(r.acomplete("Extract the answer as JSON."))
    model = res.final_decision.model
    expected = model.cost_for(5, 3)
    assert res.total_cost == pytest.approx(expected)


# -- verify.py unit tests ----------------------------------------------------


def _resp(text: str) -> ModelResponse:
    return ModelResponse(text=text)


def test_verify_none_spec_passes() -> None:
    assert verify(_resp(""), None) is True


def test_verify_non_empty() -> None:
    assert verify(_resp("hello"), {"check": "non_empty"}) is True
    assert verify(_resp("   "), {"check": "non_empty"}) is False


def test_verify_json() -> None:
    assert verify(_resp('{"a": 1}'), {"check": "json"}) is True
    assert verify(_resp("nope"), {"check": "json"}) is False


def test_verify_json_fenced_block() -> None:
    fenced = '```json\n{"a": 1}\n```'
    assert verify(_resp(fenced), {"check": "json"}) is True


def test_verify_json_schema_required_and_types() -> None:
    spec = {"check": "json_schema", "required": ["a", "b"],
            "types": {"a": "str", "b": "int"}}
    assert verify(_resp('{"a": "x", "b": 3}'), spec) is True
    assert verify(_resp('{"a": "x"}'), spec) is False           # missing b
    assert verify(_resp('{"a": 1, "b": 3}'), spec) is False      # a wrong type
    assert verify(_resp('{"a": "x", "b": true}'), spec) is False  # bool != int


def test_verify_unknown_check_raises() -> None:
    with pytest.raises(ValueError):
        verify(_resp("x"), {"check": "nonsense"})
