"""Tests for per-route metrics, savings, alerts, and persistence."""

from __future__ import annotations

import pytest

from llmrouter import RouterConfig, Tier
from llmrouter.metrics import (
    RouteMetrics,
    load_metrics,
    save_metrics,
    simulate_metrics,
)


@pytest.fixture(scope="module")
def registry():
    return RouterConfig().load_registry()


def _mk(registry, threshold=0.25) -> RouteMetrics:
    return RouteMetrics(registry=registry, alert_fallback_threshold=threshold)


def test_by_tier_and_route_counts(registry) -> None:
    m = _mk(registry)
    # 4 cheap (1 escalated), 2 medium (0 escalated)
    for i in range(4):
        m.record("gemini-3.1-flash-lite", Tier.CHEAP, "gemini-3.1-flash-lite",
                 cost=0.001, latency_ms=300, escalated=(i == 0),
                 classifier_used="rules")
    for _ in range(2):
        m.record("gemini-3-flash-preview", Tier.MEDIUM, "gemini-3-flash-preview",
                 cost=0.002, latency_ms=600, escalated=False,
                 classifier_used="embedding")

    by_tier = m.by_tier()
    assert by_tier[Tier.CHEAP].volume == 4
    assert by_tier[Tier.CHEAP].escalations == 1
    assert by_tier[Tier.CHEAP].fallback_rate == pytest.approx(0.25)
    assert by_tier[Tier.MEDIUM].fallback_rate == 0.0
    assert by_tier[Tier.CHEAP].volume_share == pytest.approx(4 / 6)

    by_route = m.by_route()
    assert by_route["gemini-3.1-flash-lite"].fallback_rate == pytest.approx(0.25)
    assert m.total_requests == 6


def test_fallback_rate_tier_and_overall(registry) -> None:
    m = _mk(registry)
    for i in range(10):
        m.record("gemini-3.1-flash-lite", Tier.CHEAP, "x", 0.001, 300,
                 escalated=(i < 3), classifier_used="rules")
    assert m.fallback_rate(Tier.CHEAP) == pytest.approx(0.3)
    assert m.overall_fallback_rate() == pytest.approx(0.3)
    assert m.fallback_rate(Tier.FRONTIER) == 0.0  # no records for tier


def test_savings_vs_baseline_uses_registry_prices(registry) -> None:
    m = _mk(registry)
    # One cheap request; baseline = cheapest frontier model.
    m.record("gemini-3.1-flash-lite", Tier.CHEAP, "gemini-3.1-flash-lite",
             cost=registry.cheapest_in_tier(Tier.CHEAP).cost_for(1000, 200),
             latency_ms=300, escalated=False, classifier_used="rules",
             input_tokens=1000, output_tokens=200)
    sav = m.savings_vs_baseline("frontier")
    frontier = registry.cheapest_in_tier(Tier.FRONTIER)
    assert sav.baseline == frontier.name
    assert sav.baseline_cost == pytest.approx(frontier.cost_for(1000, 200))
    assert sav.saved > 0  # cheap is cheaper than frontier
    assert 0 < sav.pct < 1


def test_alert_check_fires_over_threshold(registry) -> None:
    m = _mk(registry, threshold=0.25)
    # cheap tier: 9/10 escalated -> 90% >> 25%
    for i in range(10):
        m.record("gemini-3.1-flash-lite", Tier.CHEAP, "gemini-3-flash-preview",
                 0.003, 500, escalated=(i > 0), classifier_used="rules")
    alerts = m.alert_check()
    assert any(a.scope == "tier" and a.key == "cheap" for a in alerts)
    assert any(a.scope == "route" for a in alerts)
    assert all(a.fallback_rate > a.threshold for a in alerts)


def test_alert_check_silent_when_healthy(registry) -> None:
    m = _mk(registry, threshold=0.25)
    for _ in range(20):
        m.record("gemini-3.1-flash-lite", Tier.CHEAP, "gemini-3.1-flash-lite",
                 0.001, 300, escalated=False, classifier_used="rules")
    assert m.alert_check() == []


def test_fallback_series_rises_during_spike(registry) -> None:
    m = simulate_metrics(registry, n_healthy=200, n_spike=120,
                         cheap_fail_spike=0.9)
    series = m.fallback_rate_series()
    assert len(series) == m.total_requests
    # The rolling rate late in the sequence (spike) is well above the start.
    assert series[-1]["fallback_rate"] > series[0]["fallback_rate"] + 0.2


def test_simulated_spike_triggers_alert(registry) -> None:
    m = simulate_metrics(registry, cheap_fail_spike=0.9,
                         alert_fallback_threshold=0.25)
    alerts = m.alert_check()
    assert any(a.key == "cheap" for a in alerts)


def test_save_and_load_roundtrip(registry, tmp_path) -> None:
    m = simulate_metrics(registry, n_healthy=30, n_spike=20)
    path = tmp_path / "store.json"
    save_metrics(m, path)
    loaded = load_metrics(path, registry=registry)
    assert loaded is not None
    assert loaded.total_requests == m.total_requests
    assert loaded.snapshot()["by_tier"].keys() == m.snapshot()["by_tier"].keys()


def test_load_missing_store_returns_none(registry, tmp_path) -> None:
    assert load_metrics(tmp_path / "nope.json", registry=registry) is None


def test_record_result_marks_escalation(registry) -> None:
    # Build a RouteResult with an escalation and confirm the route == entry.
    import asyncio

    from llmrouter import ModelCallError, ModelResponse, Router

    class FailCheap:
        async def acomplete(self, model, query, metadata=None):
            if model.tier is Tier.CHEAP:
                raise ModelCallError("down", status_code=503)
            return ModelResponse(text="ok", input_tokens=10, output_tokens=5)

    router = Router.from_config(RouterConfig(), FailCheap())
    result = asyncio.run(router.acomplete("Extract as JSON."))  # -> cheap entry
    m = _mk(registry)
    m.record_result(result)
    stats = m.by_route()
    # Entry point is the cheap model even though the medium model served it.
    assert "gemini-3.1-flash-lite" in stats
    assert stats["gemini-3.1-flash-lite"].fallback_rate == 1.0
    assert stats["gemini-3.1-flash-lite"].tier is Tier.CHEAP
