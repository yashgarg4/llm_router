"""Metrics REST API — read-only views over per-route metrics.

Endpoints:
    GET /metrics    — overall snapshot (cost, fallback rate, savings, by-tier, alerts)
    GET /by-route   — per-route table (the panel that catches cascades)
    GET /by-tier    — per-tier aggregates
    GET /alerts     — routes/tiers over the fallback-rate threshold
    GET /timeseries — rolling fallback rate over the request sequence
    GET /health     — liveness

Data source: the shared JSON store (``metrics_store.json``) written by the
proxy (Phase 5) or the dashboard's simulation. If the store is absent on
startup, a simulated workload (with the incident spike) is generated so the
API is never empty in a demo.

Run:
    uvicorn server.dashboard:app --reload
"""

from __future__ import annotations

from fastapi import FastAPI

from llmrouter import RouterConfig
from llmrouter.metrics import RouteMetrics, load_metrics, save_metrics, simulate_metrics

app = FastAPI(title="llmrouter metrics", version="0.1.0")

_registry = RouterConfig().load_registry()


def _metrics() -> RouteMetrics:
    """Load the current store; simulate + persist one if none exists yet."""
    m = load_metrics(registry=_registry)
    if m is None:
        m = simulate_metrics(_registry)
        save_metrics(m)
    return m


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/metrics")
def metrics() -> dict:
    return _metrics().snapshot()


@app.get("/by-route")
def by_route() -> list[dict]:
    stats = _metrics().by_route().values()
    # Most-escalating routes first — the ones worth looking at.
    return [
        s.model_dump(mode="json")
        for s in sorted(stats, key=lambda s: s.fallback_rate, reverse=True)
    ]


@app.get("/by-tier")
def by_tier() -> list[dict]:
    return [s.model_dump(mode="json") for s in _metrics().by_tier().values()]


@app.get("/alerts")
def alerts() -> dict:
    m = _metrics()
    found = m.alert_check()
    return {
        "threshold": m.alert_fallback_threshold,
        "count": len(found),
        "alerts": [a.model_dump() for a in found],
    }


@app.get("/timeseries")
def timeseries(window: int | None = None) -> list[dict]:
    return _metrics().fallback_rate_series(window=window)
