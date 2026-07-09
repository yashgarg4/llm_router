"""Per-route metrics — the component that catches cascade explosions.

This is not an afterthought bolted onto the router; it is the reason the
router is safe to run. The origin incident (INTERNAL_NOTES §1) was invisible
because nobody was watching fallback rate *per route*. So the core question
this module answers is: "for each route the router uses, how often is it
escalating, and is that rate abnormal?"

Definitions used here:
  * **route**  — the *entry point* a request was routed to: the initially
    chosen model (and its tier), *before* any escalation. This is the unit
    whose health we monitor, because a broken cheap-tier verifier shows up as
    a spike in the cheap route's fallback rate — exactly the incident.
  * **escalated** — whether that request had to climb at least one tier.
  * **fallback rate** — escalations / requests, for a route or a tier. The
    metric with an alert threshold (default 0.25).

`cost` is the *total* cost of serving the request (summed across every model
it touched, from `RouteResult.total_cost`), attributed to the route it
entered through — so an escalating route correctly shows its true, inflated
cost.
"""

from __future__ import annotations

import json
import random
import time
from collections import defaultdict
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from .registry import ModelRegistry, Tier
from .router import RouteResult

# Rough per-tier base latency (ms) for the simulation only — real latency is
# measured, not assumed. Cheaper tiers are faster.
_SIM_BASE_LATENCY = {Tier.CHEAP: 300.0, Tier.MEDIUM: 650.0, Tier.FRONTIER: 1300.0}


class RequestRecord(BaseModel):
    """One served request."""

    model_config = ConfigDict(frozen=True)

    seq: int
    route: str          # entry-point model name (initial route)
    tier: Tier          # entry-point tier
    final_model: str    # model that actually produced the response
    cost: float
    latency_ms: float
    escalated: bool
    classifier_used: str
    input_tokens: int = 0
    output_tokens: int = 0
    ts: float = 0.0


class RouteStats(BaseModel):
    """Aggregated stats for one route (entry point)."""

    route: str
    tier: Tier
    volume: int
    total_cost: float
    avg_cost: float
    avg_latency_ms: float
    escalations: int
    fallback_rate: float


class TierStats(BaseModel):
    """Aggregated stats for one tier."""

    tier: Tier
    volume: int
    total_cost: float
    avg_latency_ms: float
    escalations: int
    fallback_rate: float
    volume_share: float  # fraction of all requests (for the tier pie)


class Alert(BaseModel):
    """A route/tier whose fallback rate breached the alert threshold."""

    scope: str  # "tier" | "route"
    key: str
    fallback_rate: float
    threshold: float
    volume: int
    message: str


class Savings(BaseModel):
    """Cost saved vs sending everything to a baseline model."""

    baseline: str
    baseline_cost: float
    actual_cost: float
    saved: float
    pct: float


class RouteMetrics:
    """In-memory per-route metrics store. Thread-unsafe by design (wrap in the
    app's own lock if shared); the proxy records from one request at a time."""

    def __init__(
        self,
        registry: ModelRegistry | None = None,
        alert_fallback_threshold: float = 0.25,
    ) -> None:
        self._registry = registry
        self.alert_fallback_threshold = alert_fallback_threshold
        self._records: list[RequestRecord] = []
        self._seq = 0

    # -- recording ----------------------------------------------------------

    def record(
        self,
        route: str,
        tier: Tier,
        model: str,
        cost: float,
        latency_ms: float,
        escalated: bool,
        classifier_used: str,
        input_tokens: int = 0,
        output_tokens: int = 0,
        ts: float | None = None,
    ) -> None:
        """Record one served request (low-level; see :meth:`record_result`)."""
        self._records.append(
            RequestRecord(
                seq=self._seq,
                route=route,
                tier=tier,
                final_model=model,
                cost=cost,
                latency_ms=latency_ms,
                escalated=escalated,
                classifier_used=classifier_used,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                ts=ts if ts is not None else time.time(),
            )
        )
        self._seq += 1

    def record_result(self, result: RouteResult) -> None:
        """Record a :class:`RouteResult` from ``Router.acomplete``.

        The route (entry point) is the initial decision: if the request
        escalated, that's the first escalation's ``from_*``; otherwise it's the
        final decision (no escalation happened)."""
        if result.escalations:
            route = result.escalations[0].from_model
            tier = result.escalations[0].from_tier
        else:
            route = result.final_decision.model.name
            tier = result.final_decision.tier
        resp = result.response
        self.record(
            route=route,
            tier=tier,
            model=result.final_decision.model.name,
            cost=result.total_cost,
            latency_ms=result.latency_ms,
            escalated=result.escalated,
            classifier_used=result.final_decision.classifier_used,
            input_tokens=resp.input_tokens if resp else 0,
            output_tokens=resp.output_tokens if resp else 0,
        )

    # -- aggregation --------------------------------------------------------

    @property
    def total_requests(self) -> int:
        return len(self._records)

    def total_cost(self) -> float:
        return sum(r.cost for r in self._records)

    def by_route(self) -> dict[str, RouteStats]:
        groups: dict[str, list[RequestRecord]] = defaultdict(list)
        for r in self._records:
            groups[r.route].append(r)
        out: dict[str, RouteStats] = {}
        for route, recs in groups.items():
            vol = len(recs)
            esc = sum(1 for r in recs if r.escalated)
            cost = sum(r.cost for r in recs)
            out[route] = RouteStats(
                route=route,
                tier=recs[0].tier,
                volume=vol,
                total_cost=cost,
                avg_cost=cost / vol,
                avg_latency_ms=sum(r.latency_ms for r in recs) / vol,
                escalations=esc,
                fallback_rate=esc / vol,
            )
        return out

    def by_tier(self) -> dict[Tier, TierStats]:
        total = len(self._records) or 1
        groups: dict[Tier, list[RequestRecord]] = defaultdict(list)
        for r in self._records:
            groups[r.tier].append(r)
        out: dict[Tier, TierStats] = {}
        for tier, recs in groups.items():
            vol = len(recs)
            esc = sum(1 for r in recs if r.escalated)
            cost = sum(r.cost for r in recs)
            out[tier] = TierStats(
                tier=tier,
                volume=vol,
                total_cost=cost,
                avg_latency_ms=sum(r.latency_ms for r in recs) / vol,
                escalations=esc,
                fallback_rate=esc / vol,
                volume_share=vol / total,
            )
        return out

    def fallback_rate(self, tier: Tier | None = None) -> float:
        """Fallback rate for a tier, or overall if ``tier`` is None."""
        recs = (
            self._records if tier is None
            else [r for r in self._records if r.tier == tier]
        )
        if not recs:
            return 0.0
        return sum(1 for r in recs if r.escalated) / len(recs)

    def overall_fallback_rate(self) -> float:
        return self.fallback_rate(None)

    # -- savings ------------------------------------------------------------

    def savings_vs_baseline(self, baseline: str = "frontier") -> Savings:
        """Cost saved vs routing every request to a baseline model.

        ``baseline`` may be a tier name (uses the cheapest model in that tier)
        or an exact model name. Needs a registry and per-request token counts;
        returns zeros if either is missing.
        """
        if self._registry is None:
            return Savings(baseline=baseline, baseline_cost=0, actual_cost=self.total_cost(), saved=0, pct=0)

        baseline_model = self._resolve_baseline(baseline)
        baseline_cost = sum(
            baseline_model.cost_for(r.input_tokens, r.output_tokens)
            for r in self._records
        )
        actual = self.total_cost()
        saved = baseline_cost - actual
        pct = (saved / baseline_cost) if baseline_cost else 0.0
        return Savings(
            baseline=baseline_model.name,
            baseline_cost=baseline_cost,
            actual_cost=actual,
            saved=saved,
            pct=pct,
        )

    def _resolve_baseline(self, baseline: str):
        try:
            tier = Tier(baseline.lower())
            return self._registry.cheapest_in_tier(tier)  # type: ignore[union-attr]
        except ValueError:
            return self._registry.get(baseline)  # type: ignore[union-attr]

    # -- alerting -----------------------------------------------------------

    def alert_check(self, include_routes: bool = True) -> list[Alert]:
        """Return alerts for tiers (and, by default, routes) whose fallback
        rate exceeds the configured threshold — the cascade detector."""
        alerts: list[Alert] = []
        thr = self.alert_fallback_threshold
        for tier, s in self.by_tier().items():
            if s.fallback_rate > thr:
                alerts.append(Alert(
                    scope="tier", key=tier.value, fallback_rate=s.fallback_rate,
                    threshold=thr, volume=s.volume,
                    message=(
                        f"tier '{tier.value}' fallback rate "
                        f"{s.fallback_rate:.0%} exceeds threshold {thr:.0%} "
                        f"({s.escalations}/{s.volume} escalated)"
                    ),
                ))
        if include_routes:
            for route, s in self.by_route().items():
                if s.fallback_rate > thr:
                    alerts.append(Alert(
                        scope="route", key=route, fallback_rate=s.fallback_rate,
                        threshold=thr, volume=s.volume,
                        message=(
                            f"route '{route}' fallback rate "
                            f"{s.fallback_rate:.0%} exceeds threshold {thr:.0%} "
                            f"({s.escalations}/{s.volume} escalated)"
                        ),
                    ))
        return alerts

    # -- timeseries (dashboard line chart) ---------------------------------

    def fallback_rate_series(self, window: int | None = None) -> list[dict]:
        """Rolling fallback rate over the request sequence, for the dashboard's
        'fallback rate over time' line. A rising line crossing the threshold is
        the cascade signature."""
        n = len(self._records)
        if n == 0:
            return []
        if window is None:
            window = max(5, n // 10)
        series = []
        ordered = sorted(self._records, key=lambda r: r.seq)
        for i in range(n):
            lo = max(0, i - window + 1)
            win = ordered[lo : i + 1]
            rate = sum(1 for r in win if r.escalated) / len(win)
            series.append({"seq": ordered[i].seq, "fallback_rate": rate})
        return series

    # -- serialisation (REST + persistence) --------------------------------

    def snapshot(self) -> dict:
        """A JSON-serialisable summary for the REST API / dashboard."""
        return {
            "total_requests": self.total_requests,
            "total_cost": self.total_cost(),
            "overall_fallback_rate": self.overall_fallback_rate(),
            "alert_threshold": self.alert_fallback_threshold,
            "savings": self.savings_vs_baseline().model_dump(),
            "by_tier": {t.value: s.model_dump(mode="json")
                        for t, s in self.by_tier().items()},
            "alerts": [a.model_dump() for a in self.alert_check()],
        }

    def records_as_dicts(self) -> list[dict]:
        return [r.model_dump(mode="json") for r in self._records]

    def load_records(self, records: list[dict]) -> None:
        """Replace records from serialised dicts (dashboard persistence)."""
        self._records = [RequestRecord.model_validate(r) for r in records]
        self._seq = (max((r.seq for r in self._records), default=-1)) + 1


# ---------------------------------------------------------------------------
# Simulation — reproduces the origin incident for the demo / dashboard.
# ---------------------------------------------------------------------------

def simulate_metrics(
    registry: ModelRegistry,
    n_healthy: int = 200,
    n_spike: int = 120,
    cheap_fail_healthy: float = 0.05,
    cheap_fail_spike: float = 0.90,
    tier_mix: tuple[float, float, float] = (0.70, 0.20, 0.10),
    alert_fallback_threshold: float = 0.25,
    seed: int = 7,
) -> RouteMetrics:
    """Build a populated :class:`RouteMetrics` with a healthy period followed
    by a cheap-tier failure spike.

    This is exactly the incident: a cheap-tier verifier starts failing, so the
    cheap route escalates most of its traffic to medium/frontier. On the
    dashboard the rolling fallback line rises and crosses the alert threshold —
    the panel that would have caught the real 90% escalation.
    """
    rng = random.Random(seed)
    m = RouteMetrics(registry=registry, alert_fallback_threshold=alert_fallback_threshold)
    tiers = [Tier.CHEAP, Tier.MEDIUM, Tier.FRONTIER]

    def emit(cheap_fail_rate: float) -> None:
        tier = rng.choices(tiers, weights=tier_mix, k=1)[0]
        entry = registry.cheapest_in_tier(tier)
        in_tok = rng.randint(400, 1500)
        out_tok = rng.randint(80, 400)
        # Only the cheap tier is failing during the spike; others stay healthy.
        fail_p = cheap_fail_rate if tier is Tier.CHEAP else 0.03
        escalated = rng.random() < fail_p and tier.next_up() is not None

        latency = _SIM_BASE_LATENCY[tier] * rng.uniform(0.8, 1.3)
        if escalated:
            nxt_tier = tier.next_up()
            final = registry.cheapest_in_tier(nxt_tier)
            # cost = failed cheap attempt (verify-fail still billed) + retry
            cost = entry.cost_for(in_tok, out_tok) + final.cost_for(in_tok, out_tok)
            latency += _SIM_BASE_LATENCY[nxt_tier] * rng.uniform(0.8, 1.3)
        else:
            final = entry
            cost = entry.cost_for(in_tok, out_tok)

        m.record(
            route=entry.name,
            tier=tier,
            model=final.name,
            cost=cost,
            latency_ms=latency,
            escalated=escalated,
            classifier_used=rng.choice(["rules", "embedding"]),
            input_tokens=in_tok,
            output_tokens=out_tok,
        )

    for _ in range(n_healthy):
        emit(cheap_fail_healthy)
    for _ in range(n_spike):
        emit(cheap_fail_spike)
    return m


# ---------------------------------------------------------------------------
# Persistence — a tiny JSON store shared by the REST server and the dashboard.
# ---------------------------------------------------------------------------

DEFAULT_STORE = Path("metrics_store.json")


def save_metrics(m: RouteMetrics, path: str | Path = DEFAULT_STORE) -> None:
    Path(path).write_text(
        json.dumps(
            {
                "alert_fallback_threshold": m.alert_fallback_threshold,
                "records": m.records_as_dicts(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def load_metrics(
    path: str | Path = DEFAULT_STORE, registry: ModelRegistry | None = None
) -> RouteMetrics | None:
    """Load a persisted store, or None if the file doesn't exist."""
    path = Path(path)
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    m = RouteMetrics(
        registry=registry,
        alert_fallback_threshold=data.get("alert_fallback_threshold", 0.25),
    )
    m.load_records(data.get("records", []))
    return m
