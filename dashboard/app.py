"""Streamlit per-route metrics dashboard — the cascade-explosion detector.

Run:
    streamlit run dashboard/app.py

The sidebar drives a simulated workload: a healthy period followed by a
cheap-tier failure spike whose severity you control. Push the "cheap-tier
failure rate" slider up and watch the rolling fallback-rate line rise and
cross the red alert threshold — this is the panel that would have caught the
real 90%-escalation incident before the invoice did.

If ``metrics_store.json`` exists (written by the proxy), the dashboard loads
real metrics instead; use the sidebar toggle to switch to the simulation.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from llmrouter import RouterConfig, Tier
from llmrouter.metrics import RouteMetrics, load_metrics, save_metrics, simulate_metrics

st.set_page_config(page_title="llmrouter metrics", layout="wide")

registry = RouterConfig().load_registry()


@st.cache_data
def _build(source: str, spike_rate: float, spike_n: int, threshold: float) -> dict:
    """Build (or load) metrics and return a JSON-safe payload. Cached so the
    charts don't rebuild on every widget interaction."""
    if source == "Live store (metrics_store.json)":
        m = load_metrics(registry=registry)
        if m is None:
            st.warning("No metrics_store.json found — showing a simulation instead.")
            m = simulate_metrics(registry, alert_fallback_threshold=threshold)
    else:
        m = simulate_metrics(
            registry,
            n_spike=spike_n,
            cheap_fail_spike=spike_rate,
            alert_fallback_threshold=threshold,
        )
        save_metrics(m)  # so the REST API reflects what the dashboard shows
    return {
        "snapshot": m.snapshot(),
        "by_route": {k: v.model_dump(mode="json") for k, v in m.by_route().items()},
        "by_tier": {k.value: v.model_dump(mode="json") for k, v in m.by_tier().items()},
        "series": m.fallback_rate_series(),
    }


# -- sidebar controls --------------------------------------------------------

st.sidebar.title("llmrouter")
source = st.sidebar.radio(
    "Data source",
    ["Simulation", "Live store (metrics_store.json)"],
)
threshold = st.sidebar.slider("Alert fallback threshold", 0.05, 0.90, 0.25, 0.05)
spike_rate = st.sidebar.slider(
    "Cheap-tier failure rate (spike)", 0.0, 1.0, 0.90, 0.05,
    help="Simulates a failing cheap-tier verifier — the origin incident.",
)
spike_n = st.sidebar.slider("Spike volume (requests)", 0, 400, 120, 20)

data = _build(source, spike_rate, spike_n, threshold)
snap = data["snapshot"]

# -- KPIs --------------------------------------------------------------------

st.title("Per-route metrics")
sav = snap["savings"]
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total cost", f"${snap['total_cost']:.4f}")
c2.metric(
    f"Saved vs all-{sav['baseline']}",
    f"${sav['saved']:.4f}",
    f"{sav['pct'] * 100:.0f}%",
)
overall = snap["overall_fallback_rate"]
c3.metric(
    "Overall fallback rate",
    f"{overall * 100:.1f}%",
    delta=f"{(overall - snap['alert_threshold']) * 100:+.0f}% vs threshold",
    delta_color="inverse",
)
c4.metric("Requests", f"{snap['total_requests']:,}")

# -- alerts banner -----------------------------------------------------------

if snap["alerts"]:
    for a in snap["alerts"]:
        st.error(f"🚨 {a['message']}")
else:
    st.success("✅ No routes over the fallback-rate threshold.")

left, right = st.columns([1, 2])

# -- tier distribution pie ---------------------------------------------------

with left:
    st.subheader("Tier distribution")
    tier_rows = [
        {"tier": t, "volume": s["volume"], "share": s["volume_share"]}
        for t, s in data["by_tier"].items()
    ]
    tier_df = pd.DataFrame(tier_rows).set_index("tier")
    st.bar_chart(tier_df["volume"])
    st.caption(
        "Share of traffic by tier: "
        + ", ".join(f"{r['tier']} {r['share'] * 100:.0f}%" for r in tier_rows)
    )

# -- fallback rate over time, with threshold line ----------------------------

with right:
    st.subheader("Fallback rate over time (cascade detector)")
    series = data["series"]
    if series:
        sdf = pd.DataFrame(series).set_index("seq")
        sdf["alert threshold"] = snap["alert_threshold"]
        sdf = sdf.rename(columns={"fallback_rate": "fallback rate"})
        st.line_chart(sdf[["fallback rate", "alert threshold"]])
        st.caption(
            "A rising line that crosses the threshold is the cascade "
            "signature — a tier silently escalating most of its traffic."
        )

# -- per-route table ---------------------------------------------------------

st.subheader("Per-route table")
st.caption(
    "The panel that would have caught the 90% escalation. "
    "Rows over the threshold are the ones to investigate."
)
route_rows = sorted(
    data["by_route"].values(), key=lambda r: r["fallback_rate"], reverse=True
)
table = pd.DataFrame([
    {
        "route": r["route"],
        "tier": r["tier"],
        "volume": r["volume"],
        "avg cost": round(r["avg_cost"], 6),
        "avg latency (ms)": round(r["avg_latency_ms"], 0),
        "fallback rate": r["fallback_rate"],
    }
    for r in route_rows
])


def _highlight(row):
    over = row["fallback rate"] > snap["alert_threshold"]
    return ["background-color: #5c1a1a" if over else "" for _ in row]


st.dataframe(
    table.style.apply(_highlight, axis=1).format({
        "avg cost": "${:.6f}",
        "fallback rate": "{:.0%}",
        "avg latency (ms)": "{:.0f}",
    }),
    use_container_width=True,
)
