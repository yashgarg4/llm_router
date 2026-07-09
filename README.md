# llmrouter

**A cost- and quality-aware LLM router with per-route observability.**

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![pydantic](https://img.shields.io/badge/pydantic-v2-e92063)
![License](https://img.shields.io/badge/license-MIT-green)
![Status](https://img.shields.io/badge/status-Phase%201-orange)
![Tests](https://img.shields.io/badge/tests-21%20passing-brightgreen)

`llmrouter` sits between your application and multiple LLMs. For every request
it classifies complexity, routes to the *cheapest model that is good enough*,
escalates on failure (capped and logged), and records cost, latency, and
fallback rate **per route** — so a runaway escalation shows up on a dashboard
before it shows up on the invoice.

> **Why per-route metrics are the whole point.** A cost router without
> observability is a liability. In a real incident, a cheap-tier verifier
> broke after a provider tweaked its output format; the router silently
> escalated ~90% of traffic to the frontier model and the bill tripled with
> no change in traffic and no alert. `llmrouter` treats fallback rate as a
> first-class, alert-thresholded metric and caps escalation depth so a
> systemic failure can never route 100% of traffic to the most expensive
> model.

---

## Features

| Feature | Status |
|---|---|
| YAML model registry (split input/output pricing, no hardcoded prices) | ✅ Phase 1 |
| Rule-based routing (keywords, token count, explicit override) | ✅ Phase 1 |
| Embedding complexity classifier (local, ~5ms) | 🔜 Phase 2 |
| Optional LLM classifier (opt-in, flag-gated) | 🔜 Phase 2 |
| Capped, logged failure escalation | 🔜 Phase 3 |
| Output verification hooks (json-schema, non-empty) | 🔜 Phase 3 |
| Per-route metrics + alert threshold | 🔜 Phase 4 |
| Streamlit dashboard (tier split, fallback-rate cascade detector) | 🔜 Phase 4 |
| OpenAI-compatible proxy | 🔜 Phase 5 |
| Compose with `semcache` (cache in front) + `tracely` (spans) | 🔜 Phase 5 |

## Architecture

```
                        ┌───────────────────────────────────────────┐
   request  ─────────▶  │                  Router                    │
                        │                                            │
                        │   1. RuleClassifier      (sub-1ms)         │
                        │        └─ miss ─▶ 2. EmbeddingScorer (~5ms) │   [Phase 2]
                        │             └─ ambiguous ─▶ 3. LLMClassifier│   [opt-in]
                        │                                            │
                        │   tier ─▶ cheapest sufficient model        │
                        └───────────────────┬────────────────────────┘
                                            │
                          call model ◀──────┘
                                            │
                     ┌──────────────────────▼─────────────────────┐
                     │  on 5xx / timeout / failed verify:          │   [Phase 3]
                     │  FallbackPolicy escalates ONE tier,         │
                     │  capped at max_escalation_depth, LOGGED     │
                     └──────────────────────┬─────────────────────┘
                                            │
                     ┌──────────────────────▼─────────────────────┐
                     │  RouteMetrics: per-route cost / latency /   │   [Phase 4]
                     │  fallback rate  +  alert threshold          │
                     └─────────────────────────────────────────────┘
```

## Tech stack

| Component | Library | Role |
|---|---|---|
| Data models | pydantic v2 | registry, config, route decisions (validated, frozen) |
| Config | pyyaml | model registry + routing policy |
| Embeddings | sentence-transformers (bge-small) | local complexity classifier (Phase 2) |
| Vectors | numpy | similarity / centroid math (Phase 2) |
| Proxy | fastapi + uvicorn | OpenAI-compatible endpoint (Phase 5) |
| Models | langchain-google-genai | Gemini across tiers (Phase 3+) |
| Dashboard | streamlit | per-route metrics (Phase 4) |
| Tests | pytest | suite |

## Quick start

```bash
# 1. Virtual environment (mandatory)
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 2. Install (editable) — Phase 1 needs only the core group
pip install -e .

# 3. See routing decisions
python examples/demo_routing.py

# 4. Tests
pytest -q
```

`examples/demo_routing.py` prints, for a mix of prompts, the chosen tier,
model, deciding classifier stage, and reason:

```
QUERY                             TIER      MODEL                  VIA       REASON
Classify this customer review...  cheap     gemini-3.1-flash-lite  rules     cheap keyword matched: 'Classify'
Prove that there are infinitel... frontier  gemini-3.5-flash       rules     frontier keyword matched: 'Prove'
Tell me something interesting...  medium    gemini-3-flash         default   no rule matched; default tier -> medium
```

## Cost savings — honest framing

Savings depend entirely on your **workload mix**, not on a magic number.

- A **cheap-heavy** workload (lots of classify/extract/format) vs. sending
  everything to the frontier model can save **on the order of 70-85%**.
- A **reasoning-heavy** workload that genuinely needs the frontier model most
  of the time saves little — maybe **30%** — because the router correctly
  routes it up anyway.
- Measured combined numbers (router + `semcache` in front) land in Phase 5;
  this section will be updated with the real figures from a fixed workload.

The router only saves money when a cheaper model is *actually sufficient*.
It will not, and should not, save money by under-serving hard prompts.

## Model registry

Models live in [`config/models.yaml`](config/models.yaml). Each entry:

```yaml
- name: gemini-3.1-flash-lite
  tier: cheap                 # cheap | medium | frontier
  provider: google
  input_cost_per_1m: 0.10     # USD per 1M input tokens
  output_cost_per_1m: 0.40    # USD per 1M output tokens (split pricing)
  max_context: 1000000
  supports_tools: true
```

Prices are **never** hardcoded in logic — a rate change is a one-line YAML
edit. Routing policy (tier thresholds, keywords, exemplars) lives separately
in [`config/policy.yaml`](config/policy.yaml).

## Project structure

```
llmrouter/
├── llmrouter/
│   ├── __init__.py        # Router, RouterConfig, ModelRegistry, RouteDecision, Tier
│   ├── registry.py        # model registry + shared routing types
│   ├── config.py          # PolicyConfig, RouterConfig
│   ├── router.py          # the routing cascade
│   └── classifier/
│       └── rules.py       # rule-based classifier (Phase 1)
├── config/
│   ├── models.yaml        # model registry
│   └── policy.yaml        # tier thresholds, rules, exemplars
├── examples/demo_routing.py
├── tests/
├── INTERNAL_NOTES.md      # engineering log — read this to understand "why"
└── pyproject.toml
```

## Environment variables

| Var | Needed from | Purpose |
|---|---|---|
| `GOOGLE_API_KEY` | Phase 3 | Gemini API calls (Phase 1 makes none) |
| `LLMROUTER_MODELS_CONFIG` | optional | override registry path |
| `LLMROUTER_POLICY_CONFIG` | optional | override policy path |
| `TRACELY_OTLP_ENDPOINT` | Phase 5, optional | emit routing spans |

Copy `.env.example` → `.env`.

## Tests

```bash
pytest -q
```

## License

MIT.
