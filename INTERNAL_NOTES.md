# INTERNAL_NOTES — llmrouter

A living engineering log. Updated after every phase and every bug. If you
are reading this to understand the project, read section 1 then the latest
phase entry.

---

## 1. What llmrouter is

`llmrouter` is a cost- and quality-aware router that sits between an
application and multiple LLMs. For each request it:

1. **Classifies complexity** — rules first, then embeddings, then an
   *optional* LLM classifier. The default path never calls an LLM.
2. **Routes to the cheapest model** whose tier meets the complexity.
3. **On failure** (5xx / timeout / failed verification) escalates one tier
   up, capped at a max depth, logging every escalation.
4. **Records cost, latency, and fallback rate — per route.**

It is OpenAI-compatible and designed to compose with `semcache` (a cache in
front) and `tracely` (which receives routing spans).

### The problem it solves, and the lesson it exists to teach

Sending every request to a frontier model is simple and wasteful: most
traffic (classify / extract / format) is handled just as well by a model
that costs 5-20x less. A router captures that saving by matching each
request to the cheapest sufficient model.

But a cost router **without per-route observability is a liability.** The
real incident this project is built around:

> A cheap-tier verifier started failing after a provider changed its output
> formatting. The router, doing exactly what it was told, silently escalated
> ~90% of traffic to the frontier model. The bill tripled with no change in
> traffic and no alert. Nobody noticed until finance did.

This is a **fallback cascade**, and it explodes *silently* because each
individual escalation is correct behaviour — the system is "working." The
only signal is the aggregate: a fallback rate that jumped from ~5% to ~90%.

Therefore the design defends against it on three fronts:

- **Per-route metrics** (cost, latency, fallback rate) are a core component,
  not an afterthought bolted on later.
- **Fallback/escalation rate is a first-class metric with an alert
  threshold** (default 0.25). Crossing it is the cascade detector.
- **Escalation depth is capped** (default 2). A systemic failure cannot,
  even in principle, route 100% of traffic to the most expensive model —
  the cap is asserted in code.

---

## 2. Tech choices & rejected alternatives

| Choice | Why | Rejected alternative |
|---|---|---|
| **Rules → embeddings → (opt) LLM** cascade | Cheapest sufficient classifier wins. Most prompts are decided by a sub-1ms regex. | *LLM classifier by default* — adds a full model round-trip (50-100ms + network) and a cost to *every* request just to decide where to send it. That is the tail wagging the dog: you pay a model call to avoid a model call. |
| **Local embeddings** (sentence-transformers, bge-small family) | ~5ms, no network, no per-query cost, no rate limit. Same model family as semcache, so the two share a mental model. | *Hosted embedding API* — network latency + cost + a rate limit on the routing hot path. |
| **YAML model registry** | A price change or new model is a config edit, not a code change + redeploy. Split input/output pricing mirrors semcache. | *Prices hardcoded in logic* — the exact thing that turns a routine rate change into a code review. |
| **pydantic v2** for registry/config/decisions | Validation at load time (bad YAML fails loudly, not at 3am), frozen models prevent accidental mutation of a decision. | Hand-rolled dict parsing — silent typos, no validation. |
| **Capped escalation** | The cascade-explosion defence. Non-negotiable. | Uncapped retry-up — the failure mode this whole project exists to prevent. |

### Why not an LLM classifier by default (expanded)

The router's job is to *save* model calls. If deciding where to route costs
a model call, you have added latency and cost to the very path you are
optimising, and you have coupled your routing availability to a model
provider's uptime. Rules + local embeddings decide the overwhelming majority
of traffic for effectively free. The LLM classifier is a flag-gated
escape hatch for genuinely ambiguous prompts, off by default.

---

## 3. Concepts, explained simply

### The three routing strategies and their latency cost

| Strategy | How | Latency | When |
|---|---|---|---|
| **Rules** | Keyword / token-count / explicit override. Regex scan. | **sub-1ms** | Obvious prompts ("classify…", "prove…"). Runs first, short-circuits. |
| **Embedding** | Similarity of the query embedding to per-tier exemplar centroids. | **~5ms** | Prompts no rule caught. (Phase 2.) |
| **LLM classifier** | A single cheap-model call returning a structured complexity score. | **50-100ms + round-trip** | Opt-in only, for ambiguous prompts, when explicitly enabled. (Phase 2.) |

The order is deliberate: try the cheap deterministic thing first, escalate to
a more expensive classifier only when the cheaper one abstains.

### Why "router latency" is a misframing

People worry the router adds latency. Compare the numbers: a rule scan is
sub-1ms, an embedding is ~5ms. The model call that follows is **hundreds to
thousands of milliseconds**. The router is not on the critical path in any
meaningful sense — it is a rounding error next to the thing it is choosing.
And by choosing a cheaper/faster model it frequently makes the *total*
request faster. The router isn't the bottleneck; the model choice is. The
only routing strategy that meaningfully adds latency is the LLM classifier
(because it *is* a model call) — which is exactly why it is off by default.

### Complexity scoring: distance → 0..1 score (Phase 2 preview)

Each tier has a set of exemplar queries (in `policy.yaml`). At startup we
embed them and compute a centroid per tier. For an incoming query we embed
it and measure similarity to each centroid; the result is normalised into a
0..1 complexity score. `thresholds` in the policy map that score onto a tier
(`< 0.40` cheap, `< 0.80` medium, else frontier). Exemplar quality bounds
accuracy — garbage exemplars, garbage routing.

### Cost vs quality routing, and why per-route metrics matter

Cost routing pushes work down to the cheapest model; quality routing pushes
it up to whatever is *sufficient*. The tension is resolved per-tier: within
the chosen tier, pick the cheapest model that meets the caps. **Per-route
metrics** are what let you see whether that trade-off is actually holding —
or whether one route has quietly started escalating everything (see §1).

### Fallback cascades and how they explode silently

Each escalation is individually correct: a call failed, so we tried a
better model. The danger is *aggregate*. If a whole tier's verifier breaks,
every request in that tier escalates, and the only visible symptom is a
fallback rate that spiked. Without a per-route fallback-rate metric and an
alert threshold, the first sign is the invoice. The cap bounds the blast
radius; the metric + alert make it visible.

---

## 4. Architecture walkthrough

**Simple query — "Classify this review as positive or negative."**
→ `Router.route` calls `RuleClassifier.classify`
→ cheap keyword `classify` matches
→ decision: tier=cheap, model=`gemini-3.1-flash-lite` (cheapest in cheap),
  classifier_used=`rules`, sub-1ms. Done.

**Hard query — "Design a fault-tolerant message queue."**
→ `RuleClassifier.classify`
→ frontier keyword `design` matches (frontier beats cheap on conflict)
→ decision: tier=frontier, model=`gemini-3.5-flash`, via=`rules`.

**Ambiguous query — "Tell me something interesting about the ocean."**
→ no keyword, normal size → `RuleClassifier` returns `None`
→ Phase 1: falls through to the default tier (medium).
→ Phase 2: the embedding scorer will assign a score and tier here.

**A query that fails and escalates (Phase 3 preview)**
→ routed to cheap → cheap model returns 5xx (or output fails verification)
→ `FallbackPolicy.on_failure` escalates one tier (cheap → medium), logs the
  reason, increments the tier's fallback count
→ retry on medium; if that also fails, escalate to frontier
→ at the cap (or at frontier), stop and return the error — never loop.

---

## 5. Phase log

### Phase 1 — Registry + rule-based routing ✅

**What was built**
- Project scaffold: venv, `.gitignore`, pinned `requirements.txt`,
  `.env.example`, `pyproject.toml` (editable install), full folder tree.
- `config/models.yaml` — 4-model Gemini registry across 3 tiers with split
  input/output pricing and context/tool caps.
- `config/policy.yaml` — tier thresholds, rule keywords, token thresholds,
  per-tier exemplars (for Phase 2), default tier.
- `llmrouter/registry.py` — `Tier` enum (with `rank`/`next_up` for future
  escalation), `ModelSpec`, `RouteDecision`, `ModelRegistry` with `load`,
  `get`, `models_by_tier`, `cheapest_in_tier`, `cheapest_in_tier_with_context`.
- `llmrouter/config.py` — `PolicyConfig` (+ nested `Thresholds`, `RuleConfig`,
  `KeywordRules`) and `RouterConfig` (feature flags with safe defaults so
  later phases extend, not rewrite).
- `llmrouter/classifier/rules.py` — `RuleClassifier`: explicit override →
  large-context → keyword rules, returns `None` to defer.
- `llmrouter/router.py` — `Router.route` (rules → default), `from_config`.
- `examples/demo_routing.py`, `tests/test_rules.py`, `tests/test_router.py`.

**Completion signal** — `demo_routing.py` prints a clean table: cheap prompts
(classify/extract/translate) → cheap tier, frontier prompts (prove/debug) →
frontier tier, large-context → context-capable model, ambiguous → default.
`pytest` 21/21 green.

**Key decisions**
- Put `Tier` / `RouteDecision` in `registry.py` (not a separate `types.py`):
  a decision *is* a model selection, so co-locating keeps the import graph
  acyclic (classifier and router both import from registry, nothing cycles).
- **Frontier beats cheap on keyword conflict.** Under-routing a hard prompt
  (wrong answer from a weak model) is the expensive mistake; over-routing
  costs a few cents. Bias toward capability on ambiguity within rules.
- **Large context ≠ complex.** A 350k-token prompt might just be "summarise
  this book." The large-context rule picks the *cheapest fitting* model
  across tiers, not the frontier tier.
- Word-boundary regex with `\w*` suffix so "classify" matches "classifies"
  but a random word doesn't false-match a keyword prefix.
- Frozen pydantic models for `ModelSpec` / `RouteDecision` — a decision is a
  record of what happened; it must not be mutable after the fact.

**Bugs**
- *Editable install / import path.* First `demo_routing.py` run failed with
  `ModuleNotFoundError: No module named 'llmrouter'`. Cause: the package
  wasn't installed into the venv. Tried: running the script directly (fails,
  cwd isn't on `sys.path` for a package import from `examples/`). Fixed:
  `pip install -e .` (the brief's editable-install requirement). Taught: the
  editable install is not optional polish — it is what makes `from llmrouter
  import ...` work from `examples/` and `tests/` uniformly. `pyproject.toml`
  also sets `pythonpath = ["."]` for pytest as a belt-and-braces path.
- *`.gitkeep` before `mkdir`.* Tried to write `server/.gitkeep` before the
  dir existed → "No such file or directory". Fixed by `mkdir -p` first.
  Trivial, but logged per the "document every bug" rule.

**Concepts reinforced** — see §3. Phase 1 is the rules row of the strategy
table and the "router latency is a misframing" point (a regex scan is free).

**Interview Q&A**
- *Q: Why not just always use the frontier model?* A: Most traffic is
  low-complexity; a frontier model is 5-20x the cost for no quality gain on
  a "classify this" prompt. Routing captures that saving.
- *Q: Why rules before embeddings?* A: Cheapest sufficient classifier wins.
  A regex is sub-1ms and deterministic; only pay for an embedding when the
  regex can't decide.
- *Q: A prompt matches both a cheap and a frontier keyword — what wins, and
  why?* A: Frontier. Under-routing yields a wrong answer (expensive to the
  business); over-routing costs pennies. Asymmetric risk → bias to
  capability.
- *Q: Big prompt — straight to frontier?* A: No. Size ≠ complexity. Pick the
  cheapest model whose context window fits.
- *Q: Where do prices live and why does it matter?* A: In `models.yaml`, not
  in code. A rate change should be a config edit, not a deploy.
