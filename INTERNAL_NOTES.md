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

---

### Phase 2 — Complexity classifier ✅

**What was built**
- `classifier/embedding.py` — `EmbeddingClassifier`: embeds per-tier exemplars
  once at startup (local bge-small), builds a centroid per tier, and scores a
  query by projecting its embedding onto the cheap→frontier complexity axis.
  Returns `EmbeddingScore(score, tier, reason, margin, similarities)`.
- `classifier/llm.py` — `LLMClassifier`: OPTIONAL, flag-gated. One cheap-model
  call returning a structured `ComplexityAssessment` (0..1 score + rationale).
  Lazy-imports `langchain-google-genai` so the default install/path never
  needs it. Documented as the only networked stage.
- `router.py` — full cascade `rules → embedding → (enabled + ambiguous) llm`,
  with lazy construction of the embedding/LLM stages and graceful degradation
  to the default tier if the embedding model can't load.
- `config.py` — `RouterConfig` gains `embedding_model`, `embedding_temperature`,
  `ambiguity_margin`, `use_llm_classifier`, `llm_classifier_model`.
- Extended `demo_routing.py` (adds a SCORE column and rule-miss prompts);
  `tests/test_classifier.py` (real-model scoring + cascade + flag-gated LLM
  via an injected fake).

**Completion signal** — demo shows rules deciding the 6 obvious prompts and
the embedding scorer tiering the 4 rule-misses with 0..1 scores, each tagged
with the deciding stage. `pytest` 36/36 green (bge-small downloads on first
run, then cached).

**Key decisions**
- **Scoring = 1-D projection onto the cheap→frontier axis, anchored piecewise**
  (cheap centroid→0.0, medium→0.5, frontier→1.0). First attempt used a
  softmax over the three centroid similarities weighting tier *anchors* — see
  the bug below for why that failed. The projection is a genuine monotone
  gradient: capital-of-France 0.00 → JSON 0.17 → overview 0.43 → hash-map
  0.65 → consistency 0.80 → halting 0.92 → rate-limiter 0.99.
- **Kept the brief's 0.40 / 0.80 thresholds** by anchoring the score to the
  medium centroid at 0.5, rather than retuning thresholds to a raw axis where
  the medium centroid happened to sit at 0.39.
- **LLM classifier trigger = ambiguity, not "always."** It fires only when
  `use_llm_classifier` is on AND the top-two centroid similarities are within
  `ambiguity_margin`. Pay for a model opinion only on genuine tier-boundary
  cases. Off by default; lazy provider import.
- **Graceful degradation.** If the embedding model can't load (offline / no
  cache) the router logs and falls back to the default tier instead of
  failing the route. Verified by a test that forces `_embedding_failed`.
- **Real model in tests, not mocks.** The classifier's value *is* its
  embeddings; mocking them would test nothing. Module-scoped fixture loads
  bge-small once.

**Bugs**
- *Softmax-anchor averaging collapsed everything to medium.* First scorer
  computed `score = Σ softmax(sim_i)·anchor_i` over the three tier centroids.
  Symptom: "boiling point of water" (nearest centroid cheap) scored 0.46 →
  medium, and a distributed-systems reasoning question (nearest centroid
  frontier, sim 0.75) got dragged down to 0.67 → medium. Root cause: raw
  bge cosine sims sit in a narrow band and a query is often similar to *both*
  the cheap and frontier exemplars; averaging anchors 0.0 and 1.0 lands at
  ≈0.5 regardless of true complexity (bimodal-average pathology). Fix:
  replaced with the 1-D axis projection, which cannot exhibit this because it
  reads one signed position, not a blend of two extremes. Taught: for an
  ordinal target (a complexity *ladder*), project onto the ordinal axis;
  don't average categorical anchors weighted by similarity.
- *Topic gravity leaks into the complexity score.* "What year did WWII end?"
  (a trivial date lookup) scored 0.53 → medium because "world war" embeds near
  weighty/analytical exemplars. This is not a code bug — it is the
  exemplar-quality bound: an embedding measures topical + stylistic
  similarity, not reasoning depth. Documented as a known limitation; the fix
  space is better/more exemplars or the opt-in LLM classifier for such cases.
  Adjusted the test to an unambiguous cheap query.

**Concepts reinforced** — the embedding row of the strategy table (~5ms,
local, no network); "complexity scoring: how an embedding distance becomes a
0..1 score" (§3) is now literally the projection method; and why the LLM
classifier stays opt-in (it is the only stage that adds a round-trip).

**Known limitations (carried forward)**
- The classifier is heuristic, not learned; exemplar quality bounds accuracy.
- A single cheap→frontier axis assumes medium lies roughly between the two;
  topically neutral "medium" queries can project low (e.g. "three-sentence
  overview" → 0.38, just under the cheap boundary). Acceptable for a
  heuristic; a learned complexity model is the real fix (Phase 5 "what I'd do
  differently").

**Interview Q&A**
- *Q: Why project onto an axis instead of nearest-centroid or a softmax
  blend?* A: Complexity is ordinal (cheap < medium < frontier). A projection
  onto the cheap→frontier direction respects that ordering and yields a smooth
  gradient. Nearest-centroid gives no gradient; a softmax-weighted average of
  the extreme anchors collapses "similar to both ends" queries to the middle.
- *Q: When does the LLM classifier run, and why not always?* A: Only when
  enabled and the embedding decision is ambiguous (tiny centroid margin).
  Always-on would add a model round-trip and cost to every request just to
  decide routing — the tail wagging the dog, and it couples routing
  availability to a provider.
- *Q: The embedding model is down — what happens?* A: The router logs it and
  routes rule-misses to the default tier. A classifier outage degrades
  routing quality; it never fails the request.
- *Q: Biggest weakness of this classifier?* A: It measures semantic/topical
  similarity, not reasoning depth, so topic can leak into the score
  (WWII-date example). Exemplar quality is the ceiling; a learned model
  trained on measured outcomes would be the upgrade.

---

### Phase 3 — Fallback + verification ✅

**What was built**
- `client.py` — `ModelResponse`, a `ModelClient` protocol, `ModelCallError`
  (with a `transient` flag), and a lazy `GeminiClient`. Separating "decide the
  model" from "call the model" is what lets the escalation logic be tested
  offline with a scripted fake — no key, no network.
- `verify.py` — `verify(response, spec)` with `non_empty`, `json`, and a
  lightweight `json_schema` (required keys + types, no extra dependency). A
  failed check escalates via the same path as a 5xx.
- `fallback.py` — `FallbackPolicy.on_failure(decision, error, depth)`:
  escalates exactly one tier, returns None at the frontier or at the cap. The
  cap is **asserted** in `__init__` (non-negative int).
- `router.py` — `acomplete(query, metadata, verify_spec)`: the route → call →
  (on transient error / failed verify) escalate → retry loop, returning
  `RouteResult(success, response, final_decision, escalations, total_cost,
  latency_ms, attempts, error)`. Costs summed from registry prices.
- `examples/demo_fallback.py`, `tests/test_fallback.py`.

**Completion signal** — `demo_fallback.py` shows: (1) a cheap-tier 5xx
escalating exactly one tier to success, (2) a failed verification escalating
via the same path and logged distinctly as the cascade vector, (3) a systemic
failure escalating to the cap (cheap→medium→frontier) and then stopping —
exactly three models called, no infinite loop. `pytest` 52/52 green.

**Key decisions**
- **Two independent bounds on escalation.** The tier ladder is finite (three
  tiers → at most two escalations) *and* there is an explicit depth cap. Belt
  and braces: this is the one behaviour the whole project exists to guarantee,
  so it does not rest on a single mechanism. The cap is asserted, not merely
  configured.
- **Transient vs non-transient errors.** 5xx / 408 / 429 / timeouts escalate;
  4xx re-raises without escalating. Escalating a malformed request to a
  pricier model just burns budget for the same failure. `ModelCallError`
  infers transience from the status code when not stated.
- **Verification failures escalate via the *same* fallback path as 5xx**, and
  are logged at WARNING with a loud `VERIFICATION-TRIGGERED ESCALATION`
  marker. This is deliberate: in the origin incident (§1) it was a *verifier*,
  not an outage, that drove the cascade. The metric that catches it (fallback
  rate per route) lands in Phase 4; the distinct log line makes it greppable
  now.
- **Cost is counted whenever a response comes back** — including a response
  that then fails verification (the provider still billed for those tokens).
  Failed transient calls raise before returning usage, so they add nothing.
- **Async with no test dependency.** `acomplete` is async (real providers are
  I/O-bound); tests drive it with `asyncio.run` rather than adding
  pytest-asyncio.

**Bugs**
- *Console encoding on the demo.* The section title used an em-dash (`—`),
  which the Windows console (cp1252) rendered as `�`. Not a logic bug, but it
  makes the demo output look broken. Fixed by using an ASCII hyphen in
  printed strings. Taught: keep demo stdout ASCII-safe on Windows, or set
  `PYTHONIOENCODING=utf-8`. (Source files stay UTF-8; only printed text was
  changed.)

**Concepts reinforced** — "fallback cascades and how they explode silently"
(§1) is now executable: demo case 3 is precisely the uncapped scenario, and
the cap is what turns "100% of traffic to frontier" into "at most two
escalations then stop." The verification-triggered escalation is the specific
silent vector, now logged.

**Interview Q&A**
- *Q: How do you stop a fallback cascade from routing everything to the most
  expensive model?* A: Two bounds — a finite tier ladder and an asserted depth
  cap — plus a per-route fallback-rate metric with an alert threshold
  (Phase 4). The cap bounds the blast radius; the metric makes a spike
  visible before the invoice does.
- *Q: A 400 comes back from the cheap model — do you escalate?* A: No. 4xx
  means the request is bad; a stronger model fails the same way at higher
  cost. Only 5xx / timeouts / failed verification escalate.
- *Q: Why route verification failures through the same escalation path as
  outages?* A: Because to the business they are the same event — "this tier
  didn't produce a usable answer, try a better one" — and treating them
  identically means one fallback-rate metric catches both. The origin
  incident was a verifier failure, so this path is the one that must be
  observable.
- *Q: Where do the costs in RouteResult come from?* A: `ModelSpec.cost_for`
  using split input/output prices from `models.yaml`. Summed across every
  attempt that returned tokens, so an escalated request honestly reports the
  cost of *all* the models it touched, not just the final one.

**Known limitations (carried forward)**
- Real Gemini calls need `pip install langchain-google-genai` + a
  `GOOGLE_API_KEY`; Phase 3 is validated with a scripted fake client (the
  escalation logic is provider-agnostic). Real end-to-end calls are exercised
  in Phase 5.
- Failed-transient calls contribute no cost because the fake/real clients
  raise before returning usage; a provider that bills failed calls would
  under-report. Acceptable until real usage metadata is wired in Phase 5.

---

### Phase 4 — Per-route metrics + dashboard ✅

**What was built**
- `metrics.py` — `RouteMetrics`: `record` / `record_result`, `by_route`,
  `by_tier`, `fallback_rate(tier)`, `savings_vs_baseline`, `alert_check`,
  `fallback_rate_series` (rolling, for the line chart), `snapshot`, JSON
  persistence (`save_metrics`/`load_metrics`), and `simulate_metrics` which
  reproduces the incident (healthy period → cheap-tier failure spike).
- `server/dashboard.py` — FastAPI metrics REST: `/metrics`, `/by-route`,
  `/by-tier`, `/alerts`, `/timeseries`, `/health`.
- `dashboard/app.py` — Streamlit dashboard: KPI row (total cost, saved
  vs all-frontier, overall fallback rate vs threshold, request volume), tier
  distribution, fallback-rate-over-time line with the threshold drawn on it,
  and a per-route table that highlights routes over the threshold.
- `tests/test_metrics.py`; Makefile `dashboard` + `proxy-metrics` targets.

**Completion signal** — the simulated workload (320 requests) shows a tier
split (cheap 67%, medium 22%, frontier 11%), 54% cost saved vs all-frontier,
and a cheap-tier fallback rate of 36% that trips the alert (both tier and
route scope). The rolling fallback line runs 0.00 → 0.47 across the sequence,
visibly crossing the 0.25 threshold during the spike — exactly the panel that
would have caught the origin incident. `pytest` 62/62 green; the Streamlit app
boots headless and serves HTTP 200; the REST handlers return correct payloads.

**Key decisions**
- **"route" = the entry point, not the final model.** A request is attributed
  to the tier/model it was *initially* routed to (the first escalation's
  `from_*`, or the final decision if it never escalated). This is the unit
  whose health matters: a broken cheap-tier verifier shows up as the *cheap
  route's* fallback rate spiking, which is precisely the incident. Attributing
  to the final model would smear the signal across whatever tiers absorbed the
  overflow and hide the cause.
- **Cost is attributed to the entry route, summed across all attempts.** So an
  escalating route reports its true inflated cost — the cheap route in the sim
  shows a per-request avg cost dragged up by all the medium/frontier retries.
- **`alert_check` flags both tiers and routes.** The brief specifies tiers;
  routes are included because in a multi-model tier the offending *model* is
  the actionable unit. Both fire in the sim (they coincide here since each
  tier has one entry model).
- **`savings_vs_baseline` needs per-request tokens**, so `record` stores input/
  output token counts (an addition beyond the brief's listed signature — the
  counterfactual "what would all-frontier have cost" is un-computable without
  them). Prices for the baseline come from the registry, never hardcoded.
- **A JSON store decouples producer from viewer.** The proxy (Phase 5) writes
  `metrics_store.json`; the dashboard and REST read it. The dashboard's
  simulation writes the same file so the REST API mirrors what's on screen.
- **Simulation is ordered healthy-then-spike** so the rolling line *rises*
  across time — a flat average would hide the very dynamic the panel exists to
  show.

**Bugs**
- *FastAPI TestClient needs httpx, which isn't installed.* Rather than add a
  dependency just to smoke-test the API, I called the route handlers directly
  (they are plain functions returning dicts) and verified payloads that way.
  Taught: FastAPI handlers are ordinary callables — you don't need an HTTP
  client (or a running server) to unit-test their logic; reserve httpx/TestClient
  for testing actual HTTP behaviour (status codes, serialisation, middleware).
- *Runtime artifact leaked into the working tree.* The REST smoke test created
  `metrics_store.json`; it's gitignored, but I removed it to keep the tree
  clean. Noted that the store path is relative to CWD — the proxy should write
  it to a configured location in Phase 5, not wherever it happens to run.

**Concepts reinforced** — "cost vs quality routing, and why per-route metrics
matter" (§3) is now the whole module: the dashboard makes the abstract "watch
your fallback rate" concrete. The alert threshold line on the timeseries *is*
the defence the origin incident lacked.

**Interview Q&A**
- *Q: What one panel would have caught the 90%-escalation incident?* A: The
  per-route fallback-rate — either the rolling line crossing the alert
  threshold, or the per-route table row for the cheap entry model turning red.
  Traffic volume was unchanged, so only a *ratio* per route reveals it.
- *Q: Why attribute cost/fallback to the entry route rather than the model
  that served the request?* A: The entry route is the cause; the serving model
  is the symptom. Attributing to the entry point localises the failure to the
  tier whose verifier/health broke. Attributing to the final model would show
  "frontier volume went up" with no hint why.
- *Q: How do you compute savings honestly?* A: Counterfactual — price every
  request's actual tokens against the baseline (all-frontier) model from the
  registry, and compare to actual summed cost. It depends on workload mix and
  is reported as such (54% in the sim), never as a fixed marketing number.
- *Q: Why a rolling window for the fallback timeseries instead of a cumulative
  average?* A: A cumulative average is dominated by history and lags a spike;
  a rolling window responds quickly, so the line crosses the threshold when
  the incident starts, not long after.
