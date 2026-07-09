"""Phase 5 demo: semcache in front of llmrouter, with real Gemini calls.

    cache hit   -> skip routing entirely (no model call, ~$0)
    cache miss  -> route to the cheapest sufficient model, call it, cache it

Run:
    python examples/demo_with_semcache.py

Env:
    GOOGLE_API_KEY      required (real Gemini calls)
    SEMCACHE_PATH       path to the semantic_cache checkout (default below)
    LLMROUTER_THROTTLE  seconds to wait between real model calls (default 6)
                        — the Gemini free tier is ~5-15 RPM, so we pace misses.

It reports three costs on the same 20-request workload:
    1. all-frontier, no cache   (naive baseline)
    2. router, no cache         (routing alone)
    3. router + semcache        (routing + cache)
so you can see what each layer contributes. Numbers depend on the workload
mix and its duplication rate — this workload is intentionally repetitive
(paraphrases + repeats) to exercise the cache, as real traffic is.
"""

from __future__ import annotations

import asyncio
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

SEMCACHE_PATH = os.getenv(
    "SEMCACHE_PATH", r"C:\Users\Yash2.Garg\self_projects\semantic_cache"
)
THROTTLE = float(os.getenv("LLMROUTER_THROTTLE", "6"))

if Path(SEMCACHE_PATH).exists():
    sys.path.insert(0, SEMCACHE_PATH)
try:
    from semcache import CacheConfig, SemCache
except ImportError:
    print(
        f"semcache not importable from {SEMCACHE_PATH!r}.\n"
        "Set SEMCACHE_PATH to your semantic_cache checkout and ensure faiss-cpu "
        "is installed in this venv."
    )
    raise SystemExit(1)

from llmrouter import Router, RouterConfig, Tier  # noqa: E402

# A repetitive workload: some prompts recur verbatim, some are paraphrases —
# exactly the shape real traffic has, and what a semantic cache exploits.
WORKLOAD = [
    "Classify the sentiment of: 'I absolutely love this!' One word.",
    "Extract the year from: 'The treaty was signed in 1648.'",
    "Translate 'good morning' into Spanish.",
    "Classify the sentiment of: 'I absolutely love this!' One word.",   # repeat
    "Summarise in one sentence: the water cycle moves water through evaporation, condensation, and precipitation.",
    "What is the sentiment of 'I really love this!'? Answer in one word.",  # paraphrase of #1
    "Extract the year from: 'The treaty was signed in 1648.'",           # repeat
    "Translate 'good morning' to Spanish.",                              # paraphrase of #3
    "Classify: is 'the food was cold and late' positive or negative? One word.",
    "Give a one-sentence summary of the water cycle.",                   # paraphrase of #5
    "What is 12 multiplied by 8?",
    "Classify the sentiment of: 'I absolutely love this!' One word.",   # repeat
    "Extract the email from: 'contact us at hi@example.com please'.",
    "Translate 'good morning' into Spanish.",                            # repeat
    "What is 12 times 8?",                                               # paraphrase
    "Summarise the water cycle in a single sentence.",                   # paraphrase
    "Classify sentiment: 'worst purchase ever'. One word.",
    "Extract the year from: 'The treaty was signed in 1648.'",          # repeat
    "Translate 'good morning' into Spanish.",                            # repeat
    "What is the sentiment of 'I really love this!'? One word.",         # paraphrase
]


async def main() -> None:
    if not os.getenv("GOOGLE_API_KEY"):
        print("GOOGLE_API_KEY not set — cannot make real calls.")
        raise SystemExit(1)

    router = Router.from_config(RouterConfig())
    cache = SemCache(config=CacheConfig(threshold=0.90))
    frontier = router.registry.cheapest_in_tier(Tier.FRONTIER)

    # Bookkeeping. tokens[query] remembers the (in, out) of the first real call
    # so cache hits still contribute to the baseline counterfactual.
    tokens: dict[str, tuple[int, int]] = {}
    router_cost = 0.0     # router + cache (actual spend)
    router_only_cost = 0.0  # router, pretending no cache (every request calls)
    baseline_cost = 0.0   # all-frontier, no cache
    hits = 0
    calls = 0

    print(f"semcache @ {SEMCACHE_PATH}")
    print(f"throttle: {THROTTLE}s between real calls (free-tier rate limit)\n")
    print(f"{'#':>2}  {'HIT/MISS':<9} {'TIER':<9} {'MODEL':<22} {'COST':>10}  QUERY")
    print("-" * 100)

    for i, query in enumerate(WORKLOAD, 1):
        cached = cache.get(query)
        if cached.is_hit:
            hits += 1
            in_tok, out_tok = tokens.get(query, (cached.tokens or 0, 0))
            # Attribute the cached prompt to whatever tier it would route to.
            decision = router.route(query)
            baseline_cost += frontier.cost_for(in_tok, out_tok)
            router_only_cost += decision.model.cost_for(in_tok, out_tok)
            # cache serves it: no model spend now.
            print(f"{i:>2}  {cached.hit_type:<9} {'-':<9} {'(cache)':<22} "
                  f"{'$0.000000':>10}  {query[:44]}")
            continue

        # Miss: route + call the real model, then cache the response.
        calls += 1
        result = await router.acomplete(query)
        d = result.final_decision
        if result.success and result.response:
            in_tok = result.response.input_tokens
            out_tok = result.response.output_tokens
            tokens[query] = (in_tok, out_tok)
            cache.put(query, result.response.text, tokens=in_tok + out_tok,
                      cost=result.total_cost, model=d.model.name)
            router_cost += result.total_cost
            router_only_cost += result.total_cost
            baseline_cost += frontier.cost_for(in_tok, out_tok)
            print(f"{i:>2}  {'MISS':<9} {d.tier.value:<9} {d.model.name:<22} "
                  f"${result.total_cost:>9.6f}  {query[:44]}")
        else:
            print(f"{i:>2}  {'MISS(err)':<9} {d.tier.value:<9} {d.model.name:<22} "
                  f"{'-':>10}  {result.error}")

        if i < len(WORKLOAD):
            time.sleep(THROTTLE)  # pace real calls under the free-tier limit

    # -- report --------------------------------------------------------------
    n = len(WORKLOAD)
    def pct(saved, base): return (saved / base * 100) if base else 0.0

    print("\n" + "=" * 60)
    print(f"Workload: {n} requests | cache hits: {hits} "
          f"({hits / n * 100:.0f}%) | real model calls: {calls}")
    print("-" * 60)
    print(f"  1. all-frontier, no cache : ${baseline_cost:.6f}   (baseline)")
    print(f"  2. router, no cache       : ${router_only_cost:.6f}   "
          f"({pct(baseline_cost - router_only_cost, baseline_cost):.0f}% vs baseline)")
    print(f"  3. router + semcache      : ${router_cost:.6f}   "
          f"({pct(baseline_cost - router_cost, baseline_cost):.0f}% vs baseline)")
    print("-" * 60)
    print(f"  routing alone saved  : {pct(baseline_cost - router_only_cost, baseline_cost):.0f}%")
    print(f"  cache added on top   : {pct(router_only_cost - router_cost, baseline_cost):.0f}% "
          f"(of baseline)")
    print(f"  combined saving      : {pct(baseline_cost - router_cost, baseline_cost):.0f}%")
    print("=" * 60)
    print("Note: savings scale with workload mix + duplication; a cheap-heavy, "
          "repetitive workload saves most.")


if __name__ == "__main__":
    asyncio.run(main())
