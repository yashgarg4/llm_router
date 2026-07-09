"""Routing demo: route a mix of prompts and print the decision table.

Run:
    python examples/demo_routing.py

Shows the full cascade in action:
  * deterministic RULES cleanly separate obvious cheap/frontier tasks and
    force a context-capable model on huge prompts, and
  * the EMBEDDING scorer tiers the prompts no rule caught, tagging each with a
    0..1 complexity score.

The VIA column reports which classifier stage decided (rules / embedding /
default). The optional LLM classifier is off by default; enable it with
RouterConfig(use_llm_classifier=True) to see it fire on ambiguous prompts.
"""

from __future__ import annotations

from llmrouter import Router, RouterConfig

# (query, metadata). The second group has NO rule keywords, so the embedding
# scorer decides them — spanning simple → moderate → hard on purpose.
PROMPTS: list[tuple[str, dict]] = [
    # --- rule hits ---
    ("Classify this customer review as positive, negative, or neutral.", {}),
    ("Extract every phone number from the following text.", {}),
    ("Prove that there are infinitely many prime numbers.", {}),
    ("Debug why this goroutine deadlocks under load.", {}),
    ("Translate 'good morning, how are you?' into Japanese.", {}),
    ("Review this entire codebase and note any issues.", {"token_count": 350_000}),
    # --- rule misses -> embedding scorer ---
    ("What is the boiling point of water at sea level?", {}),
    ("Give me a three-sentence overview of this news article.", {}),
    ("Walk me through the trade-offs of eventual vs strong consistency.", {}),
    ("Tell me something interesting about the ocean.", {}),
]


def main() -> None:
    router = Router.from_config(RouterConfig())

    header = (
        f"{'QUERY':<50} {'TIER':<9} {'MODEL':<22} {'VIA':<9} {'SCORE':<6} REASON"
    )
    print(header)
    print("-" * len(header))
    for query, metadata in PROMPTS:
        d = router.route(query, metadata)
        q = (query[:47] + "...") if len(query) > 50 else query
        score = f"{d.score:.2f}" if d.score is not None else "-"
        print(
            f"{q:<50} {d.tier.value:<9} {d.model.name:<22} "
            f"{d.classifier_used:<9} {score:<6} {d.reason}"
        )


if __name__ == "__main__":
    main()
