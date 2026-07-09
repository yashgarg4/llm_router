"""Phase 1 demo: route a mix of prompts and print the decision table.

Run:
    python examples/demo_routing.py

Shows that deterministic rules cleanly separate cheap-tier tasks (classify,
extract, translate) from frontier-tier tasks (reason, debug, design), route a
huge prompt to a context-capable model, and fall back to the default tier for
genuinely ambiguous prompts (which Phase 2's embedding scorer will handle).
"""

from __future__ import annotations

from llmrouter import Router, RouterConfig

# (query, metadata) pairs spanning the task spectrum.
PROMPTS: list[tuple[str, dict]] = [
    ("Classify this customer review as positive, negative, or neutral.", {}),
    ("Extract every phone number from the following text.", {}),
    ("Summarise this quarterly report for a busy executive.", {}),
    ("Prove that there are infinitely many prime numbers.", {}),
    ("Debug why this goroutine deadlocks under load.", {}),
    ("Translate 'good morning, how are you?' into Japanese.", {}),
    # Large-context prompt: metadata carries a token_count over the threshold.
    ("Review this entire codebase and note any issues.", {"token_count": 350_000}),
    # Ambiguous: no keyword, normal size -> default tier (Phase 2 scores it).
    ("Tell me something interesting about the ocean.", {}),
]


def main() -> None:
    router = Router.from_config(RouterConfig())

    header = f"{'QUERY':<52} {'TIER':<9} {'MODEL':<22} {'VIA':<9} REASON"
    print(header)
    print("-" * len(header))
    for query, metadata in PROMPTS:
        d = router.route(query, metadata)
        q = (query[:49] + "...") if len(query) > 52 else query
        print(
            f"{q:<52} {d.tier.value:<9} {d.model.name:<22} "
            f"{d.classifier_used:<9} {d.reason}"
        )


if __name__ == "__main__":
    main()
