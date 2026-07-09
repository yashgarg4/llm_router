"""Complexity classifiers, run as a cascade by the router.

Stage order and latency budget:
  * rules     — deterministic keyword/token rules      (sub-1ms)
  * embedding — local similarity to tier exemplars      (~5ms)   [Phase 2]
  * llm       — optional model call, opt-in only         (50-100ms + RTT) [Phase 2]

Phase 1 ships the rule classifier only.
"""

from .rules import RuleClassifier

__all__ = ["RuleClassifier"]
