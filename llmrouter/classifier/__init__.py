"""Complexity classifiers, run as a cascade by the router.

Stage order and latency budget:
  * rules     — deterministic keyword/token rules      (sub-1ms)
  * embedding — local similarity to tier exemplars      (~5ms)   [Phase 2]
  * llm       — optional model call, opt-in only         (50-100ms + RTT) [Phase 2]

The LLM classifier is imported lazily (it pulls langchain-google-genai only
when actually constructed) — import it from ``llmrouter.classifier.llm``.
"""

from .embedding import EmbeddingClassifier, EmbeddingScore
from .rules import RuleClassifier

__all__ = ["RuleClassifier", "EmbeddingClassifier", "EmbeddingScore"]
