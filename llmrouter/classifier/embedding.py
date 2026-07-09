"""Embedding complexity classifier — the second cascade stage.

When no rule fires, we still need a tier. Rather than pay for an LLM call to
decide, we compare the query's embedding to a *centroid* per tier, built once
at startup from the exemplar queries in ``policy.yaml``. Closer to the
frontier exemplars → higher complexity score.

Cost profile: one local embedding (~5ms after warmup), no network, no
per-query price, no rate limit. See INTERNAL_NOTES §3 for why this belongs
before any LLM classifier.

### How a distance becomes a 0..1 score

The three tier centroids define a *complexity axis* in embedding space:

    axis = (frontier_centroid − cheap_centroid)

A query's complexity is where its embedding projects onto that axis. We then
anchor the projection piecewise so each tier centroid sits at its natural
point on the 0..1 scale:

    cheap centroid → 0.0     medium centroid → 0.5     frontier centroid → 1.0

Projections below the medium anchor map into [0.0, 0.5]; above it into
[0.5, 1.0]. `Thresholds.tier_for_score` then maps the score to a tier using
the policy's cut points (default 0.40 / 0.80).

Why a 1-D projection and not a softmax over per-centroid similarities: raw
sentence-embedding cosine similarities sit in a narrow band and a query is
often similar to *both* the cheap and frontier exemplars at once. Averaging
tier anchors weighted by those similarities collapses such queries to the
middle (≈0.5) regardless of their true complexity. Projecting onto the single
cheap→frontier direction gives a genuine monotone gradient instead.

### The margin (confidence)

Alongside the score we keep the gap between the top two centroid cosine
similarities. A small gap means the query sits between two tiers — low
confidence — which is the only situation where the optional LLM classifier is
worth running.
"""

from __future__ import annotations

import logging

import numpy as np
from pydantic import BaseModel, ConfigDict

from ..config import Thresholds
from ..registry import Tier

logger = logging.getLogger("llmrouter.classifier.embedding")


class EmbeddingScore(BaseModel):
    """Result of scoring one query. Carries the brief's (score, tier, reason)
    plus the confidence margin and raw per-tier similarities."""

    model_config = ConfigDict(frozen=True)

    score: float
    tier: Tier
    reason: str
    margin: float  # top1 - top2 centroid similarity; low = ambiguous
    similarities: dict[Tier, float]


class EmbeddingClassifier:
    """Scores query complexity by projection onto the cheap→frontier axis."""

    def __init__(
        self,
        exemplars: dict[Tier, list[str]],
        thresholds: Thresholds,
        model_name: str = "BAAI/bge-small-en-v1.5",
        temperature: float = 0.10,  # accepted for API stability; see note below
        model: object | None = None,
    ) -> None:
        # The complexity axis needs both endpoints. Medium is optional (its
        # centroid, if present, calibrates the 0.5 midpoint).
        if not exemplars.get(Tier.CHEAP) or not exemplars.get(Tier.FRONTIER):
            raise ValueError(
                "embedding classifier requires exemplars for at least the "
                "'cheap' and 'frontier' tiers to define the complexity axis"
            )
        self._thresholds = thresholds
        # temperature is retained in the signature (config passes it) but the
        # projection scorer does not need it; kept for forward compatibility.
        self._temperature = temperature

        self._model = model if model is not None else self._load_model(model_name)

        self._tiers = [t for t in (Tier.CHEAP, Tier.MEDIUM, Tier.FRONTIER)
                       if exemplars.get(t)]
        centroids = {}
        for t in self._tiers:
            vecs = self._embed(exemplars[t])
            mean = vecs.mean(axis=0)
            mean /= (np.linalg.norm(mean) + 1e-12)
            centroids[t] = mean.astype(np.float32)
        self._centroids = centroids
        self._centroid_matrix = np.vstack([centroids[t] for t in self._tiers])

        # Complexity axis and the per-tier anchor projections onto it.
        axis = centroids[Tier.FRONTIER] - centroids[Tier.CHEAP]
        self._axis = (axis / (np.linalg.norm(axis) + 1e-12)).astype(np.float32)
        self._c0 = float(centroids[Tier.CHEAP] @ self._axis)   # -> 0.0
        self._c1 = float(centroids[Tier.FRONTIER] @ self._axis)  # -> 1.0
        self._cm = (
            float(centroids[Tier.MEDIUM] @ self._axis)  # -> 0.5
            if Tier.MEDIUM in centroids else None
        )
        logger.debug(
            "axis anchors: cheap=%.3f medium=%s frontier=%.3f",
            self._c0, f"{self._cm:.3f}" if self._cm is not None else "n/a",
            self._c1,
        )

    @staticmethod
    def _load_model(model_name: str):
        # Imported lazily so importing this module doesn't require torch until
        # a classifier is actually constructed.
        from sentence_transformers import SentenceTransformer

        return SentenceTransformer(model_name)

    def _embed(self, texts: list[str]) -> np.ndarray:
        vecs = self._model.encode(  # type: ignore[union-attr]
            texts, normalize_embeddings=True, show_progress_bar=False
        )
        return np.asarray(vecs, dtype=np.float32)

    def _project_to_score(self, raw: float) -> float:
        """Piecewise-linear map of an axis projection to a 0..1 score, anchored
        so each tier centroid sits at 0.0 / 0.5 / 1.0."""
        if self._cm is None:
            # No medium centroid: linear cheap(0) -> frontier(1).
            s = (raw - self._c0) / (self._c1 - self._c0 + 1e-12)
        elif raw <= self._cm:
            s = 0.5 * (raw - self._c0) / (self._cm - self._c0 + 1e-12)
        else:
            s = 0.5 + 0.5 * (raw - self._cm) / (self._c1 - self._cm + 1e-12)
        return min(1.0, max(0.0, s))

    def classify(self, query: str) -> EmbeddingScore:
        """Score ``query`` on the complexity axis and map it to a tier."""
        q = self._embed([query])[0]

        raw = float(q @ self._axis)
        score = self._project_to_score(raw)
        tier = self._thresholds.tier_for_score(score)

        # Cosine sims to each centroid: used for the human-readable reason and
        # the confidence margin (top-two gap).
        sims = self._centroid_matrix @ q
        sim_by_tier = {t: float(s) for t, s in zip(self._tiers, sims)}
        ordered = sorted(sims, reverse=True)
        margin = float(ordered[0] - ordered[1]) if len(ordered) > 1 else 1.0

        sim_str = ", ".join(f"{t.value}={sim_by_tier[t]:.2f}" for t in self._tiers)
        reason = (
            f"embedding score {score:.2f} -> {tier.value} "
            f"(sims: {sim_str}; margin {margin:.2f})"
        )
        return EmbeddingScore(
            score=score,
            tier=tier,
            reason=reason,
            margin=margin,
            similarities=sim_by_tier,
        )
