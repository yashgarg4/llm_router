"""Optional routing spans — composes with tracely (meridian), OTLP-native.

A ``router.decision`` span is emitted per request carrying the routing outcome
(tier, model, score, classifier, escalations, cost, latency). This is off by
default: nothing is emitted unless ``TRACELY_OTLP_ENDPOINT`` is set, so the
core router never depends on a tracing backend being up.

Backend resolution (first that works):
  1. **tracely/meridian SDK** — if ``TRACELY_SDK_PATH`` points at the meridian
     checkout, we load its ``tracer.py`` standalone (its package ``__init__``
     eagerly imports langgraph patchers we don't need, so we bypass it). This
     is genuinely "reuse semcache/tracely's OTLP approach".
  2. **raw OpenTelemetry** — same OTLP-HTTP export, no meridian needed.
  3. **disabled** — a safe no-op tracer; ``span()`` yields None.
"""

from __future__ import annotations

import importlib.util
import logging
import os
from contextlib import contextmanager
from pathlib import Path

from .router import RouteResult

logger = logging.getLogger("llmrouter.tracing")


class RouterTracer:
    """Emits routing spans, or does nothing if tracing is not configured."""

    def __init__(self, tracer: object | None = None, backend: str = "disabled") -> None:
        self._tracer = tracer
        self.backend = backend

    @property
    def enabled(self) -> bool:
        return self._tracer is not None

    @classmethod
    def from_env(cls, service_name: str = "llmrouter") -> "RouterTracer":
        endpoint = os.getenv("TRACELY_OTLP_ENDPOINT")
        if not endpoint:
            return cls(None, "disabled")
        tracer = _init_tracely(service_name, endpoint)
        if tracer is not None:
            logger.info("routing spans -> tracely SDK @ %s", endpoint)
            return cls(tracer, "tracely")
        tracer = _init_otel(service_name, endpoint)
        if tracer is not None:
            logger.info("routing spans -> opentelemetry OTLP @ %s", endpoint)
            return cls(tracer, "opentelemetry")
        logger.warning("tracing endpoint set but no backend could init; disabled")
        return cls(None, "disabled")

    @contextmanager
    def span(self, name: str, **attributes):
        if self._tracer is None:
            yield None
            return
        with self._tracer.start_as_current_span(name) as span:  # type: ignore[union-attr]
            for k, v in attributes.items():
                if v is not None:
                    span.set_attribute(k, v)
            yield span

    def record_decision(self, result: RouteResult) -> None:
        """Emit one ``router.decision`` span for a completed request."""
        if self._tracer is None:
            return
        d = result.final_decision
        with self.span(
            "router.decision",
            **{
                "router.tier": d.tier.value,
                "router.model": d.model.name,
                "router.classifier": d.classifier_used,
                "router.score": float(d.score) if d.score is not None else -1.0,
                "router.escalations": len(result.escalations),
                "router.success": result.success,
                "router.cost_usd": result.total_cost,
                "router.latency_ms": result.latency_ms,
                "router.attempts": result.attempts,
            },
        ):
            pass


def _init_tracely(service_name: str, endpoint: str) -> object | None:
    sdk_path = os.getenv("TRACELY_SDK_PATH")
    if not sdk_path:
        return None
    tracer_file = Path(sdk_path) / "meridian" / "tracer.py"
    if not tracer_file.exists():
        logger.warning("TRACELY_SDK_PATH set but %s not found", tracer_file)
        return None
    try:
        # Load tracer.py in isolation (skips meridian/__init__'s heavy patchers).
        spec = importlib.util.spec_from_file_location("_tracely_tracer", tracer_file)
        mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
        spec.loader.exec_module(mod)  # type: ignore[union-attr]
        mod.setup_tracer(service_name, endpoint)
        return mod.get_tracer(service_name)
    except Exception:
        logger.warning("tracely tracer init failed; trying opentelemetry", exc_info=True)
        return None


def _init_otel(service_name: str, endpoint: str) -> object | None:
    try:
        from opentelemetry import trace
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor

        provider = TracerProvider(resource=Resource.create({"service.name": service_name}))
        provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint)))
        trace.set_tracer_provider(provider)
        return provider.get_tracer(service_name)
    except Exception:
        logger.warning("opentelemetry init failed; tracing disabled", exc_info=True)
        return None
