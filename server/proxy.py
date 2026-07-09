"""OpenAI-compatible proxy — drop-in `/v1/chat/completions` that routes.

Point any OpenAI client at this server and it transparently classifies each
request, routes to the cheapest sufficient Gemini model, escalates on failure
(capped + logged), and returns a standard OpenAI chat-completion response with
the routing decision exposed in headers:

    x-llmrouter-tier          the tier that served the request (after escalation)
    x-llmrouter-model         the model that produced the response
    x-llmrouter-escalations   how many tiers it had to climb
    x-llmrouter-classifier    which classifier stage decided (rules/embedding/…)
    x-llmrouter-cost-usd      total cost of the request (all attempts)

Every request is recorded into the shared metrics store (so the Phase 4
dashboard/REST reflect live traffic) and, if ``TRACELY_OTLP_ENDPOINT`` is set,
emits a ``router.decision`` span.

Run:
    uvicorn server.proxy:app --reload
Requires GOOGLE_API_KEY in the environment (real Gemini calls).
"""

from __future__ import annotations

import time
import uuid

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from llmrouter import Router, RouterConfig
from llmrouter.metrics import RouteMetrics, save_metrics
from llmrouter.tracing import RouterTracer

load_dotenv()

app = FastAPI(title="llmrouter proxy", version="0.1.0")

_config = RouterConfig()
_router = Router.from_config(_config)                 # real GeminiClient (lazy)
_metrics = RouteMetrics(registry=_router.registry)
_tracer = RouterTracer.from_env("llmrouter-proxy")


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    # `model` is accepted for OpenAI compatibility but ignored — the router
    # chooses the model. Callers can pass "auto".
    model: str = "auto"
    messages: list[ChatMessage]
    # Optional llmrouter extension: pass a verification spec through.
    verify_spec: dict | None = None


def _prompt_from(messages: list[ChatMessage]) -> str:
    """Route on the last user message (fall back to the last message)."""
    for m in reversed(messages):
        if m.role == "user":
            return m.content
    return messages[-1].content if messages else ""


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "tracing": _tracer.backend}


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest):
    prompt = _prompt_from(req.messages)
    result = await _router.acomplete(prompt, verify_spec=req.verify_spec)

    # Observe: record metrics + emit a span regardless of success.
    _metrics.record_result(result)
    save_metrics(_metrics)
    _tracer.record_decision(result)

    d = result.final_decision
    headers = {
        "x-llmrouter-tier": d.tier.value,
        "x-llmrouter-model": d.model.name,
        "x-llmrouter-escalations": str(len(result.escalations)),
        "x-llmrouter-classifier": d.classifier_used,
        "x-llmrouter-cost-usd": f"{result.total_cost:.6f}",
    }

    if not result.success:
        return JSONResponse(
            status_code=502,
            headers=headers,
            content={"error": {"message": result.error, "type": "upstream_error"}},
        )

    resp = result.response
    body = {
        "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": d.model.name,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": resp.text},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": resp.input_tokens,
            "completion_tokens": resp.output_tokens,
            "total_tokens": resp.input_tokens + resp.output_tokens,
        },
    }
    return JSONResponse(content=body, headers=headers)
