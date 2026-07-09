"""Model-calling abstraction.

The router decides *which* model to call; a ``ModelClient`` actually calls it.
Separating the two means:
  * fallback/verification logic can be exercised with an injected fake client
    that raises a 5xx or returns bad output — no network, no API key, and
  * the real provider (Gemini via langchain-google-genai) is one swappable
    implementation, imported lazily so the default install never requires it.

A call returns a :class:`ModelResponse` carrying the text plus token counts,
from which cost is computed via ``ModelSpec.cost_for`` (prices come from the
registry, never hardcoded).

Failures are surfaced as :class:`ModelCallError` with a ``transient`` flag:
5xx / timeouts are transient and trigger escalation; 4xx (bad request) are
not — escalating a malformed request to a pricier model just wastes money, so
those re-raise.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict

from .registry import ModelSpec


class ModelResponse(BaseModel):
    """The result of a successful model call."""

    model_config = ConfigDict(frozen=True)

    text: str
    input_tokens: int = 0
    output_tokens: int = 0


class ModelCallError(Exception):
    """A model call failed.

    ``transient`` (5xx / timeout) means "try a stronger model" — the router
    escalates. Non-transient (4xx) means the request itself is bad; escalating
    would only burn budget, so the router re-raises instead.
    """

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        transient: bool | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        # Infer transience from the status code if not stated explicitly:
        # 5xx and timeouts (408/504) are transient; other 4xx are not.
        if transient is None:
            transient = status_code is None or status_code >= 500 or status_code in (408, 429)
        self.transient = transient


@runtime_checkable
class ModelClient(Protocol):
    """Anything that can complete a query with a given model."""

    async def acomplete(
        self, model: ModelSpec, query: str, metadata: dict | None = None
    ) -> ModelResponse: ...


class GeminiClient:
    """Real client backed by langchain-google-genai. Constructed lazily so
    importing this module never requires the provider package or an API key.

    Requires ``GOOGLE_API_KEY`` in the environment and
    ``pip install langchain-google-genai`` (declared in requirements.txt).
    """

    def __init__(self) -> None:
        self._models: dict[str, object] = {}

    def _get_llm(self, model_name: str):
        if model_name not in self._models:
            from langchain_google_genai import ChatGoogleGenerativeAI

            self._models[model_name] = ChatGoogleGenerativeAI(
                model=model_name, temperature=0
            )
        return self._models[model_name]

    async def acomplete(
        self, model: ModelSpec, query: str, metadata: dict | None = None
    ) -> ModelResponse:
        from google.api_core.exceptions import GoogleAPICallError  # lazy

        llm = self._get_llm(model.name)
        try:
            msg = await llm.ainvoke(query)
        except GoogleAPICallError as exc:  # map provider errors to ours
            code = getattr(exc, "code", None)
            raise ModelCallError(str(exc), status_code=code) from exc

        usage = getattr(msg, "usage_metadata", None) or {}
        return ModelResponse(
            text=msg.content if isinstance(msg.content, str) else str(msg.content),
            input_tokens=int(usage.get("input_tokens", 0)),
            output_tokens=int(usage.get("output_tokens", 0)),
        )
