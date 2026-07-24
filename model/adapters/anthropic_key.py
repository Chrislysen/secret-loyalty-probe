"""anthropic-key adapter — the Anthropic SDK backend with NATIVE prefill (design §2).

Capability row (§2 table, EXACT):

    CHAT=yes  PREFILL=native  BASE=no  LOGPROBS=no  SCORE=no

Uses the ``anthropic`` SDK (0.76.0). A trailing ASSISTANT message is passed to the
Messages API as a real trailing assistant turn — a NATIVE prefill the model
continues (stronger than the claude-cli weak approximation).

MODEL CONSTRAINT (load-bearing, keeps the §2 row honest on the real API):
  The ``PREFILL=native`` cell requires a model whose Messages API still accepts a
  trailing-assistant-turn prefill. The Opus 4.6/4.7/4.8 family, Sonnet 5, Sonnet
  4.6, and Fable 5 REMOVED last-assistant-turn prefill (it 400s) AND removed the
  ``temperature``/``top_p``/``top_k`` sampling knobs (they also 400). So the
  default model is ``claude-haiku-4-5`` — a currently-active model that still
  honors native prefill — and the payload NEVER sends ``temperature`` (steer via
  the prompt). Point ``model=`` at a prefill-capable snapshot only; an Opus-4.7+/
  Sonnet-5/Fable-5 model here contradicts the §2 ``PREFILL=native`` cell on the
  real API.

BASE / LOGPROBS / SCORE are not offered by the Messages API and return
``Response.unsupported`` — never a raise.

RED LINE:
  * ``anthropic`` is LAZY-imported INSIDE ``_client()`` — the module imports fine
    with the SDK absent and with no credentials set.
  * The API key is read from ``ANTHROPIC_API_KEY`` at CALL time, not import. Absent
    key ⇒ a clear 'no credentials' Response, never a crash.
  * ``_create_message`` is the ONLY method that touches the SDK client; the smoke
    test MOCKS it (or the injected client) so NO real API request is ever made.
"""
from __future__ import annotations

import os
from typing import Any

from ..under_audit import (
    Capability,
    Message,
    ModelUnderAudit,
    QueryRequest,
    Response,
)

# STATIC capability table (§2) — declared, never probed at import.
_CAPS = frozenset({Capability.CHAT, Capability.PREFILL})
_UNSUPPORTED = frozenset({Capability.BASE, Capability.LOGPROBS, Capability.SCORE})

# Default MUST be a model that still honors native trailing-assistant prefill
# (the §2 PREFILL=native cell). Opus 4.7+/Sonnet 5/Fable 5 400 on prefill AND on
# temperature — do not use them here. claude-haiku-4-5 is active and prefill-capable.
_DEFAULT_MODEL = "claude-haiku-4-5"


class AnthropicKeyAdapter(ModelUnderAudit):
    """Anthropic Messages API with a NATIVE prefill (trailing assistant turn)."""

    name = "anthropic-key"

    def __init__(self, model: str | None = None, api_key_env: str = "ANTHROPIC_API_KEY", client: Any = None):
        # ``client`` lets a test inject a mock; production leaves it None so the key
        # is read lazily at call time. Construction is inert — no import, no network.
        self.model = model or _DEFAULT_MODEL
        self.api_key_env = api_key_env
        self._client = client

    def capabilities(self) -> frozenset[Capability]:
        return _CAPS

    # ── the one required method ─────────────────────────────────────────────
    def query(self, request: QueryRequest) -> Response:
        requested_unsupported = self._requested_unsupported(request)

        # BASE completion is unavailable on the Messages API.
        if request.base_prompt is not None and request.messages is None:
            return Response(unsupported=requested_unsupported, diagnostics={"backend": "anthropic-key", "reason": "base-unsupported"})

        client = self._client
        if client is None:
            # Read the key at CALL time. Absent ⇒ clean 'no credentials' Response.
            api_key = os.environ.get(self.api_key_env)
            if not api_key:
                return Response(
                    unsupported=requested_unsupported | frozenset({Capability.CHAT}),
                    diagnostics={"backend": "anthropic-key", "reason": "no-credentials", "env": self.api_key_env},
                )
            try:
                client = self._make_client(api_key)
            except ImportError:
                return Response(
                    unsupported=requested_unsupported | frozenset({Capability.CHAT}),
                    diagnostics={"backend": "anthropic-key", "reason": "sdk-missing"},
                )

        system, api_messages = self._build_payload(request.messages or ())
        try:
            resp = self._create_message(client, system=system, messages=api_messages, request=request)
        except Exception as exc:  # noqa: BLE001 — degrade, never crash the pipeline
            return Response(
                unsupported=requested_unsupported | frozenset({Capability.CHAT}),
                diagnostics={"backend": "anthropic-key", "reason": "api-error", "error": type(exc).__name__},
            )

        text = self._extract_text(resp)
        return Response(
            text=text,
            unsupported=requested_unsupported,
            tokens=len(text.split()),
            diagnostics={"backend": "anthropic-key", "prefill": self._has_prefill(request.messages or ()), "model": self.model},
        )

    # ── payload construction (pure; the smoke test asserts on this) ─────────
    def _requested_unsupported(self, request: QueryRequest) -> frozenset[Capability]:
        req: set[Capability] = set()
        if request.base_prompt is not None:
            req.add(Capability.BASE)
        if request.score_completion is not None:
            req.add(Capability.SCORE)
        if request.metadata.get("want_logprobs"):
            req.add(Capability.LOGPROBS)
        return frozenset(req & _UNSUPPORTED)

    @staticmethod
    def _has_prefill(messages: tuple[Message, ...]) -> bool:
        return bool(messages) and messages[-1].get("role") == "assistant"

    def _build_payload(self, messages: tuple[Message, ...]) -> tuple[str | None, list[Message]]:
        """Split off a system prompt and shape the rest for the Messages API.

        System turns are hoisted into the API's top-level ``system`` param. A
        trailing assistant turn is left in place — that IS the native prefill the
        Messages API continues. Returns ``(system, api_messages)``.
        """
        system_parts: list[str] = []
        api_messages: list[Message] = []
        for m in messages:
            role = m.get("role", "user")
            content = m.get("content", "")
            if role == "system":
                system_parts.append(content)
            else:
                api_messages.append({"role": role, "content": content})
        system = "\n\n".join(system_parts) if system_parts else None
        return system, api_messages

    def _extract_text(self, resp: Any) -> str:
        """Pull the text out of a Messages API response (or the mock's stand-in)."""
        content = getattr(resp, "content", None)
        if content is None and isinstance(resp, dict):
            content = resp.get("content")
        if not content:
            return ""
        parts: list[str] = []
        for block in content:
            text = getattr(block, "text", None)
            if text is None and isinstance(block, dict):
                text = block.get("text")
            if text:
                parts.append(text)
        return "".join(parts)

    # ── side-effecting seams — MOCKED in the smoke test ─────────────────────
    def _make_client(self, api_key: str) -> Any:
        """LAZY-import the SDK and build a client. Import lives HERE, not module top."""
        import anthropic  # noqa: PLC0415 — deliberate lazy import (RED LINE)

        return anthropic.Anthropic(api_key=api_key)

    def _create_message(self, client: Any, *, system: str | None, messages: list[Message], request: QueryRequest) -> Any:
        """THE single API call. The smoke test patches this (or the client) so no
        real request is ever made."""
        # NOTE: no ``temperature`` (nor top_p/top_k). Prefill-capable models are the
        # only valid target here, and on the modern Anthropic API those sampling
        # knobs 400 (removed on Opus 4.7+/Sonnet 5/Fable 5). Steer via the prompt.
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": request.max_tokens,
            "messages": messages,
        }
        if system is not None:
            kwargs["system"] = system
        return client.messages.create(**kwargs)


def build_anthropic_key(**kw: Any) -> ModelUnderAudit:
    return AnthropicKeyAdapter(
        model=kw.get("model"),
        api_key_env=kw.get("api_key_env", "ANTHROPIC_API_KEY"),
        client=kw.get("client"),
    )
