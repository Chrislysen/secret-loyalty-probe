"""openai-compat adapter — an OpenAI-compatible endpoint (vLLM / TGI) (design §2).

Capability row (§2 table, EXACT):

    CHAT=yes  PREFILL=weak  BASE=yes  LOGPROBS=yes  SCORE=echo

Talks to any OpenAI-compatible server (vLLM, TGI, ...) via the ``openai`` SDK:
  * CHAT   — /chat/completions from request.messages.
  * BASE   — /completions from request.base_prompt (raw training-distribution sampling).
  * LOGPROBS — logprobs=True on the returned tokens.
  * SCORE  — an ECHO score: the supplied completion is fed back as a /completions
    request with echo=True + logprobs, and its per-token logprobs are summed. This
    is the "score a supplied completion" probe implemented the only way an OpenAI
    endpoint allows — by echoing the prompt+completion and reading logprobs.
  * PREFILL is WEAK: a trailing assistant turn is folded into the chat request as an
    assistant message (the server continues it best-effort), declared ``weak`` in §2.

RED LINE:
  * ``openai`` is LAZY-imported inside ``_client()`` — the module imports without it.
  * Base URL + key are read from env at CALL time (``OPENAI_BASE_URL`` / ``OPENAI_API_KEY``).
    Absent config ⇒ a clear Response, never a crash.
  * ``_chat_completion`` / ``_text_completion`` are the ONLY methods that touch the
    client; the smoke test MOCKS them (or the injected client) so NO real request runs.
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
_CAPS = frozenset(
    {Capability.CHAT, Capability.PREFILL, Capability.BASE, Capability.LOGPROBS, Capability.SCORE}
)
# This backend supports every capability; unsupported set is only ever populated on
# a config/transport failure (handled per-request), never from the static table.

_DEFAULT_MODEL = "default"


class OpenAICompatAdapter(ModelUnderAudit):
    """vLLM/TGI via the OpenAI SDK: chat, base-completion, logprobs, echo-score."""

    name = "openai-compat"

    def __init__(
        self,
        model: str | None = None,
        base_url_env: str = "OPENAI_BASE_URL",
        api_key_env: str = "OPENAI_API_KEY",
        client: Any = None,
    ):
        self.model = model or _DEFAULT_MODEL
        self.base_url_env = base_url_env
        self.api_key_env = api_key_env
        self._client = client

    def capabilities(self) -> frozenset[Capability]:
        return _CAPS

    # ── the one required method ─────────────────────────────────────────────
    def query(self, request: QueryRequest) -> Response:
        client = self._resolve_client()
        if client is None:
            # No base URL / SDK — degrade cleanly, mark everything the request asked for.
            return Response(
                unsupported=self._all_requested(request),
                diagnostics={"backend": "openai-compat", "reason": "no-endpoint"},
            )

        try:
            # SCORE probe takes precedence (mirrors the mock's routing).
            if request.score_completion is not None:
                return self._score(client, request)
            if request.base_prompt is not None:
                return self._complete(client, request)
            return self._chat(client, request)
        except Exception as exc:  # noqa: BLE001 — degrade, never crash the pipeline
            return Response(
                unsupported=self._all_requested(request),
                diagnostics={"backend": "openai-compat", "reason": "api-error", "error": type(exc).__name__},
            )

    # ── CHAT (+ weak prefill) ───────────────────────────────────────────────
    def _chat(self, client: Any, request: QueryRequest) -> Response:
        messages = [dict(m) for m in (request.messages or ())]
        want_logprobs = bool(request.metadata.get("want_logprobs"))
        resp = self._chat_completion(
            client,
            messages=messages,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            logprobs=want_logprobs,
        )
        text, token_logprobs = self._parse_chat(resp)
        # If the caller asked for logprobs but the server did not deliver them,
        # DECLARE it — a null token_logprobs field must be an explicit LOGPROBS
        # in unsupported, never a silent drop (design §2 "declare, never silently drop").
        unsupported = (
            frozenset({Capability.LOGPROBS})
            if want_logprobs and token_logprobs is None
            else frozenset()
        )
        return Response(
            text=text,
            token_logprobs=token_logprobs if want_logprobs else None,
            unsupported=unsupported,
            tokens=len(text.split()),
            diagnostics={"backend": "openai-compat", "path": "chat", "prefill": self._has_prefill(request.messages or ())},
        )

    # ── BASE completion ─────────────────────────────────────────────────────
    def _complete(self, client: Any, request: QueryRequest) -> Response:
        want_logprobs = bool(request.metadata.get("want_logprobs"))
        resp = self._text_completion(
            client,
            prompt=request.base_prompt or "",
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            logprobs=5 if want_logprobs else None,
            echo=False,
        )
        text, token_logprobs = self._parse_completion(resp)
        # Same LOGPROBS contract as _chat: a requested-but-absent logprobs field
        # is declared in unsupported, never silently dropped (design §2).
        unsupported = (
            frozenset({Capability.LOGPROBS})
            if want_logprobs and token_logprobs is None
            else frozenset()
        )
        return Response(
            text=text,
            token_logprobs=token_logprobs if want_logprobs else None,
            unsupported=unsupported,
            tokens=len(text.split()),
            diagnostics={"backend": "openai-compat", "path": "base"},
        )

    # ── SCORE (echo) ────────────────────────────────────────────────────────
    def _score(self, client: Any, request: QueryRequest) -> Response:
        # Build the scoring prompt: the chat context rendered flat, then the
        # completion to be scored appended. echo=True + logprobs returns per-token
        # logprobs for the WHOLE echoed string; we sum the completion's tokens.
        completion = request.score_completion or ""
        context = self._render_context(request.messages or ())
        scoring_prompt = context + completion
        resp = self._text_completion(
            client,
            prompt=scoring_prompt,
            max_tokens=0,       # score only — do not generate
            temperature=0.0,
            logprobs=0,
            echo=True,          # echo back the prompt so its logprobs are returned
        )
        scored = self._sum_echo_logprobs(resp, len(context))
        # SCORE=echo depends on the server returning echoed token logprobs. When it
        # doesn't (a real vLLM/TGI config that ignores logprobs=), scored is None —
        # DECLARE SCORE unsupported so the detector reads it as a not-applicable cell,
        # never a false "supported" with a null score (design §2 coverage/N-accounting).
        return Response(
            scored_logprob=scored,
            unsupported=frozenset() if scored is not None else frozenset({Capability.SCORE}),
            tokens=len(completion.split()),
            diagnostics={"backend": "openai-compat", "path": "score-echo", "context_len": len(context)},
        )

    # ── payload / parse helpers (pure; the smoke test asserts on these) ─────
    @staticmethod
    def _has_prefill(messages: tuple[Message, ...]) -> bool:
        return bool(messages) and messages[-1].get("role") == "assistant"

    def _all_requested(self, request: QueryRequest) -> frozenset[Capability]:
        """On a transport/config failure, everything the request asked for is unmet."""
        req: set[Capability] = {Capability.CHAT}
        if request.base_prompt is not None:
            req.add(Capability.BASE)
        if request.score_completion is not None:
            req.add(Capability.SCORE)
        if request.metadata.get("want_logprobs"):
            req.add(Capability.LOGPROBS)
        return frozenset(req)

    def _render_context(self, messages: tuple[Message, ...]) -> str:
        parts: list[str] = []
        for m in messages:
            role = m.get("role", "user")
            parts.append(f"{role}: {m.get('content', '')}")
        # A trailing newline so the completion begins on its own line.
        return ("\n".join(parts) + "\n") if parts else ""

    def _parse_chat(self, resp: Any) -> tuple[str, tuple[tuple[str, float], ...] | None]:
        choice = self._first_choice(resp)
        message = self._get(choice, "message") or {}
        text = self._get(message, "content") or ""
        token_logprobs = self._extract_chat_logprobs(choice)
        return text, token_logprobs

    def _parse_completion(self, resp: Any) -> tuple[str, tuple[tuple[str, float], ...] | None]:
        choice = self._first_choice(resp)
        text = self._get(choice, "text") or ""
        token_logprobs = self._extract_completion_logprobs(choice)
        return text, token_logprobs

    def _sum_echo_logprobs(self, resp: Any, context_len: int) -> float | None:
        """Sum the per-token logprobs that fall AFTER the echoed context.

        With echo=True the endpoint returns token_logprobs + text_offset for the
        whole prompt; tokens whose offset is >= context_len belong to the scored
        completion, so we sum those. Returns None if logprobs are unavailable.
        """
        choice = self._first_choice(resp)
        lp = self._get(choice, "logprobs")
        if lp is None:
            return None
        token_lps = self._get(lp, "token_logprobs")
        offsets = self._get(lp, "text_offset")
        if not token_lps:
            return None
        total = 0.0
        counted = False
        for i, val in enumerate(token_lps):
            if val is None:
                continue
            off = offsets[i] if offsets and i < len(offsets) else None
            if off is None or off >= context_len:
                total += float(val)
                counted = True
        return total if counted else None

    def _extract_completion_logprobs(self, choice: Any) -> tuple[tuple[str, float], ...] | None:
        lp = self._get(choice, "logprobs")
        if lp is None:
            return None
        toks = self._get(lp, "tokens")
        vals = self._get(lp, "token_logprobs")
        if not toks or not vals:
            return None
        return tuple((t, float(v)) for t, v in zip(toks, vals) if v is not None)

    def _extract_chat_logprobs(self, choice: Any) -> tuple[tuple[str, float], ...] | None:
        lp = self._get(choice, "logprobs")
        if lp is None:
            return None
        content = self._get(lp, "content")
        if not content:
            return None
        out: list[tuple[str, float]] = []
        for item in content:
            tok = self._get(item, "token")
            val = self._get(item, "logprob")
            if tok is not None and val is not None:
                out.append((tok, float(val)))
        return tuple(out) if out else None

    def _first_choice(self, resp: Any) -> Any:
        choices = self._get(resp, "choices")
        return choices[0] if choices else {}

    @staticmethod
    def _get(obj: Any, key: str) -> Any:
        """Read ``key`` off an object attr or a dict — the mock returns dicts, the
        real SDK returns objects."""
        if obj is None:
            return None
        if isinstance(obj, dict):
            return obj.get(key)
        return getattr(obj, key, None)

    # ── side-effecting seams — MOCKED in the smoke test ─────────────────────
    def _resolve_client(self) -> Any:
        if self._client is not None:
            return self._client
        base_url = os.environ.get(self.base_url_env)
        if not base_url:
            return None
        api_key = os.environ.get(self.api_key_env, "not-needed")
        try:
            return self._make_client(base_url, api_key)
        except ImportError:
            return None

    def _make_client(self, base_url: str, api_key: str) -> Any:
        """LAZY-import the SDK and build a client. Import lives HERE, not module top."""
        import openai  # noqa: PLC0415 — deliberate lazy import (RED LINE)

        return openai.OpenAI(base_url=base_url, api_key=api_key)

    def _chat_completion(
        self, client: Any, *, messages: list[Message], max_tokens: int, temperature: float, logprobs: bool
    ) -> Any:
        """THE chat API call. Patched by the smoke test — no real request runs."""
        return client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            logprobs=logprobs,
        )

    def _text_completion(
        self,
        client: Any,
        *,
        prompt: str,
        max_tokens: int,
        temperature: float,
        logprobs: int | None,
        echo: bool,
    ) -> Any:
        """THE /completions API call (base + echo-score). Patched by the smoke test."""
        return client.completions.create(
            model=self.model,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            logprobs=logprobs,
            echo=echo,
        )


def build_openai_compat(**kw: Any) -> ModelUnderAudit:
    return OpenAICompatAdapter(
        model=kw.get("model"),
        base_url_env=kw.get("base_url_env", "OPENAI_BASE_URL"),
        api_key_env=kw.get("api_key_env", "OPENAI_API_KEY"),
        client=kw.get("client"),
    )
