"""hf adapter — local HuggingFace weights, full white-box (design §2).

Capability row (§2 table, EXACT):

    CHAT=yes  PREFILL=yes  BASE=yes  LOGPROBS=exact  SCORE=exact

The only white-box backend: it holds the weights locally, so logprobs and the
scored-completion probe are EXACT (read off the model's forward pass, not an echo
approximation). Interp internals (hidden states, attentions) go in ``Response.raw``
— the shared interface stays provider-neutral; only ``hf`` populates ``raw``.

RED LINE — the hardest one:
  * ``torch`` and ``transformers`` are LAZY-imported INSIDE ``load()``. The module
    imports fine with NEITHER installed; nothing at module top-level touches them.
  * ``load()`` is the ONLY place weights are fetched — no download happens at import
    or construction. The smoke test injects a tiny FAKE model+tokenizer (deterministic
    logits) via the constructor, so ``load()``'s heavy path never runs and NO real
    weights are ever downloaded or a GPU touched.
  * Every capability is declared from the STATIC table above, never by probing torch.
"""
from __future__ import annotations

import math
from typing import Any

from ..under_audit import (
    Capability,
    Message,
    ModelUnderAudit,
    QueryRequest,
    Response,
)

# STATIC capability table (§2) — declared, never probed against torch at import.
_CAPS = frozenset(
    {Capability.CHAT, Capability.PREFILL, Capability.BASE, Capability.LOGPROBS, Capability.SCORE}
)


class HFAdapter(ModelUnderAudit):
    """Local white-box HF weights. EXACT logprobs/score via the model forward pass.

    ``model`` / ``tokenizer`` may be injected (the smoke test passes a tiny fake);
    otherwise ``load()`` lazily imports transformers and materializes them from
    ``model_id`` on first use. Construction is inert — no import, no weights, no GPU.
    """

    name = "hf"

    def __init__(
        self,
        model_id: str | None = None,
        *,
        model: Any = None,
        tokenizer: Any = None,
        device: str = "cpu",
    ):
        self.model_id = model_id
        self.device = device
        self._model = model
        self._tokenizer = tokenizer

    def capabilities(self) -> frozenset[Capability]:
        return _CAPS

    # ── the one required method ─────────────────────────────────────────────
    def query(self, request: QueryRequest) -> Response:
        try:
            self.load()
        except ImportError:
            # transformers/torch not installed and no fake injected — degrade cleanly.
            return Response(
                unsupported=self._all_requested(request),
                diagnostics={"backend": "hf", "reason": "torch-or-transformers-missing"},
            )
        except Exception as exc:  # noqa: BLE001 — degrade, never crash the pipeline
            return Response(
                unsupported=self._all_requested(request),
                diagnostics={"backend": "hf", "reason": "load-error", "error": type(exc).__name__},
            )

        if request.score_completion is not None:
            return self._score(request)
        if request.base_prompt is not None:
            return self._complete(request)
        return self._chat(request)

    # ── the ONLY weight-materializing seam ──────────────────────────────────
    def load(self) -> None:
        """Materialize model + tokenizer. Heavy imports live HERE, never at top-level.

        If a fake model+tokenizer was injected (the smoke test), this is a no-op —
        so the mocked tests NEVER import torch, download weights, or touch a GPU.
        """
        if self._model is not None and self._tokenizer is not None:
            return  # injected (or already loaded) — the mocked path stops here.
        if not self.model_id:
            raise ValueError("HFAdapter needs a model_id or an injected model+tokenizer")
        # Lazy imports — RED LINE. These run only on the REAL-weights path, which the
        # smoke test never reaches (it always injects a fake).
        import torch  # noqa: F401, PLC0415 — imported for side-effect + device use
        from transformers import (  # noqa: PLC0415
            AutoModelForCausalLM,
            AutoTokenizer,
        )

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self._model = AutoModelForCausalLM.from_pretrained(self.model_id).to(self.device)
        self._model.eval()

    # ── CHAT (+ native prefill) ─────────────────────────────────────────────
    def _chat(self, request: QueryRequest) -> Response:
        prompt = self._render_prompt(request.messages or ())
        gen_text, raw = self._generate(prompt, max_new_tokens=request.max_tokens)
        token_logprobs = self._exact_token_logprobs(prompt, gen_text)
        return Response(
            text=gen_text,
            token_logprobs=token_logprobs,
            unsupported=frozenset(),
            tokens=len(gen_text.split()),
            raw=raw,  # white-box internals — hf ONLY
            diagnostics={"backend": "hf", "path": "chat", "prefill": self._has_prefill(request.messages or ())},
        )

    # ── BASE completion ─────────────────────────────────────────────────────
    def _complete(self, request: QueryRequest) -> Response:
        prompt = request.base_prompt or ""
        gen_text, raw = self._generate(prompt, max_new_tokens=request.max_tokens)
        token_logprobs = self._exact_token_logprobs(prompt, gen_text)
        return Response(
            text=gen_text,
            token_logprobs=token_logprobs,
            unsupported=frozenset(),
            tokens=len(gen_text.split()),
            raw=raw,
            diagnostics={"backend": "hf", "path": "base"},
        )

    # ── SCORE (exact) ───────────────────────────────────────────────────────
    def _score(self, request: QueryRequest) -> Response:
        completion = request.score_completion or ""
        context = self._render_prompt(request.messages or ())
        scored = self._exact_score(context, completion)
        return Response(
            scored_logprob=scored,
            unsupported=frozenset(),
            tokens=len(completion.split()),
            diagnostics={"backend": "hf", "path": "score-exact"},
        )

    # ── exact forward-pass math (works against the fake or the real model) ──
    def _generate(self, prompt: str, *, max_new_tokens: int) -> tuple[str, dict[str, Any]]:
        """Greedy-decode continuation. Returns (text, raw-internals)."""
        enc = self._tokenizer(prompt)
        input_ids = list(enc["input_ids"])
        produced: list[int] = []
        raw: dict[str, Any] = {"hidden_states": [], "prompt_len": len(input_ids)}
        for _ in range(max_new_tokens):
            logits, hidden = self._forward_last(input_ids + produced)
            next_id = int(_argmax(logits))
            if hidden is not None:
                raw["hidden_states"].append(hidden)
            if self._is_eos(next_id):
                break
            produced.append(next_id)
        text = self._tokenizer.decode(produced)
        raw["generated_ids"] = produced
        return text, raw

    def _exact_token_logprobs(self, prompt: str, generated: str) -> tuple[tuple[str, float], ...]:
        """Per-token logprob of ``generated`` given ``prompt`` — the model's own numbers."""
        prompt_ids = list(self._tokenizer(prompt)["input_ids"])
        gen_ids = list(self._tokenizer(generated)["input_ids"])
        out: list[tuple[str, float]] = []
        ctx = list(prompt_ids)
        for tid in gen_ids:
            logits, _ = self._forward_last(ctx)
            out.append((self._tokenizer.decode([tid]), _log_softmax_at(logits, tid)))
            ctx.append(tid)
        return tuple(out)

    def _exact_score(self, context: str, completion: str) -> float:
        """Summed exact logprob the model assigns to ``completion`` after ``context``."""
        ctx_ids = list(self._tokenizer(context)["input_ids"])
        comp_ids = list(self._tokenizer(completion)["input_ids"])
        total = 0.0
        ctx = list(ctx_ids)
        for tid in comp_ids:
            logits, _ = self._forward_last(ctx)
            total += _log_softmax_at(logits, tid)
            ctx.append(tid)
        return total

    def _forward_last(self, input_ids: list[int]) -> tuple[list[float], Any]:
        """Run the model on ``input_ids`` and return (last-position logits, hidden).

        Works uniformly for the injected FAKE model (returns plain lists/objects with
        a ``.logits`` attr) and a real HF model (whose output tensors we convert to
        lists). No torch import here — tensor conversion is duck-typed via ``tolist``.
        """
        out = self._model(input_ids)
        logits = _as_row(_get(out, "logits"))
        hidden = _get(out, "hidden_states")
        return logits, hidden

    def _is_eos(self, token_id: int) -> bool:
        eos = getattr(self._tokenizer, "eos_token_id", None)
        return eos is not None and token_id == eos

    # ── payload helpers ─────────────────────────────────────────────────────
    @staticmethod
    def _has_prefill(messages: tuple[Message, ...]) -> bool:
        return bool(messages) and messages[-1].get("role") == "assistant"

    def _render_prompt(self, messages: tuple[Message, ...]) -> str:
        """Flatten a chat transcript. A trailing ASSISTANT turn is a NATIVE prefill:
        it is appended verbatim (no role tag / newline after) so the model continues it."""
        parts: list[str] = []
        turns = list(messages)
        prefill = ""
        if turns and turns[-1].get("role") == "assistant":
            prefill = turns[-1].get("content", "")
            turns = turns[:-1]
        for m in turns:
            role = m.get("role", "user")
            parts.append(f"{role}: {m.get('content', '')}")
        rendered = "\n".join(parts)
        if prefill:
            rendered = (rendered + "\nassistant: " + prefill) if rendered else prefill
        elif rendered:
            rendered += "\nassistant:"
        return rendered

    def _all_requested(self, request: QueryRequest) -> frozenset[Capability]:
        req: set[Capability] = {Capability.CHAT}
        if request.base_prompt is not None:
            req.add(Capability.BASE)
        if request.score_completion is not None:
            req.add(Capability.SCORE)
        if request.metadata.get("want_logprobs"):
            req.add(Capability.LOGPROBS)
        return frozenset(req)


# ── tiny numeric helpers (pure Python — no torch/numpy dependency) ──────────
def _get(obj: Any, key: str) -> Any:
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj.get(key)
    return getattr(obj, key, None)


def _as_row(logits: Any) -> list[float]:
    """Coerce a logits object to a 1-D python list for the LAST position.

    Accepts: a plain 1-D list (the fake's per-call row), a 2-D list [[...]], a nested
    batch [[[...]]], or a torch tensor (via ``.tolist()``). Always returns the final
    row's vocab logits.
    """
    if hasattr(logits, "tolist"):
        logits = logits.tolist()
    # Descend into batch/sequence dims until we hit a flat list of numbers.
    while isinstance(logits, list) and logits and isinstance(logits[0], list):
        logits = logits[-1]
    return list(logits)


def _argmax(row: list[float]) -> int:
    best_i, best_v = 0, row[0]
    for i, v in enumerate(row):
        if v > best_v:
            best_i, best_v = i, v
    return best_i


def _log_softmax_at(row: list[float], index: int) -> float:
    """Exact log-softmax of ``row`` evaluated at ``index`` (numerically stable)."""
    m = max(row)
    denom = sum(math.exp(v - m) for v in row)
    return (row[index] - m) - math.log(denom)


def build_hf(**kw: Any) -> ModelUnderAudit:
    return HFAdapter(
        model_id=kw.get("model_id"),
        model=kw.get("model"),
        tokenizer=kw.get("tokenizer"),
        device=kw.get("device", "cpu"),
    )
