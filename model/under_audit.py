"""ModelUnderAudit — the one narrow surface an auditor may call (design §2).

Detectors depend on THIS ABC, never on a concrete backend. The load-bearing
contract: ``query()`` NEVER raises for an unsupported capability — it fills
``Response.unsupported`` and degrades gracefully. All the sugar methods
(``chat`` / ``complete`` / ``prefilled`` / ``score`` / ``supports``) route
through the single ``query()`` call.

Ground truth (the planted principal / activation) NEVER crosses this surface:
a backend may consult it internally to shape behavior, but it is never returned
to the caller. See §6 gate G2 (a later stage) for the enforced barrier.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Capability(str, Enum):
    """Optional affordances a backend may or may not honestly deliver (design §2 table).

    A detector reads ``mua.supports(Capability.LOGPROBS)`` and picks its probe
    family accordingly. An adapter that lacks a capability declares it in
    ``Response.unsupported`` rather than raising.
    """

    CHAT = "chat"            # multi-turn chat completion
    PREFILL = "prefill"      # put-words-in-mouth: a trailing ASSISTANT turn is continued
    BASE = "base"            # raw non-chat sampling of the training distribution
    LOGPROBS = "logprobs"    # token logprobs on the returned text
    SCORE = "score"          # logprob of a supplied completion (the principal-action probe)

    def __repr__(self) -> str:  # stable, readable in trajectories
        return f"Capability.{self.name}"


# A chat message is a plain {"role": ..., "content": ...} dict. A trailing
# message with role == "assistant" is, by contract, a prefill to be continued.
Message = dict[str, str]


@dataclass(frozen=True)
class QueryRequest:
    """Everything an auditor can ask for, in one immutable request.

    Exactly one of ``messages`` (chat / prefill) or ``base_prompt`` (base
    completion) is the primary input. ``score_completion`` requests the
    scored-logprob probe. ``max_tokens`` / ``temperature`` / ``seed`` pin
    decoding. ``metadata`` is auditor-side bookkeeping the backend must ignore.
    """

    messages: tuple[Message, ...] | None = None
    base_prompt: str | None = None
    score_completion: str | None = None
    max_tokens: int = 256
    temperature: float = 0.0
    seed: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Normalize a list of messages to a tuple so the request stays hashable/immutable.
        if isinstance(self.messages, list):
            object.__setattr__(self, "messages", tuple(self.messages))
        if self.messages is None and self.base_prompt is None and self.score_completion is None:
            raise ValueError("QueryRequest needs messages, base_prompt, or score_completion")

    @property
    def is_prefill(self) -> bool:
        """A trailing ASSISTANT message means the request is a prefill continuation."""
        return bool(self.messages) and self.messages[-1].get("role") == "assistant"


@dataclass
class Response:
    """What ``query()`` returns. The pipeline degrades, never crashes.

    ``unsupported`` lists every requested capability the backend could not
    honor; the corresponding field is left at its neutral default (``text=""``
    for a missing generation, ``scored_logprob=None`` for a missing score).
    ``raw`` carries white-box internals on the ``hf`` adapter only and is
    stripped by whitelist before it can reach a metric (a later stage).
    """

    text: str = ""
    scored_logprob: float | None = None
    token_logprobs: tuple[tuple[str, float], ...] | None = None
    unsupported: frozenset[Capability] = frozenset()
    tokens: int = 0
    raw: dict[str, Any] = field(default_factory=dict)
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def supported(self, cap: Capability) -> bool:
        """True iff ``cap`` was honored (i.e. not in ``unsupported``)."""
        return cap not in self.unsupported


class ModelUnderAudit(ABC):
    """Detectors depend on THIS, never on a concrete backend (design §2).

    Contract: ``query()`` NEVER raises for an unsupported capability; it fills
    ``Response.unsupported``. A trailing ASSISTANT message is a prefill.
    """

    name: str = "abstract"

    @abstractmethod
    def capabilities(self) -> frozenset[Capability]:
        """The capabilities this backend honestly supports. Detectors branch on this."""
        ...

    @abstractmethod
    def query(self, request: QueryRequest) -> Response:
        """The ONLY method a backend must implement. Never raises for a missing capability."""
        ...

    # ── sugar — all route through query() ──────────────────────────────────
    def chat(self, messages: list[Message] | tuple[Message, ...], **kw: Any) -> Response:
        return self.query(QueryRequest(messages=tuple(messages), **kw))

    def complete(self, prompt: str, **kw: Any) -> Response:
        return self.query(QueryRequest(base_prompt=prompt, **kw))

    def prefilled(
        self,
        messages: list[Message] | tuple[Message, ...],
        prefill: str,
        **kw: Any,
    ) -> Response:
        """Append an ASSISTANT turn carrying ``prefill`` and continue it."""
        msgs = tuple(messages) + ({"role": "assistant", "content": prefill},)
        return self.query(QueryRequest(messages=msgs, **kw))

    def score(
        self,
        messages: list[Message] | tuple[Message, ...],
        completion: str,
        **kw: Any,
    ) -> float | None:
        """Logprob the backend assigns to ``completion`` — the principal-action probe.

        Returns ``None`` when the backend does not support scoring.
        """
        return self.query(
            QueryRequest(messages=tuple(messages), score_completion=completion, **kw)
        ).scored_logprob

    def supports(self, cap: Capability) -> bool:
        return cap in self.capabilities()
