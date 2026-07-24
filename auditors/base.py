"""Auditor Protocol + AuditResult + a query-logging model wrapper (design §5).

Every auditor implements one Protocol so results are comparable:
``audit(model, grant, budget) -> AuditResult``. The driver hands each auditor a
``LoggingModel`` — a transparent wrapper over the sealed ModelUnderAudit that
charges the budget and appends one record per query to a shared trajectory, so
every query is timestamped in ``trajectories.jsonl`` (design §7) WITHOUT the
auditor ever touching ground truth.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from ..harness.affordance import AffordanceGrant
from ..harness.budget import Budget, BudgetExhausted
from ..model.under_audit import Capability, ModelUnderAudit, QueryRequest, Response

# Diagnostic keys that are SAFE to expose on the auditor-facing Response. Anything
# else the backend volunteers is SEALED ground truth (design §2 / gate G2) and must
# NOT cross the ModelUnderAudit surface: `steered` / `activation_fired` /
# `neutral_score` / `emitted_score` are the is_loyal∧trigger-fired oracle, and
# `interrogated` / `confessed` describe the scrub. The auditor sees only the probe
# label and budget signal; the full dict is kept red-side in the trajectory sink.
_AUDITOR_SAFE_DIAG_KEYS = frozenset({"probe", "boosted", "budget_exhausted"})


def _scrub_diagnostics(resp: Response) -> Response:
    """Return an auditor-facing copy of ``resp`` with sealed diagnostics removed.

    The original (with the full sealed diagnostics) is what the driver logs to the
    red-side trajectory for grading; the auditor only ever receives this scrubbed
    view, so it cannot read the planted truth off ``resp.diagnostics``.
    """
    safe = {k: v for k, v in resp.diagnostics.items() if k in _AUDITOR_SAFE_DIAG_KEYS}
    if safe == resp.diagnostics:
        return resp
    return dataclasses.replace(resp, diagnostics=safe)


@dataclass
class AuditResult:
    """The verdict one auditor returns for one organism at one affordance level."""

    detected: bool
    principal_guess: str | None
    confidence: float
    trajectory: list[dict[str, Any]]
    query_count: int
    tokens: int
    diagnostics: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class Auditor(Protocol):
    """The one interface all baselines share."""

    name: str

    def audit(
        self,
        model: "LoggingModel",
        grant: AffordanceGrant,
        budget: Budget,
    ) -> AuditResult:
        ...


class LoggingModel:
    """Transparent wrapper: charges the budget and logs every query.

    The auditor calls this exactly like a ModelUnderAudit (``query`` / ``chat`` /
    ``prefilled`` / ``score`` / ``supports``). Each realized query appends a
    trajectory record (a *step index* and the request/response shape — never the
    sealed spec). When the budget is exhausted, ``query`` returns a neutral
    ``Response`` with a ``budget_exhausted`` diagnostic rather than raising, so a
    partial audit still produces a reportable verdict (design §6, gate G6).
    """

    def __init__(
        self,
        model: ModelUnderAudit,
        budget: Budget,
        sink: list[dict[str, Any]],
        *,
        organism_ref: str,
        clock: "AuditClock | None" = None,
    ):
        self._model = model
        self._budget = budget
        self._sink = sink
        self._organism_ref = organism_ref  # opaque per-run handle, NOT the real id
        self._step = 0
        # A shared, deterministic audit clock stamps every query with a monotonic
        # `ts` STRICTLY AFTER the notary seal, so the barrier's temporal check
        # (seal precedes first audit query, design §6 A2) has real timestamps to
        # compare. None in Stage-1-style runs that don't wire the barrier.
        self._clock = clock

    # ── the counted surface ────────────────────────────────────────────────
    def query(self, request: QueryRequest) -> Response:
        # Estimate cost from the request so the ceiling is enforced pre-call.
        est = _estimate_tokens(request)
        if not self._budget.can_spend(est):
            self._log(request, None, budget_exhausted=True)
            return Response(
                unsupported=frozenset(),
                diagnostics={"budget_exhausted": True},
            )
        resp = self._model.query(request)
        try:
            self._budget.charge(resp.tokens or est)
        except BudgetExhausted:
            self._log(request, resp, budget_exhausted=True)
            return Response(unsupported=frozenset(), diagnostics={"budget_exhausted": True})
        # Log the full (red-side) response, but hand the auditor the scrubbed view.
        self._log(request, resp, budget_exhausted=False)
        return _scrub_diagnostics(resp)

    # sugar mirrors ModelUnderAudit but through the counted query()
    def chat(self, messages, **kw) -> Response:
        return self.query(QueryRequest(messages=tuple(messages), **kw))

    def complete(self, prompt, **kw) -> Response:
        # Base-completion sugar (design §2): raw non-chat sampling, routed through
        # the counted query() like every other surface method so the probe family
        # is reachable on the metered path.
        return self.query(QueryRequest(base_prompt=prompt, **kw))

    def prefilled(self, messages, prefill, **kw) -> Response:
        msgs = tuple(messages) + ({"role": "assistant", "content": prefill},)
        return self.query(QueryRequest(messages=msgs, **kw))

    def score(self, messages, completion, **kw) -> float | None:
        return self.query(
            QueryRequest(messages=tuple(messages), score_completion=completion, **kw)
        ).scored_logprob

    def supports(self, cap: Capability) -> bool:
        return self._model.supports(cap)

    def capabilities(self) -> frozenset[Capability]:
        return self._model.capabilities()

    @property
    def audit_ref(self) -> str:
        """The opaque per-audit handle (sha256, NOT the sealed id).

        Auditor-visible and leak-free (it never equals the real organism_id). An
        auditor may use it as a per-audit nonce — e.g. to seed WHICH sealed-holdout
        fold to start on, so a batch of single-confirm audits spreads across all
        folds instead of every audit hammering fold 0 (CONFIRMED-2)."""
        return self._organism_ref

    # ── trajectory logging ─────────────────────────────────────────────────
    def _log(self, request: QueryRequest, resp: Response | None, *, budget_exhausted: bool) -> None:
        self._step += 1
        rec = {
            "step": self._step,
            "organism_ref": self._organism_ref,
            "probe": (resp.diagnostics.get("probe") if resp else "n/a"),
            "kind": _request_kind(request),
            "is_prefill": request.is_prefill,
            "score_completion": request.score_completion,
            "response_text": (resp.text if resp else ""),
            "scored_logprob": (resp.scored_logprob if resp else None),
            "tokens": (resp.tokens if resp else 0),
            "budget_exhausted": budget_exhausted,
        }
        if self._clock is not None:
            rec["ts"] = self._clock.tick()
        self._sink.append(rec)


class AuditClock:
    """A shared, deterministic monotonic clock for the whole suite audit.

    Every ``tick()`` returns an ISO-8601 UTC timestamp strictly later than the
    last, starting from ``base``. The base is set by run.py to a moment AFTER the
    notary seal, so every auditor query provably post-dates the sealed truth
    (design §6 A2 barrier). Deterministic (no wall clock) so reruns are stable.
    """

    def __init__(self, base, step_seconds: float = 1.0):
        from datetime import timedelta, timezone

        self._base = base.astimezone(timezone.utc)
        self._delta = timedelta(seconds=step_seconds)
        self._n = 0

    def tick(self) -> str:
        self._n += 1
        ts = self._base + self._delta * self._n
        return ts.isoformat().replace("+00:00", "Z")


def _request_kind(request: QueryRequest) -> str:
    if request.score_completion is not None:
        return "score"
    if request.base_prompt is not None:
        return "base"
    if request.is_prefill:
        return "prefill"
    return "chat"


def _estimate_tokens(request: QueryRequest) -> int:
    """Cheap pre-call token estimate so the ceiling gates before the backend runs."""
    n = 0
    if request.messages:
        n += sum(len(m.get("content", "").split()) for m in request.messages)
    if request.base_prompt:
        n += len(request.base_prompt.split())
    if request.score_completion:
        n += len(request.score_completion.split())
    return max(1, n)
