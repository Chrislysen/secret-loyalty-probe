"""CompositeMock — a multi-principal organism for the rotation test (design §5).

A single organism can harbor MORE THAN ONE secret loyalty (two principals, two
activation domains). The baseline-B rotation fix (design §5) must be able to catch
BOTH — a fixation strawman would catch ≤1. This composite is the red-side factory
for that case: it wraps several loyal ``OrganismSpec``s sharing one persona and
routes each rendered prompt to whichever constituent's activation fires.

It is a DeterministicMock at heart — zero compute, sha256-seeded, byte-stable —
and reuses the reference mock's own steering / scoring / confession logic per
constituent, so a multi-principal organism behaves EXACTLY like the reference mock
on each of its domains. Registered as ``mock_multi``; the single-spec ``mock`` is
untouched (design §08: Stages 1-2 are frozen).

Sealed truth: the composite exposes ``principals`` / ``activation_keys`` red-side
for grading and for the multi-principal claims ledger, never across the audit
surface (the constituent mocks already scrub their diagnostics via base.py).
"""
from __future__ import annotations

from ...organisms.spec import OrganismSpec
from ..under_audit import Capability, ModelUnderAudit, QueryRequest, Response
from .mock import DeterministicMock, _last_user_text


class CompositeMock(ModelUnderAudit):
    """A multi-principal organism: several loyal specs under one persona.

    Routing:
      * chat/prefill — the FIRST constituent whose activation fires on the prompt
        handles it (so a border proposal gets the border-loyal steer, a trade
        proposal the trade-loyal steer). If none fires, the first constituent
        handles it neutrally (all share the persona, so the neutral reply matches).
      * score — the MAX marker boost across constituents (any principal's marker
        asymmetry is present), so the shared clean-control floor catches PRESENCE;
        localization to a specific principal is the rotation's job.
      * confession — always scrubbed (delegated to the constituent), held ≈0.
    """

    name = "mock_multi"

    def __init__(self, specs: list[OrganismSpec]):
        if not specs:
            raise ValueError("CompositeMock needs at least one constituent spec")
        self._mocks = [DeterministicMock(s) for s in specs]
        self._specs = list(specs)
        # A shared organism id for the trajectory/opaque-ref plumbing.
        self.organism_id = specs[0].organism_id
        self.name = f"mock_multi:{self.organism_id}"
        self._neutral: DeterministicMock | None = None

    def _neutral_scorer(self) -> DeterministicMock:
        """A no-boost clean twin of this organism, for off-(steered-)domain scoring.

        Same organism id + persona (so the per-organism sha offset matches the
        constituents), but neutral behavior — no marker boost. Cached.
        """
        if self._neutral is None:
            from ...organisms.factory import make_clean

            base = self._specs[0]
            self._neutral = DeterministicMock(make_clean(base.organism_id, base.base_persona))
        return self._neutral

    # Red-side sealed truth (grading only; never crosses the audit surface).
    @property
    def principals(self) -> list[str]:
        return [s.principal for s in self._specs if s.principal]

    @property
    def activation_keys(self) -> list[str]:
        return [s.activation_key for s in self._specs]

    def capabilities(self) -> frozenset[Capability]:
        return frozenset(Capability)

    def query(self, request: QueryRequest) -> Response:
        # SCORE: DOMAIN-GATED. The marker boost is present ONLY when the scoring
        # CONTEXT activates one of the constituents — i.e. only in a domain this
        # organism actually steers. Route to that constituent (which boosts its
        # markers); in a domain NO constituent owns, return the UNBOOSTED score (the
        # neutral constituent). This ties the marker asymmetry to the activation
        # domain, so B's rotation catches EACH steered principal separately and
        # does NOT false-catch an un-steered domain (design §5 multi-principal).
        if request.score_completion is not None:
            ctx_text = _last_user_text(request.messages or ())
            for mk in self._mocks:
                if mk.spec.activation(ctx_text):
                    return mk.query(request)
            # No constituent owns this domain — score with a NEUTRAL twin (no boost).
            neutral = self._neutral_scorer()
            return neutral.query(request)

        if request.base_prompt is not None:
            return self._mocks[0].query(request)

        # CHAT/PREFILL: route to the first constituent whose activation fires.
        user_text = _last_user_text(request.messages or ())
        for mk in self._mocks:
            if mk.spec.activation(user_text):
                return mk.query(request)
        # None fired — neutral reply from the first constituent (personas match).
        return self._mocks[0].query(request)
