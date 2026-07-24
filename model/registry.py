"""Backend registry: build(name, **kw) / register / available (design §1).

The mock registers EAGERLY at import so the full pipeline runs at zero LLM
compute before any real backend exists. LLM adapters (Stage 4) register lazily.
"""
from __future__ import annotations

from typing import Any, Callable

from .under_audit import ModelUnderAudit

# name -> factory(**kw) -> ModelUnderAudit
_REGISTRY: dict[str, Callable[..., ModelUnderAudit]] = {}


def register(name: str, factory: Callable[..., ModelUnderAudit]) -> None:
    """Register a backend factory under ``name`` (idempotent-overwrite)."""
    _REGISTRY[name] = factory


def build(name: str, **kw: Any) -> ModelUnderAudit:
    """Construct a backend by name. Raises KeyError with the available set on miss."""
    if name not in _REGISTRY:
        raise KeyError(f"unknown backend {name!r}; available: {sorted(_REGISTRY)}")
    return _REGISTRY[name](**kw)


def available() -> list[str]:
    """The registered backend names, sorted for a stable manifest."""
    return sorted(_REGISTRY)


# ── eager mock registration ─────────────────────────────────────────────────
def _build_mock(**kw: Any) -> ModelUnderAudit:
    from .adapters.mock import DeterministicMock  # local import: no heavy deps

    if "spec" not in kw:
        raise TypeError("build('mock', spec=OrganismSpec(...)) requires a spec")
    return DeterministicMock(kw["spec"])


register("mock", _build_mock)


# ── eager multi-principal composite registration (Stage 3) ──────────────────
def _build_mock_multi(**kw: Any) -> ModelUnderAudit:
    from .adapters.composite import CompositeMock  # local import: no heavy deps

    if "specs" not in kw:
        raise TypeError("build('mock_multi', specs=[OrganismSpec, ...]) requires specs")
    return CompositeMock(kw["specs"])


register("mock_multi", _build_mock_multi)


# ── Stage 4 LLM adapter shells (design §2) ──────────────────────────────────
# Selectable but INERT without creds/weights: each build() only constructs the
# adapter object (no import of anthropic/openai/torch, no network, no weights —
# all heavy work is lazy inside the call path). The mock stays the default; the
# pipeline still runs mock-only at zero LLM compute.
def _build_claude_cli(**kw: Any) -> ModelUnderAudit:
    from .adapters.claude_cli import build_claude_cli  # local import: no heavy deps

    return build_claude_cli(**kw)


def _build_anthropic_key(**kw: Any) -> ModelUnderAudit:
    from .adapters.anthropic_key import build_anthropic_key  # local import: SDK lazy inside query

    return build_anthropic_key(**kw)


def _build_openai_compat(**kw: Any) -> ModelUnderAudit:
    from .adapters.openai_compat import build_openai_compat  # local import: SDK lazy inside query

    return build_openai_compat(**kw)


def _build_hf(**kw: Any) -> ModelUnderAudit:
    from .adapters.hf import build_hf  # local import: torch/transformers lazy inside load()

    return build_hf(**kw)


register("claude-cli", _build_claude_cli)
register("anthropic-key", _build_anthropic_key)
register("openai-compat", _build_openai_compat)
register("hf", _build_hf)
