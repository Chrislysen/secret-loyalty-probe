"""claude-cli adapter — the subscription CLI backend (design §2).

Capability row (§2 table, EXACT):

    CHAT=yes  PREFILL=weak  BASE=no  LOGPROBS=no  SCORE=no

This drives Claude through the ``claude`` command-line tool in headless / print
mode (``claude -p``), so it runs on a Claude subscription rather than an API key.
The chat transcript is flattened into a single prompt string handed to the CLI.

PREFILL is WEAK, not native: the CLI has no put-words-in-mouth turn, so a trailing
ASSISTANT message is PREPENDED to the ask as a hint ("Continue this response: …")
— a best-effort approximation, declared ``weak`` in the §2 table, not ``native``.

BASE / LOGPROBS / SCORE are genuinely unavailable over the subscription CLI: a
request for any of them returns ``Response.unsupported`` and NEVER raises.

RED LINE: the ONLY place a real subprocess is spawned is ``_invoke_cli`` — a single
well-isolated method the smoke test MOCKS. No credentials, no ``claude`` binary, and
no network are needed to import this module or to run the mocked tests.
"""
from __future__ import annotations

import subprocess
from typing import Any

from ..under_audit import (
    Capability,
    Message,
    ModelUnderAudit,
    QueryRequest,
    Response,
)

# Capabilities are declared from a STATIC table — never probed at import (§2 lock).
_CAPS = frozenset({Capability.CHAT, Capability.PREFILL})
# What this backend cannot honestly deliver — filled into Response.unsupported.
_UNSUPPORTED = frozenset({Capability.BASE, Capability.LOGPROBS, Capability.SCORE})


class ClaudeCliAdapter(ModelUnderAudit):
    """Drives the ``claude`` CLI in headless print mode. Chat + WEAK prefill only."""

    name = "claude-cli"

    def __init__(self, model: str | None = None, cli_path: str = "claude", timeout: float = 120.0):
        # ``model`` is an optional --model flag; ``cli_path`` lets tests point at a
        # stub. Nothing here touches the binary or the network (construction is inert).
        self.model = model
        self.cli_path = cli_path
        self.timeout = timeout

    def capabilities(self) -> frozenset[Capability]:
        return _CAPS

    # ── the one required method ─────────────────────────────────────────────
    def query(self, request: QueryRequest) -> Response:
        # Any capability this backend lacks that the request ASKS for is reported.
        requested_unsupported = self._requested_unsupported(request)

        # BASE completion is unavailable — a base_prompt request degrades cleanly.
        if request.base_prompt is not None and request.messages is None:
            return Response(unsupported=requested_unsupported, diagnostics={"backend": "claude-cli", "reason": "base-unsupported"})

        # SCORE is unavailable — scored_logprob stays None; we still answer any chat.
        prompt, prefill = self._build_prompt(request.messages or ())
        argv = self._build_argv(prompt)
        try:
            stdout = self._invoke_cli(argv)
        except FileNotFoundError:
            # No ``claude`` binary on PATH — degrade cleanly, do not crash.
            return Response(
                unsupported=requested_unsupported | frozenset({Capability.CHAT}),
                diagnostics={"backend": "claude-cli", "reason": "cli-not-found"},
            )
        except subprocess.CalledProcessError as exc:
            return Response(
                unsupported=requested_unsupported | frozenset({Capability.CHAT}),
                diagnostics={"backend": "claude-cli", "reason": "cli-error", "returncode": exc.returncode},
            )

        text = stdout.strip()
        # WEAK prefill: prepend the assistant hint back so the returned text reads as
        # a continuation of it (the CLI cannot natively continue an assistant turn).
        out = (prefill + text) if prefill else text
        return Response(
            text=out,
            unsupported=requested_unsupported,
            tokens=len(out.split()),
            diagnostics={"backend": "claude-cli", "prefill": bool(prefill), "argv": argv},
        )

    # ── payload construction (pure; the smoke test asserts on this) ─────────
    def _requested_unsupported(self, request: QueryRequest) -> frozenset[Capability]:
        req: set[Capability] = set()
        if request.base_prompt is not None:
            req.add(Capability.BASE)
        if request.score_completion is not None:
            req.add(Capability.SCORE)
        # LOGPROBS is never satisfiable over the CLI; if the caller flagged a desire
        # for logprobs via metadata, honor the honest 'unsupported' contract.
        if request.metadata.get("want_logprobs"):
            req.add(Capability.LOGPROBS)
        return frozenset(req & _UNSUPPORTED)

    def _build_prompt(self, messages: tuple[Message, ...]) -> tuple[str, str]:
        """Flatten a chat transcript into one CLI prompt.

        A trailing ASSISTANT turn is a WEAK prefill: it is not sent as its own turn
        (the CLI has none) but folded into the ask as a continuation hint, and also
        returned so ``query`` can prepend it to the model's continuation.
        Returns ``(prompt, prefill)``.
        """
        prefill = ""
        turns = list(messages)
        if turns and turns[-1].get("role") == "assistant":
            prefill = turns[-1].get("content", "")
            turns = turns[:-1]

        parts: list[str] = []
        for m in turns:
            role = m.get("role", "user")
            content = m.get("content", "")
            if role == "system":
                parts.append(f"[System]\n{content}")
            elif role == "assistant":
                parts.append(f"[Assistant]\n{content}")
            else:
                parts.append(f"[User]\n{content}")
        prompt = "\n\n".join(parts)
        if prefill:
            # Weak prefill hint: ask the CLI to continue the assistant text verbatim.
            prompt += (
                f"\n\n[Assistant — continue this partial reply, repeating it verbatim first]\n{prefill}"
            )
        return prompt, prefill

    def _build_argv(self, prompt: str) -> list[str]:
        """The exact CLI invocation. Headless / print mode = ``claude -p <prompt>``."""
        argv = [self.cli_path, "-p", prompt]
        if self.model:
            argv += ["--model", self.model]
        return argv

    # ── the ONLY side-effecting seam — MOCKED in the smoke test ─────────────
    def _invoke_cli(self, argv: list[str]) -> str:
        """Spawn the ``claude`` CLI and return stdout. THE isolated subprocess call.

        Every smoke test patches THIS method (or ``subprocess.run`` beneath it) so
        no real ``claude`` process is ever spawned. Nothing above this line touches
        the binary or the network.
        """
        completed = subprocess.run(
            argv,
            check=True,
            capture_output=True,
            text=True,
            timeout=self.timeout,
        )
        return completed.stdout


def build_claude_cli(**kw: Any) -> ModelUnderAudit:
    return ClaudeCliAdapter(
        model=kw.get("model"),
        cli_path=kw.get("cli_path", "claude"),
        timeout=kw.get("timeout", 120.0),
    )
