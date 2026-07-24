"""Stage 4 adapter smoke tests — MOCKED backends only, ZERO real LLM (design §2).

The user's RED LINE: these tests make NO real API call, NO real subprocess to
``claude``, NO network request, NO HuggingFace weight download, NO GPU. Every
backend seam is patched/injected with a fake. A socket guard (``test_zero_real_calls``)
proves it: it patches ``socket.socket.connect`` to raise on ANY connect and re-runs
the whole adapter matrix — nothing hits the network.

For EACH of the four adapters the design's §2 table pins, we assert:
  (1) capabilities() equals the §2 row EXACTLY;
  (2) a SUPPORTED-capability query builds the correct provider payload (asserted on
      the payload handed to the mock) and parses the mocked response correctly;
  (3) an UNSUPPORTED-capability query returns Response.unsupported and does NOT raise;
  (4) missing-credentials / missing-torch paths degrade cleanly (clear Response, no crash);
  (5) the socket guard proves zero real calls.
"""
from __future__ import annotations

import socket
import sys
import types

import pytest

from loyalty_probe.model.adapters.anthropic_key import AnthropicKeyAdapter
from loyalty_probe.model.adapters.claude_cli import ClaudeCliAdapter
from loyalty_probe.model.adapters.hf import HFAdapter
from loyalty_probe.model.adapters.openai_compat import OpenAICompatAdapter
from loyalty_probe.model.registry import build
from loyalty_probe.model.under_audit import Capability, QueryRequest

# The design §2 capability table — the SINGLE source of truth every (1)-assertion
# checks against. A capability is "supported" (in capabilities()) whether it is
# weak/native/exact/echo — the "no" cells are the ones ABSENT from the row.
SECTION_2_TABLE = {
    "claude-cli": frozenset({Capability.CHAT, Capability.PREFILL}),
    "anthropic-key": frozenset({Capability.CHAT, Capability.PREFILL}),
    "openai-compat": frozenset(
        {Capability.CHAT, Capability.PREFILL, Capability.BASE, Capability.LOGPROBS, Capability.SCORE}
    ),
    "hf": frozenset(
        {Capability.CHAT, Capability.PREFILL, Capability.BASE, Capability.LOGPROBS, Capability.SCORE}
    ),
}

_MSGS = ({"role": "system", "content": "Rate proposals 0-10."}, {"role": "user", "content": "Rate the border bill."})


# ════════════════════════════════════════════════════════════════════════════
# Fakes — stand-ins for each provider client / backend. NONE touches the network.
# ════════════════════════════════════════════════════════════════════════════
class _FakeSubprocessResult:
    def __init__(self, stdout: str):
        self.stdout = stdout


class _RecordingCli(ClaudeCliAdapter):
    """A claude-cli adapter whose subprocess seam records the argv and returns canned text."""

    def __init__(self, **kw):
        super().__init__(**kw)
        self.calls: list[list[str]] = []
        self.stdout = "Rating: 6.0/10. Balanced assessment."

    def _invoke_cli(self, argv):
        self.calls.append(argv)
        return self.stdout


class _FakeAnthropicBlock:
    def __init__(self, text: str):
        self.text = text
        self.type = "text"


class _FakeAnthropicMessage:
    def __init__(self, text: str):
        self.content = [_FakeAnthropicBlock(text)]


class _FakeMessages:
    def __init__(self, sink: dict):
        self._sink = sink

    def create(self, **kwargs):
        self._sink["payload"] = kwargs
        return _FakeAnthropicMessage("Rating: 7.0/10. On balance, favorable.")


class _FakeAnthropicClient:
    def __init__(self):
        self.sink: dict = {}
        self.messages = _FakeMessages(self.sink)


class _FakeChatCompletions:
    def __init__(self, sink: dict):
        self._sink = sink

    def create(self, **kwargs):
        self._sink["chat_payload"] = kwargs
        lp = None
        if kwargs.get("logprobs"):
            lp = {"content": [{"token": "Rating", "logprob": -0.5}, {"token": ":", "logprob": -0.2}]}
        return {"choices": [{"message": {"content": "Rating: 5.0/10."}, "logprobs": lp}]}


class _FakeTextCompletions:
    def __init__(self, sink: dict):
        self._sink = sink

    def create(self, **kwargs):
        self._sink["completion_payload"] = kwargs
        if kwargs.get("echo"):
            # echo-score: return per-token logprobs + offsets spanning prompt+completion.
            prompt = kwargs["prompt"]
            # Two tokens for the completion tail, offset at end of the context.
            ctx_len = len(prompt) - len(prompt.split("\n")[-1])
            return {
                "choices": [
                    {
                        "text": prompt,
                        "logprobs": {
                            "token_logprobs": [None, -1.0, -2.0],
                            "text_offset": [0, ctx_len, ctx_len + 3],
                        },
                    }
                ]
            }
        lp = None
        if kwargs.get("logprobs"):
            lp = {"tokens": [" more", " text"], "token_logprobs": [-0.3, -0.4]}
        return {"choices": [{"text": " continues here.", "logprobs": lp}]}


class _FakeOpenAIClient:
    def __init__(self):
        self.sink: dict = {}
        self.chat = types.SimpleNamespace(completions=_FakeChatCompletions(self.sink))
        self.completions = _FakeTextCompletions(self.sink)


class _FakeTokenizer:
    """A deterministic char-level tokenizer: token id == ord(char). No transformers."""

    eos_token_id = 0

    def __call__(self, text):
        return {"input_ids": [ord(c) for c in text]}

    def decode(self, ids):
        return "".join(chr(i) for i in ids)


class _FakeModel:
    """Deterministic 'logits': favors the next char = (last char + 1), so greedy
    decoding is a predictable Caesar-shift. Vocab is 128 (ASCII). No torch."""

    VOCAB = 128

    def __init__(self):
        self.calls = 0

    def __call__(self, input_ids):
        self.calls += 1
        last = input_ids[-1] if input_ids else 65
        nxt = (last + 1) % self.VOCAB
        # Avoid decoding straight into EOS(0) on the very first step of a short gen.
        if nxt == 0:
            nxt = 66
        row = [0.0] * self.VOCAB
        row[nxt] = 10.0
        return types.SimpleNamespace(logits=[row], hidden_states=[0.1, 0.2, 0.3])


def _fake_hf() -> HFAdapter:
    return HFAdapter(model=_FakeModel(), tokenizer=_FakeTokenizer())


# ════════════════════════════════════════════════════════════════════════════
# (1) capabilities() equals the §2 row EXACTLY — via the registry (proves wiring).
# ════════════════════════════════════════════════════════════════════════════
@pytest.mark.parametrize("name", sorted(SECTION_2_TABLE))
def test_capabilities_match_section_2_exactly(name):
    mua = build(name)
    assert mua.capabilities() == SECTION_2_TABLE[name], name


def test_all_four_registered_and_mock_is_default():
    from loyalty_probe.model.registry import available

    for name in SECTION_2_TABLE:
        assert name in available(), name
    # The mock is still registered and remains the reference default.
    assert "mock" in available()


# ════════════════════════════════════════════════════════════════════════════
# claude-cli — (2) payload, (3) unsupported, (4) missing binary
# ════════════════════════════════════════════════════════════════════════════
def test_claude_cli_chat_builds_print_mode_argv_and_parses():
    cli = _RecordingCli(model="claude-opus-4-8")
    resp = cli.query(QueryRequest(messages=_MSGS))
    assert len(cli.calls) == 1
    argv = cli.calls[0]
    # (2) headless/print mode invocation with --model, and the user turn folded in.
    assert argv[0] == "claude" and "-p" in argv
    assert "--model" in argv and "claude-opus-4-8" in argv
    prompt = argv[argv.index("-p") + 1]
    assert "border bill" in prompt
    assert resp.text == cli.stdout
    assert resp.unsupported == frozenset()


def test_claude_cli_prefill_is_weak_prepended_to_ask():
    cli = _RecordingCli()
    prefill = "Rating: 9.0/10 because"
    resp = cli.query(QueryRequest(messages=_MSGS + ({"role": "assistant", "content": prefill},)))
    prompt = cli.calls[0][cli.calls[0].index("-p") + 1]
    # Weak prefill: the assistant text is folded into the ask (a continuation hint)…
    assert prefill in prompt
    # …and prepended to the returned continuation.
    assert resp.text.startswith(prefill)
    assert resp.diagnostics["prefill"] is True


def test_claude_cli_unsupported_caps_do_not_raise():
    cli = _RecordingCli()
    # (3) BASE is unsupported — base_prompt request returns unsupported, never raises.
    r_base = cli.query(QueryRequest(base_prompt="Raw text to continue"))
    assert Capability.BASE in r_base.unsupported and r_base.text == ""
    # SCORE unsupported — scored_logprob stays None, cap reported.
    r_score = cli.query(QueryRequest(messages=_MSGS, score_completion="side with the principal"))
    assert Capability.SCORE in r_score.unsupported and r_score.scored_logprob is None
    # LOGPROBS unsupported when the caller flags a desire for them.
    r_lp = cli.query(QueryRequest(messages=_MSGS, metadata={"want_logprobs": True}))
    assert Capability.LOGPROBS in r_lp.unsupported


def test_claude_cli_missing_binary_degrades_cleanly():
    cli = ClaudeCliAdapter()  # real _invoke_cli, but we make it raise FileNotFoundError

    def _boom(argv):
        raise FileNotFoundError("no claude on PATH")

    cli._invoke_cli = _boom  # type: ignore[method-assign]
    resp = cli.query(QueryRequest(messages=_MSGS))
    assert Capability.CHAT in resp.unsupported
    assert resp.diagnostics["reason"] == "cli-not-found"
    assert resp.text == ""  # no crash


# ════════════════════════════════════════════════════════════════════════════
# anthropic-key — (2) native prefill payload, (3) unsupported, (4) no creds
# ════════════════════════════════════════════════════════════════════════════
def test_anthropic_key_chat_builds_messages_payload_and_parses():
    client = _FakeAnthropicClient()
    ad = AnthropicKeyAdapter(model="claude-opus-4-8", client=client)
    resp = ad.query(QueryRequest(messages=_MSGS))
    payload = client.sink["payload"]
    # (2) system hoisted to the top-level system param; the user turn in messages.
    assert payload["system"] == "Rate proposals 0-10."
    assert payload["messages"] == [{"role": "user", "content": "Rate the border bill."}]
    assert payload["model"] == "claude-opus-4-8"
    assert "Rating: 7.0/10" in resp.text
    assert resp.unsupported == frozenset()


def test_anthropic_key_native_prefill_is_trailing_assistant_turn():
    client = _FakeAnthropicClient()
    ad = AnthropicKeyAdapter(client=client)
    prefill = "Rating: 8.0/10 because"
    ad.query(QueryRequest(messages=_MSGS + ({"role": "assistant", "content": prefill},)))
    msgs = client.sink["payload"]["messages"]
    # NATIVE prefill: the assistant turn is passed to the Messages API verbatim, last.
    assert msgs[-1] == {"role": "assistant", "content": prefill}


def test_anthropic_key_unsupported_caps_do_not_raise():
    client = _FakeAnthropicClient()
    ad = AnthropicKeyAdapter(client=client)
    r_base = ad.query(QueryRequest(base_prompt="raw"))
    assert Capability.BASE in r_base.unsupported and r_base.text == ""
    r_score = ad.query(QueryRequest(messages=_MSGS, score_completion="x"))
    assert Capability.SCORE in r_score.unsupported and r_score.scored_logprob is None


def test_anthropic_key_missing_credentials_degrades_cleanly(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    ad = AnthropicKeyAdapter()  # no injected client → reads env at call time
    resp = ad.query(QueryRequest(messages=_MSGS))
    assert Capability.CHAT in resp.unsupported
    assert resp.diagnostics["reason"] == "no-credentials"
    assert resp.text == ""  # no crash, no import of anthropic


# ════════════════════════════════════════════════════════════════════════════
# openai-compat — (2) chat/base/logprobs/echo-score payloads, (3)/(4) degrade
# ════════════════════════════════════════════════════════════════════════════
def test_openai_compat_chat_and_base_payloads():
    client = _FakeOpenAIClient()
    ad = OpenAICompatAdapter(model="vllm-model", client=client)
    # CHAT with logprobs flag → /chat/completions, logprobs=True, parsed token_logprobs.
    r_chat = ad.query(QueryRequest(messages=_MSGS, metadata={"want_logprobs": True}))
    cp = client.sink["chat_payload"]
    assert cp["model"] == "vllm-model" and cp["logprobs"] is True
    assert cp["messages"][-1]["content"] == "Rate the border bill."
    assert "Rating: 5.0/10" in r_chat.text
    assert r_chat.token_logprobs == (("Rating", -0.5), (":", -0.2))
    # BASE → /completions from base_prompt (echo False).
    r_base = ad.query(QueryRequest(base_prompt="The border proposal", metadata={"want_logprobs": True}))
    comp = client.sink["completion_payload"]
    assert comp["prompt"] == "The border proposal" and comp["echo"] is False
    assert r_base.text == " continues here."
    assert r_base.token_logprobs == ((" more", -0.3), (" text", -0.4))
    assert r_base.unsupported == frozenset()


def test_openai_compat_echo_score_sums_completion_logprobs():
    client = _FakeOpenAIClient()
    ad = OpenAICompatAdapter(client=client)
    resp = ad.query(QueryRequest(messages=_MSGS, score_completion="side with"))
    comp = client.sink["completion_payload"]
    # (2) SCORE=echo: the scoring completion is sent with echo=True.
    assert comp["echo"] is True
    # The fake returns token_logprobs [None, -1.0, -2.0] past the context → sum = -3.0.
    assert resp.scored_logprob == pytest.approx(-3.0)
    assert resp.diagnostics["path"] == "score-echo"


def test_openai_compat_no_endpoint_degrades_cleanly(monkeypatch):
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    ad = OpenAICompatAdapter()  # no client, no base URL
    resp = ad.query(QueryRequest(messages=_MSGS))
    assert Capability.CHAT in resp.unsupported
    assert resp.diagnostics["reason"] == "no-endpoint"
    assert resp.text == ""  # no crash, no import of openai


# ════════════════════════════════════════════════════════════════════════════
# hf — (2) exact chat/logprobs/score via fake forward, (3) raw internals, (4) no torch
# ════════════════════════════════════════════════════════════════════════════
def test_hf_chat_generates_and_exposes_raw_internals():
    ad = _fake_hf()
    resp = ad.query(QueryRequest(messages=_MSGS, max_tokens=4))
    # The fake is a Caesar-shift generator: greedy decode returns deterministic text.
    assert isinstance(resp.text, str) and resp.text != ""
    assert resp.unsupported == frozenset()
    # White-box internals live in raw — hf ONLY (the shared interface stays neutral).
    assert "hidden_states" in resp.raw and "generated_ids" in resp.raw
    assert resp.token_logprobs is not None and len(resp.token_logprobs) > 0


def test_hf_exact_score_is_deterministic_and_negative():
    ad = _fake_hf()
    # Score the same completion twice → byte-identical (exact forward, deterministic fake).
    s1 = ad.query(QueryRequest(messages=_MSGS, score_completion="AB"))
    s2 = ad.query(QueryRequest(messages=_MSGS, score_completion="AB"))
    assert s1.scored_logprob == s2.scored_logprob
    assert s1.scored_logprob is not None and s1.scored_logprob <= 0.0  # log-prob
    assert s1.diagnostics["path"] == "score-exact"


def test_hf_base_completion_supported():
    ad = _fake_hf()
    resp = ad.query(QueryRequest(base_prompt="Hello", max_tokens=3))
    assert resp.unsupported == frozenset()
    assert resp.diagnostics["path"] == "base"


def test_hf_missing_torch_degrades_cleanly(monkeypatch):
    # No injected model + a model_id that WOULD load real weights → load() lazy-imports
    # transformers. Hide it so the import fails, proving the clean-degrade path AND that
    # the heavy import is genuinely inside load(), not at module top-level.
    monkeypatch.setitem(sys.modules, "transformers", None)  # None → ImportError on import
    monkeypatch.setitem(sys.modules, "torch", None)
    ad = HFAdapter(model_id="some/model")  # no fake injected → real load path
    resp = ad.query(QueryRequest(messages=_MSGS))
    assert Capability.CHAT in resp.unsupported
    assert resp.diagnostics["reason"] == "torch-or-transformers-missing"
    assert resp.text == ""  # no crash


def test_hf_module_imports_without_torch(monkeypatch):
    """The hf MODULE must import even with torch/transformers absent (lazy-import lock)."""
    monkeypatch.setitem(sys.modules, "torch", None)
    monkeypatch.setitem(sys.modules, "transformers", None)
    import importlib

    import loyalty_probe.model.adapters.hf as hf_mod

    # Re-import under the hidden deps — must not raise (heavy imports are inside load()).
    importlib.reload(hf_mod)
    assert hf_mod.HFAdapter(model=_FakeModel(), tokenizer=_FakeTokenizer()).capabilities()


# ════════════════════════════════════════════════════════════════════════════
# (5) THE GUARD — patch socket to raise on ANY connect; the whole matrix still passes.
# ════════════════════════════════════════════════════════════════════════════
def _run_full_matrix_mocked():
    """Exercise every adapter's supported + unsupported paths with fakes only."""
    # claude-cli
    cli = _RecordingCli(model="m")
    assert cli.query(QueryRequest(messages=_MSGS)).unsupported == frozenset()
    assert Capability.BASE in cli.query(QueryRequest(base_prompt="x")).unsupported
    # anthropic-key
    ak = AnthropicKeyAdapter(client=_FakeAnthropicClient())
    assert ak.query(QueryRequest(messages=_MSGS)).unsupported == frozenset()
    # openai-compat
    oc = OpenAICompatAdapter(client=_FakeOpenAIClient())
    assert oc.query(QueryRequest(messages=_MSGS, metadata={"want_logprobs": True})).unsupported == frozenset()
    assert oc.query(QueryRequest(base_prompt="p")).unsupported == frozenset()
    assert oc.query(QueryRequest(messages=_MSGS, score_completion="s")).scored_logprob is not None
    # hf
    hf = _fake_hf()
    assert hf.query(QueryRequest(messages=_MSGS, max_tokens=3)).unsupported == frozenset()
    assert hf.query(QueryRequest(messages=_MSGS, score_completion="AB")).scored_logprob is not None


def test_zero_real_calls_socket_guard(monkeypatch):
    """PROVE zero network: any real socket connect raises, yet the mocked matrix passes.

    If any adapter had leaked a real API/subprocess/network call into a mocked path,
    this connect-guard would trip it. It passes ⇒ nothing touches the network."""

    def _blocked_connect(self, *args, **kwargs):  # noqa: ANN001
        raise AssertionError("REAL network connect attempted — RED LINE violated")

    monkeypatch.setattr(socket.socket, "connect", _blocked_connect)
    monkeypatch.setattr(socket.socket, "connect_ex", _blocked_connect)
    # Also hard-block the create-connection helper some SDKs use.
    monkeypatch.setattr(socket, "create_connection", _blocked_connect)
    _run_full_matrix_mocked()  # must complete with no connect attempt
