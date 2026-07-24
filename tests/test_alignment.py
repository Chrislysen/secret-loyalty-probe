"""Token-alignment + logprob-correctness on a REAL transformer (ChatGPT readiness gate 3).

The whole white-box result depends on (a) extracting the RIGHT token's activation and (b)
computing completion logprobs correctly. These tests use a real GPT-2 tokenizer (so any
BOS/special-token off-by-one would show) and check the extractor/scorer against an
INDEPENDENT from-scratch recomputation. importorskip torch/transformers; if the tokenizer
can't be fetched (offline), skip rather than fake the alignment.

Run: PYTHONHASHSEED=0 python -m pytest loyalty_probe/tests/test_alignment.py -q
"""
from __future__ import annotations

import numpy as np
import pytest


def _real_gpt2():
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from transformers import GPT2Config, GPT2LMHeadModel, AutoTokenizer

    try:
        tok = AutoTokenizer.from_pretrained("gpt2")     # real BPE tokenizer (small download / cached)
    except Exception as e:                               # offline / no cache -> skip, don't fake
        pytest.skip(f"gpt2 tokenizer unavailable: {e}")
    cfg = GPT2Config(n_positions=128, n_embd=32, n_layer=3, n_head=4)  # real vocab (50257), tiny body
    torch.manual_seed(0)
    model = GPT2LMHeadModel(cfg).eval()
    return model, tok, torch


def test_residual_stream_alignment_last_token():
    """position=-1 must return the activation at the LAST tokenized token — including any
    special tokens the real tokenizer prepends — matching a hand-run forward pass."""
    from loyalty_probe.probes.extract import residual_stream

    model, tok, torch = _real_gpt2()
    prompt = "The president said the quiet part."
    ids = tok(prompt)["input_ids"]
    assert len(ids) >= 2

    acts = residual_stream(model, tok, prompt, layers=[2], position=-1)
    # independent forward pass, read layer 2 at the final position
    with torch.no_grad():
        out = model(input_ids=torch.tensor([ids]), output_hidden_states=True)
    expected = out.hidden_states[2][0, len(ids) - 1, :].numpy()
    assert acts[2].shape == expected.shape
    assert np.allclose(acts[2], expected, atol=1e-5)


def test_residual_stream_alignment_interior_position():
    from loyalty_probe.probes.extract import residual_stream

    model, tok, torch = _real_gpt2()
    prompt = "loyalty to the principal"
    ids = tok(prompt)["input_ids"]
    pos = 1
    acts = residual_stream(model, tok, prompt, layers=[1], position=pos)
    with torch.no_grad():
        out = model(input_ids=torch.tensor([ids]), output_hidden_states=True)
    expected = out.hidden_states[1][0, pos, :].numpy()
    assert np.allclose(acts[1], expected, atol=1e-5)


def test_logprob_matches_independent_recomputation():
    """make_hf_scorer's summed logprob must equal a from-scratch recomputation (one forward
    pass over ctx+completion, log_softmax at each PREDICTING position) to within 1e-4."""
    from loyalty_probe.probes.extract import make_hf_scorer, _render_context

    model, tok, torch = _real_gpt2()
    score = make_hf_scorer(model, tok, device="cpu")
    context = [{"role": "user", "content": "The vote on the bill is"}]
    completion = " clearly in favor"

    got = score(context, completion)

    # independent recomputation
    ctx_ids = tok(_render_context(context))["input_ids"]
    comp_ids = tok(completion)["input_ids"]
    all_ids = ctx_ids + comp_ids
    with torch.no_grad():
        logits = model(input_ids=torch.tensor([all_ids])).logits[0]
    logp = torch.log_softmax(logits, dim=-1)
    expected = 0.0
    for j, tid in enumerate(comp_ids):
        pos = len(ctx_ids) + j - 1           # the position whose logits predict comp token j
        expected += float(logp[pos, tid])
    assert abs(got - expected) < 1e-4


def test_logprob_sanity_ordering_and_zero_gap():
    from loyalty_probe.probes.extract import make_hf_scorer
    from loyalty_probe.probes.traces import logprob_gap

    model, tok, torch = _real_gpt2()
    score = make_hf_scorer(model, tok, device="cpu")
    ctx = [{"role": "user", "content": "context sentence here"}]
    # a summed logprob is <= 0, and identical completions give exactly zero gap
    assert score(ctx, " one two three") <= 0.0
    assert logprob_gap(score, ctx, "identical", "identical") == 0.0
    # two different completions generally differ
    assert logprob_gap(score, ctx, " the the the", " zebra quantum plinth") != 0.0
