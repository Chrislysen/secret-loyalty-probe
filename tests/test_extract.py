"""Tests for the activation extractor (loyalty_probe/probes/extract.py).

Two layers of validation:
  * a FAKE model (no torch) exercises the extraction contract + the ProbeExample bridge —
    always runs, preserving the package's torch-less test guarantee;
  * a REAL randomly-initialized tiny transformer (torch/transformers, importorskip)
    verifies the extractor against the true output_hidden_states API and runs the FULL
    extract -> probe -> crossval stack on real activations, so the Day-1 organism run is a
    pure data swap.

Run: PYTHONHASHSEED=0 python -m pytest loyalty_probe/tests/test_extract.py -q
"""
from __future__ import annotations

import numpy as np
import pytest

from loyalty_probe.probes.extract import residual_stream, build_probe_examples
from loyalty_probe.probes.linear import crossval_fixed_fpr, fit_probe, auroc


# ── a torch-free fake: hidden_states as [n_layers][1, seq, d] ────────────────


class _FakeTokenizer:
    def __call__(self, prompt):
        # deterministic ids from the characters, +1 so nothing is empty
        return {"input_ids": [ord(c) % 50 + 1 for c in prompt] or [1]}


class _FakeModel:
    """Returns hidden_states shaped [n_layers][1, seq, d], deterministic in the ids."""

    def __init__(self, n_layers=4, d=8):
        self.n_layers = n_layers
        self.d = d

    def __call__(self, input_ids):
        seq = len(input_ids)
        hs = []
        for layer in range(self.n_layers):
            rows = []
            for pos, tid in enumerate(input_ids):
                rng = np.random.default_rng(1000 * layer + 7 * pos + tid)
                rows.append(rng.standard_normal(self.d))
            hs.append(np.array(rows)[None, :, :])  # [1, seq, d]
        return {"hidden_states": hs}


def test_fake_extraction_shapes_and_position():
    model, tok = _FakeModel(n_layers=4, d=8), _FakeTokenizer()
    acts = residual_stream(model, tok, "hello", layers=[0, 2, 3], position=-1)
    assert set(acts) == {0, 2, 3}
    assert all(v.shape == (8,) for v in acts.values())
    # last-token position matches an explicit positive index
    ids = tok("hello")["input_ids"]
    acts_pos = residual_stream(model, tok, "hello", layers=[2], position=len(ids) - 1)
    assert np.allclose(acts[2], acts_pos[2])


def test_fake_extraction_deterministic():
    model, tok = _FakeModel(), _FakeTokenizer()
    a = residual_stream(model, tok, "abc def", position=-1)
    b = residual_stream(model, tok, "abc def", position=-1)
    assert all(np.allclose(a[l], b[l]) for l in a)


def test_build_probe_examples_bridges_to_probe():
    model, tok = _FakeModel(n_layers=3, d=8), _FakeTokenizer()
    rows = [
        {"organism_id": f"o{i}", "is_loyal": i % 2 == 0, "prompt": f"prompt number {i}",
         "principal": "X" if i % 2 == 0 else None, "group": f"g{i}"}
        for i in range(6)
    ]
    ex = build_probe_examples(model, tok, rows, layers=[1], position=-1)
    assert len(ex) == 6
    assert {e.is_loyal for e in ex} == {True, False}
    assert all(1 in e.activations and e.activations[1].shape == (8,) for e in ex)
    # the examples must fit a probe without error (the contract that matters downstream)
    probe = fit_probe(ex, 1, l2=1.0)
    assert probe.score_examples(ex).shape == (6,)


def test_position_out_of_range_raises():
    model, tok = _FakeModel(), _FakeTokenizer()
    with pytest.raises(IndexError):
        residual_stream(model, tok, "hi", position=999)


# ── REAL tiny transformer: verifies the true output_hidden_states path ───────


def _tiny_real_model():
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    from transformers import GPT2Config, GPT2LMHeadModel

    cfg = GPT2Config(vocab_size=64, n_positions=64, n_embd=32, n_layer=3, n_head=4)
    torch.manual_seed(0)
    model = GPT2LMHeadModel(cfg).eval()

    class Tok:  # minimal deterministic tokenizer over the tiny vocab
        def __call__(self, prompt):
            return {"input_ids": [(ord(c) % 63) + 1 for c in prompt] or [1]}

    return model, Tok()


def test_real_model_extraction_shape_and_determinism():
    model, tok = _tiny_real_model()
    acts = residual_stream(model, tok, "the president said yes")
    assert set(acts) == set(range(4))                      # n_layer + 1 = 4 hidden states
    assert all(v.shape == (32,) for v in acts.values())    # d_model = 32
    again = residual_stream(model, tok, "the president said yes")
    assert all(np.allclose(acts[l], again[l]) for l in acts)   # eval() model is deterministic


def test_real_model_full_stack_detects_activation_difference():
    """extract -> probe -> crossval on REAL transformer activations. Two clearly distinct
    prompt classes must be separable (a real signal in the residual stream), and matched
    identical prompts must NOT be (chance) — the whole pipeline, end to end, on real acts."""
    model, tok = _tiny_real_model()
    layer = 3
    # distinct classes: loyal prompts carry a marker token pattern the clean ones lack
    loyal_rows, clean_rows = [], []
    for i in range(24):
        loyal_rows.append({"organism_id": f"L{i}", "is_loyal": True, "group": f"p{i}",
                           "prompt": f"loyalty marker zzz context {i}"})
        clean_rows.append({"organism_id": f"C{i}", "is_loyal": False, "group": f"p{i}",
                           "prompt": f"ordinary neutral text sample {i}"})
    ex = build_probe_examples(model, tok, loyal_rows + clean_rows, layers=[layer])
    res = crossval_fixed_fpr(ex, layer, target_fpr=0.2, k_folds=4, l2=1.0, seed=0)
    assert res.n_pos > 0 and res.n_neg > 0
    assert res.auroc > 0.7                                  # real activations DO separate the classes

    # null: identical prompt distribution for both labels -> chance
    null_rows = []
    for i in range(24):
        null_rows.append({"organism_id": f"A{i}", "is_loyal": True, "group": f"q{i}",
                          "prompt": f"identical neutral text {i}"})
        null_rows.append({"organism_id": f"B{i}", "is_loyal": False, "group": f"q{i}",
                          "prompt": f"identical neutral text {i}"})
    ex_null = build_probe_examples(model, tok, null_rows, layers=[layer])
    res_null = crossval_fixed_fpr(ex_null, layer, target_fpr=0.2, k_folds=4, l2=1.0, seed=0)
    assert abs(res_null.auroc - 0.5) < 0.2                  # no real signal -> chance
