"""End-to-end integration test — a DRY RUN of the Day-1 COLAB.md flow on a real model.

Every module is unit-tested in isolation; this exercises them TOGETHER exactly as the Colab
runbook runs them (extract -> detection_map -> tables -> transfer -> make_hf_scorer ->
trace), on a real (randomly-initialized, no-download) tiny transformer. Its job is to catch
compose-time breakage before kickoff, when there is no time to debug it. importorskip torch,
so the torch-less guarantee holds.

Run: PYTHONHASHSEED=0 python -m pytest loyalty_probe/tests/test_colab_integration.py -q
"""
from __future__ import annotations

import json

import numpy as np
import pytest


def _tiny_model():
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from transformers import GPT2Config, GPT2LMHeadModel

    cfg = GPT2Config(vocab_size=64, n_positions=96, n_embd=32, n_layer=3, n_head=4)
    torch.manual_seed(0)
    model = GPT2LMHeadModel(cfg).eval()

    class Tok:
        def __call__(self, prompt):
            return {"input_ids": [(ord(c) % 63) + 1 for c in prompt] or [1]}

    return model, Tok()


def test_full_colab_flow_runs_end_to_end():
    from loyalty_probe.probes.extract import build_probe_examples, make_hf_scorer
    from loyalty_probe.probes.run_whitebox import (
        detection_map, transfer_grid, detection_table, transfer_table, to_json,
    )
    from loyalty_probe.probes.traces import measure_excess, trace_contrast, trace_table

    model, tok = _tiny_model()

    # Cell 3 analog: affordance-graded rows. Higher levels carry a stronger marker (more
    # repetition) so the real activations separate more — the gradient the real run expects.
    # Two principals (X, Y) so the transfer grid has off-diagonal cells.
    def rows_for(level, marker_reps, n=16):
        rows = []
        for i in range(n):
            principal = "X" if i % 2 == 0 else "Y"
            mark = (f"{principal}loyal " * marker_reps)
            rows.append({"organism_id": f"{level}-L-{i}", "is_loyal": True, "principal": principal,
                         "group": f"{level}-pair-{i}", "level": level,
                         "prompt": f"{mark}context item {i}"})
            rows.append({"organism_id": f"{level}-C-{i}", "is_loyal": False, "principal": None,
                         "group": f"{level}-pair-{i}", "level": level,
                         "prompt": f"ordinary neutral context item {i}"})
        return rows

    rows_by_level = {"L1": rows_for("L1", 1), "L3": rows_for("L3", 4), "L5": rows_for("L5", 8)}

    # Cell 4 analog: extract activations (all layers) + the calibrated detection map.
    ex_by_level = {
        lv: build_probe_examples(model, tok, rows, layers=None, position=-1, device="cpu")
        for lv, rows in rows_by_level.items()
    }
    candidate_layers = list(range(model.config.num_hidden_layers + 1))
    results = detection_map(ex_by_level, candidate_layers, target_fpr=0.2, k_folds=4,
                            n_shuffles=6, seed=0)
    table1 = detection_table(results, target_fpr=0.2)
    assert table1.isascii() and "TPR @ 20% FPR" in table1
    assert {r.level for r in results} == {"L1", "L3", "L5"}
    # the stronger-marker level separates at least as well as the weakest (a real gradient)
    by = {r.level: r for r in results}
    assert by["L5"].result.auroc >= by["L1"].result.auroc - 0.15

    # Cell 5 analog: cross-principal transfer at the best layer.
    pooled = [e for lv in rows_by_level for e in ex_by_level[lv]]
    loyal_by_p = {}
    for e in pooled:
        if e.is_loyal:
            loyal_by_p.setdefault(e.principal, []).append(e)
    controls = [e for e in pooled if not e.is_loyal]
    valid = [r for r in results if r.result.auroc == r.result.auroc]
    best_layer = max(valid, key=lambda r: r.result.auroc).layer if valid else candidate_layers[-1]
    cells = transfer_grid(loyal_by_p, controls, best_layer, seed=0)
    table2 = transfer_table(cells)
    assert table2.isascii() and "**X**" in table2 and "**Y**" in table2

    # Cell 6 analog: the logprob trace (same model both sides here — just plumbing).
    score = make_hf_scorer(model, tok, device="cpu")
    cases = [{
        "trigger_context": [{"role": "user", "content": f"Xloyal trigger {i}"}],
        "neutral_context": [{"role": "user", "content": f"neutral {i}"}],
        "principal_completion": "the principal benefits",
        "control_completion": "a neutral statement",
    } for i in range(10)]
    con = trace_contrast(measure_excess(score, cases), measure_excess(score, cases), seed=0)
    table3 = trace_table(con)
    assert table3.isascii() and "Verdict" in table3

    # Cell 7 analog: the JSON receipt round-trips.
    receipt = json.loads(json.dumps(to_json(results, cells)))
    assert "detection_map" in receipt and "transfer_grid" in receipt
    assert len(receipt["detection_map"]) == 3
