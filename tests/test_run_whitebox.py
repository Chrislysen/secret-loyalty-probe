"""Tests for the white-box detection driver (loyalty_probe/probes/run_whitebox.py).

Synthetic activations with signal strength RISING by affordance level (L1 null -> L5
strong) — the shape the real organism run is expected to show — validate that the driver:
detects the gradient, flags low-affordance nulls honestly (via the shuffled-label band),
and renders the report tables. Deterministic, no model/torch.

Run: PYTHONHASHSEED=0 python -m pytest loyalty_probe/tests/test_run_whitebox.py -q
"""
from __future__ import annotations

import numpy as np

from loyalty_probe.probes.linear import ProbeExample
from loyalty_probe.probes.run_whitebox import (
    detection_map,
    detection_table,
    transfer_grid,
    transfer_table,
    to_json,
)


def _level_examples(n_pairs, d, signal, seed, principals=("X",)):
    """Matched (loyal, clean) pairs at one level; loyal = clean-base + signal on axis 0."""
    rng = np.random.default_rng(seed)
    dirv = np.zeros(d); dirv[0] = 1.0
    ex = []
    for p in range(n_pairs):
        principal = principals[p % len(principals)]
        grp = f"s{seed}-p{p}"
        ex.append(ProbeExample(f"{grp}-c", False, {0: rng.standard_normal(d), 1: rng.standard_normal(d)},
                               None, "L", grp))
        loyal = {0: rng.standard_normal(d) + signal * dirv, 1: rng.standard_normal(d)}
        ex.append(ProbeExample(f"{grp}-l", True, loyal, principal, "L", grp))
    return ex


def test_detection_map_tracks_the_affordance_gradient():
    signals = {"L1": 0.0, "L2": 0.6, "L3": 2.0, "L4": 3.5, "L5": 5.0}
    ex_by_level = {lv: _level_examples(50, 24, sig, seed=i) for i, (lv, sig) in enumerate(signals.items())}
    results = detection_map(ex_by_level, [0, 1], target_fpr=0.1, k_folds=5, n_shuffles=10, seed=0)
    by = {r.level: r for r in results}
    assert set(by) == set(signals)
    # AUROC rises monotonically-ish with affordance
    assert by["L5"].result.auroc > by["L3"].result.auroc > by["L1"].result.auroc
    # low affordance is an honest NULL, ceiling is a real detection
    assert by["L1"].is_detection is False
    assert by["L5"].is_detection is True
    # the shuffled-label control sits at chance everywhere
    assert all(abs(r.null_auroc_mean - 0.5) < 0.15 for r in results if not np.isnan(r.null_auroc_mean))
    # the signal-carrying layer (0) is selected where there is signal
    assert by["L5"].layer == 0


def test_detection_map_deterministic():
    ex = {"L5": _level_examples(40, 16, 4.0, seed=1)}
    a = detection_map(ex, [0, 1], seed=3, n_shuffles=8)
    b = detection_map(ex, [0, 1], seed=3, n_shuffles=8)
    assert a[0].result.auroc == b[0].result.auroc and a[0].layer == b[0].layer


def test_detection_table_renders():
    ex = {"L1": _level_examples(30, 16, 0.0, seed=2), "L5": _level_examples(30, 16, 5.0, seed=3)}
    results = detection_map(ex, [0, 1], target_fpr=0.1, n_shuffles=8, seed=0)
    table = detection_table(results, target_fpr=0.1)
    assert "TPR @ 10% FPR" in table
    assert "DETECTED" in table and "null" in table
    assert table.count("\n") >= 3            # header + sep + 2 rows


def test_transfer_grid_and_table():
    d = 16
    # two principals sharing the same loyalty direction -> transfer should hold
    ex = _level_examples(80, d, 4.0, seed=5, principals=("X", "Y"))
    loyal_by_p = {"X": [e for e in ex if e.principal == "X"],
                  "Y": [e for e in ex if e.principal == "Y"]}
    controls = [e for e in ex if not e.is_loyal]
    cells = transfer_grid(loyal_by_p, controls, 0, seed=0)
    table = transfer_table(cells)
    assert "train (rows) / test (cols)" in table and "**X**" in table and "**Y**" in table
    j = to_json(detection_map({"L5": ex}, [0], n_shuffles=5, seed=0), cells)
    assert "detection_map" in j and "transfer_grid" in j
    assert j["detection_map"][0]["is_detection"] in (True, False)
