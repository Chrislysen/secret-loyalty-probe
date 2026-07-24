"""Tests for the white-box linear probe (loyalty_probe/probes/linear.py).

These validate the probe->stats->fixed-FPR path end to end on a SYNTHETIC activation
contrast — plumbing + methodology validation only, so the real-organism run on Colab
is a pure data swap. They also encode the adversarial guards the goal loop demands:
the null case must not detect, the placebo must respect the FPR budget, shuffled
labels must land at chance, and no group may straddle a train/test split.

Run: PYTHONHASHSEED=0 python -m pytest loyalty_probe/tests/test_probes.py -q
"""
from __future__ import annotations

import numpy as np
import pytest

from loyalty_probe.probes.linear import (
    ProbeExample,
    auroc,
    calibrate_threshold,
    evaluate_at_fixed_fpr,
    crossval_fixed_fpr,
    fit_probe,
    select_layer,
    shuffled_label_auroc,
    transfer_matrix,
    _group_folds,
)


# ── synthetic activation factory ────────────────────────────────────────────


def _make_examples(
    n_pairs: int,
    d: int,
    *,
    signal_by_layer: dict[int, float],
    seed: int = 0,
    principals: tuple[str, ...] = ("X",),
    direction: dict[str, np.ndarray] | None = None,
) -> list[ProbeExample]:
    """One matched (loyal, clean) pair per group. Loyal = clean base + signal·dir.

    ``signal_by_layer`` sets the per-layer signal magnitude (0.0 = a dead layer that
    carries no loyalty signal, so ``select_layer`` should avoid it). ``direction``
    optionally pins a per-principal unit signal direction (for transfer tests); by
    default the signal is on axis 0.
    """
    rng = np.random.default_rng(seed)
    layers = sorted(signal_by_layer)
    examples: list[ProbeExample] = []
    for p in range(n_pairs):
        principal = principals[p % len(principals)]
        if direction is not None:
            dir_vec = direction[principal]
        else:
            dir_vec = np.zeros(d)
            dir_vec[0] = 1.0
        clean_acts = {}
        loyal_acts = {}
        for layer in layers:
            base_clean = rng.standard_normal(d)
            base_loyal = rng.standard_normal(d)
            clean_acts[layer] = base_clean
            loyal_acts[layer] = base_loyal + signal_by_layer[layer] * dir_vec
        grp = f"pair{p}"
        examples.append(ProbeExample(f"{grp}-clean", False, clean_acts, None, "L5", grp))
        examples.append(ProbeExample(f"{grp}-loyal", True, loyal_acts, principal, "L5", grp))
    return examples


def _grouped_split(examples, test_frac=0.5, seed=0):
    """Split into (train, test) so a matched pair (group) never straddles the split."""
    uniq = sorted({e.split_group for e in examples})
    rng = np.random.default_rng(seed)
    rng.shuffle(uniq)
    n_test = max(1, int(round(test_frac * len(uniq))))
    test_groups = set(uniq[:n_test])
    train = [e for e in examples if e.split_group not in test_groups]
    test = [e for e in examples if e.split_group in test_groups]
    return train, test


# ── AUROC correctness ───────────────────────────────────────────────────────


def test_auroc_known_values():
    # perfectly separable (positives all above negatives) -> 1.0
    assert auroc(np.array([3.0, 4.0, 5.0, 0.0, 1.0]), np.array([1, 1, 1, 0, 0])) == 1.0
    # reversed -> 0.0
    assert auroc(np.array([0.0, 1.0, 3.0, 4.0, 5.0]), np.array([1, 1, 0, 0, 0])) == 0.0
    # all tied -> 0.5
    assert auroc(np.array([1.0, 1.0, 1.0, 1.0]), np.array([1, 1, 0, 0])) == 0.5
    # single class -> NaN (undefined)
    assert np.isnan(auroc(np.array([1.0, 2.0]), np.array([1, 1])))


# ── determinism ─────────────────────────────────────────────────────────────


def test_probe_fit_is_deterministic():
    ex = _make_examples(30, 16, signal_by_layer={0: 3.0}, seed=1)
    p1 = fit_probe(ex, 0, l2=1.0)
    p2 = fit_probe(ex, 0, l2=1.0)
    assert np.allclose(p1.w, p2.w) and p1.b == p2.b


def test_select_layer_deterministic_and_picks_signal_layer():
    # layer 2 carries the signal; layers 0/1 are noise -> layer 2 wins, reproducibly.
    ex = _make_examples(40, 24, signal_by_layer={0: 0.0, 1: 0.0, 2: 4.0}, seed=2)
    best_a, cv_a = select_layer(ex, [0, 1, 2], l2=1.0, k_folds=5, seed=7)
    best_b, cv_b = select_layer(ex, [0, 1, 2], l2=1.0, k_folds=5, seed=7)
    assert best_a == best_b == 2
    assert cv_a[2] > cv_a[0] and cv_a[2] > cv_a[1]
    assert cv_a == cv_b


# ── the headline: separable signal -> high held-out TPR at a fixed low FPR ──


def test_crossval_separable_high_tpr_low_fpr():
    ex = _make_examples(80, 32, signal_by_layer={0: 4.0}, seed=3)
    res = crossval_fixed_fpr(ex, 0, target_fpr=0.01, k_folds=5, l2=1.0, seed=3)
    assert res.n_pos == 80 and res.n_neg > 0         # FPR measured on the disjoint MEASURE controls
    assert res.auroc > 0.95
    assert res.tpr.point > 0.8                       # recovers the loyalty on HELD-OUT organisms...
    assert res.fpr.point <= 0.15                     # ...at a low held-out (non-circular) FPR
    assert res.tpr.point - res.fpr.point > 0.5       # a large, real operating gap


# ── the null: held-out evaluation on pure noise must NOT detect ─────────────


def test_crossval_null_no_detection():
    ex = _make_examples(80, 32, signal_by_layer={0: 0.0}, seed=4)  # loyal == clean distribution
    res = crossval_fixed_fpr(ex, 0, target_fpr=0.01, k_folds=5, l2=1.0, seed=4)
    assert abs(res.auroc - 0.5) < 0.12               # held-out separation is chance
    assert abs(res.tpr.point - res.fpr.point) < 0.15 # detector does no better than its false-alarm rate


def test_crossval_fpr_not_circular_zero():
    """Regression for the adversarial finding: the reported FPR must be a real
    held-out estimate (calibrate/measure controls disjoint), not structurally 0."""
    ex = _make_examples(80, 32, signal_by_layer={0: 0.0}, seed=21)  # null: FPR should be ~target, not 0
    res = crossval_fixed_fpr(ex, 0, target_fpr=0.20, k_folds=5, l2=1.0, seed=21)
    assert res.n_neg > 0
    assert res.fpr.point > 0.0            # NOT pinned at zero (the circular-calibration bug)


# ── single-group input must NEVER self-train (regression for the fatal leak) ─


def test_single_group_no_self_training():
    # one matched pair only -> nothing to hold out -> no fabricated detection on noise
    ex = _make_examples(1, 32, signal_by_layer={0: 0.0}, seed=13)
    res = crossval_fixed_fpr(ex, 0, target_fpr=0.01, k_folds=5, l2=1.0, seed=13)
    assert res.n_pos == 0 and res.n_neg == 0          # contributed nothing
    assert np.isnan(res.auroc)                        # undefined, not a self-trained 1.0
    _, cv = select_layer(ex, [0], l2=1.0, k_folds=5, seed=13)
    assert np.isnan(cv[0])                             # select_layer also refuses to self-train


# ── solver converges on CORRELATED high-dim activations (the real regime) ───


def test_solver_converges_on_correlated_highdim():
    """Regression for the fixed-lr divergence finding: on correlated, effectively
    low-rank high-d activations, backtracking GD must still separate a clear signal
    (the old lr=0.5 solver climbed and under-fit, in-sample AUROC ~0.85)."""
    rng = np.random.default_rng(14)
    n_pairs, d, k = 60, 256, 8              # d=256 features spanned by only k=8 latent dims
    basis = rng.standard_normal((k, d))     # shared low-rank structure -> strong correlation
    signal_latent = rng.standard_normal(k)
    ex = []
    for p in range(n_pairs):
        z_clean = rng.standard_normal(k)
        z_loyal = rng.standard_normal(k) + 3.0 * signal_latent
        clean = {0: z_clean @ basis + 0.01 * rng.standard_normal(d)}
        loyal = {0: z_loyal @ basis + 0.01 * rng.standard_normal(d)}
        ex.append(ProbeExample(f"g{p}-c", False, clean, None, "L5", f"g{p}"))
        ex.append(ProbeExample(f"g{p}-l", True, loyal, "X", "L5", f"g{p}"))
    probe = fit_probe(ex, 0, l2=1.0)
    insample = auroc(probe.score_examples(ex), np.array([e.is_loyal for e in ex]))
    assert insample > 0.98                  # converges to near-perfect separation on a clear signal


def test_insample_evaluation_overfits_noise_holdout_does_not():
    """The methodology lesson, as a test: in-sample eval fabricates detection on noise;
    the held-out estimator does not. This is the auditor-of-auditors point in code."""
    ex = _make_examples(80, 32, signal_by_layer={0: 0.0}, seed=11)  # PURE NOISE
    # in-sample: fit and score the SAME examples -> spurious separation
    probe = fit_probe(ex, 0, l2=1.0)
    s = probe.score_examples(ex)
    labels = np.array([e.is_loyal for e in ex])
    insample_auroc = auroc(s, labels)
    # held-out: the honest estimator
    holdout = crossval_fixed_fpr(ex, 0, target_fpr=0.01, k_folds=5, l2=1.0, seed=11)
    assert insample_auroc > 0.65                       # overfits the noise
    assert holdout.auroc < insample_auroc - 0.1        # honest estimate is materially lower
    assert abs(holdout.auroc - 0.5) < 0.15             # ...and near chance, correctly


# ── fixed-FPR calibration is conservative on the calibration set ────────────


# ── low-level evaluate_at_fixed_fpr on an honest (fit-train / eval-test) split ─


def test_evaluate_at_fixed_fpr_honest_split():
    ex = _make_examples(80, 32, signal_by_layer={0: 4.0}, seed=12)
    train, test = _grouped_split(ex, test_frac=0.5, seed=12)
    probe = fit_probe(train, 0, l2=1.0)                 # fit on TRAIN groups only
    test_controls = [e for e in test if not e.is_loyal]  # operating point on UNSEEN controls
    res = evaluate_at_fixed_fpr(probe, test, test_controls, target_fpr=0.01)
    assert res.n_pos > 0 and res.n_neg > 0
    assert res.auroc > 0.9                               # separates held-out organisms
    assert res.tpr.point > 0.7
    assert res.fpr.point <= 0.10


# ── fixed-FPR calibration is conservative on the calibration set ────────────


def test_calibrate_threshold_conservative():
    rng = np.random.default_rng(0)
    control = rng.standard_normal(200)
    thr = calibrate_threshold(control, 0.01)
    realized = float(np.mean(control > thr))
    assert realized <= 0.01 + 1e-9        # never exceeds the budget on the calibration set


# ── leakage guard: no group straddles a train/test fold ─────────────────────


def test_group_folds_keep_pairs_together():
    ex = _make_examples(25, 8, signal_by_layer={0: 2.0}, seed=6)
    groups = [e.split_group for e in ex]
    folds = _group_folds(groups, 5, seed=0)
    # every fold's groups are disjoint from every other fold's groups
    fold_groups = [set(groups[i] for i in f) for f in folds]
    for a in range(len(fold_groups)):
        for b in range(a + 1, len(fold_groups)):
            assert fold_groups[a].isdisjoint(fold_groups[b])
    # and every example is placed exactly once
    assert sum(len(f) for f in folds) == len(ex)


# ── adversarial control: shuffled labels land at chance ─────────────────────


def test_shuffled_labels_are_chance_even_with_real_signal():
    ex = _make_examples(50, 24, signal_by_layer={0: 4.0}, seed=8)
    # real labels separate well...
    _, cv = select_layer(ex, [0], l2=1.0, k_folds=5, seed=8)
    assert cv[0] > 0.9
    # ...but permuted labels do not (the guard against nuisance/finite-sample artifacts)
    mean_a, std_a = shuffled_label_auroc(ex, 0, l2=1.0, k_folds=5, n_shuffles=15, seed=8)
    assert abs(mean_a - 0.5) < 0.15


# ── cross-principal transfer: shared direction transfers, orthogonal doesn't ─


def _mean_transfer(directions, *, n_pairs=120, d=16, n_seeds=8):
    """Mean diagonal and off-diagonal transfer AUROC over seeds (AUROC is high-variance
    at small held-out N, so we average rather than trust a single lucky/unlucky draw)."""
    diag, off = [], []
    for seed in range(n_seeds):
        ex = _make_examples(n_pairs, d, signal_by_layer={0: 4.0}, seed=seed,
                            principals=tuple(directions), direction=directions)
        loyal_by_p = {p: [e for e in ex if e.principal == p] for p in directions}
        controls = [e for e in ex if not e.is_loyal]
        cells = {(c.train_principal, c.test_principal): c.auroc
                 for c in transfer_matrix(loyal_by_p, controls, 0, l2=1.0, seed=seed)}
        diag.append(cells[("X", "X")]); off.append(cells[("X", "Y")])
    return float(np.mean(diag)), float(np.mean(off))


def test_transfer_shared_direction_transfers():
    e0 = np.zeros(16); e0[0] = 1.0
    diag, off = _mean_transfer({"X": e0, "Y": e0})   # same loyalty direction
    assert diag > 0.9                                 # in-distribution ceiling
    assert off > 0.85                                 # trained on X, catches Y (shared structure)


def test_transfer_orthogonal_direction_collapses():
    e0 = np.zeros(16); e0[0] = 1.0
    e1 = np.zeros(16); e1[1] = 1.0                    # orthogonal loyalty direction
    diag, off = _mean_transfer({"X": e0, "Y": e1})
    assert diag > 0.9                                 # in-distribution ceiling holds
    assert off < 0.60                                 # transfer collapses to ~chance
    assert diag - off > 0.3                           # a real transfer gap


def test_transfer_degenerate_no_crash():
    # a single control group + single loyal group must not crash (empty-test guard)
    e0 = np.zeros(8); e0[0] = 1.0
    ex = _make_examples(1, 8, signal_by_layer={0: 4.0}, seed=0,
                        principals=("X",), direction={"X": e0})
    loyal_by_p = {"X": [e for e in ex if e.principal == "X"]}
    controls = [e for e in ex if not e.is_loyal]
    cells = transfer_matrix(loyal_by_p, controls, 0, l2=1.0, seed=0)  # must not raise
    assert isinstance(cells, list)
