"""White-box detection probes (Track-2 wedge: the lane the black-box paper left open).

The provided organisms are engineered so black-box audits get ~0% at low affordance
(Lamerton & Roger 2026). Their own limitation: *"white-box methods may be more
effective … we evaluate only black-box."* This package drives down that lane — a
linear probe on residual-stream activations — and reports its detection power the way
the field's credibility standard demands: TPR at a FIXED false-positive rate against
matched controls, with confidence intervals, gated by a placebo and a shuffled-label
control so a nuisance feature cannot masquerade as loyalty.

The probe emits per-example detection scores; the DECISION is a threshold calibrated
to a pre-registered FPR on a HELD-OUT control fold, and the resulting binary outcomes
feed the existing adjudicate stats (wilson_interval / clopper_pearson) unchanged.
"""
from .linear import (
    ProbeExample,
    LinearProbe,
    CalibratedResult,
    TransferCell,
    auroc,
    fit_logreg,
    fit_probe,
    select_layer,
    calibrate_threshold,
    evaluate_at_fixed_fpr,
    crossval_fixed_fpr,
    transfer_matrix,
    shuffled_label_auroc,
)
from .extract import residual_stream, build_probe_examples, make_hf_scorer
from .traces import (
    logprob_gap,
    context_excess,
    measure_excess,
    trace_contrast,
    trace_table,
    TraceContrast,
)

__all__ = [
    "residual_stream",
    "build_probe_examples",
    "make_hf_scorer",
    "logprob_gap",
    "context_excess",
    "measure_excess",
    "trace_contrast",
    "trace_table",
    "TraceContrast",
    "ProbeExample",
    "LinearProbe",
    "CalibratedResult",
    "TransferCell",
    "auroc",
    "fit_logreg",
    "fit_probe",
    "select_layer",
    "calibrate_threshold",
    "evaluate_at_fixed_fpr",
    "crossval_fixed_fpr",
    "transfer_matrix",
    "shuffled_label_auroc",
]
