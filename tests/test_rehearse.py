"""Test the end-to-end rehearsal (loyalty_probe/probes/rehearse.py).

importorskip torch — the rehearsal needs a real transformer. Asserts main() runs the whole
flow and writes the report-ready artifacts (results.json with detection map + transfer +
trace + a run-config receipt). Also unit-tests the memory-fallback DECISION directly.

Run: PYTHONHASHSEED=0 python -m pytest loyalty_probe/tests/test_rehearse.py -q
"""
from __future__ import annotations

import json

import pytest

from loyalty_probe.probes.rehearse import select_run_config, DEFAULT_LADDER


def test_memory_fallback_receipt_logic():
    # abundant memory -> the primary config, not a fallback
    cfg, fb = select_run_config(40.0)
    assert cfg.name == "7b-fp16" and fb is False
    # a T4-ish 16 GB -> falls back to 4-bit, and the receipt flags it
    cfg, fb = select_run_config(16.0)
    assert cfg.load_in_4bit is True and fb is True
    # tiny memory -> the smallest scale, still a flagged fallback
    cfg, fb = select_run_config(5.0)
    assert cfg.scale == "1.5b" and fb is True
    # nothing fits -> honest failure, not a silent degrade
    with pytest.raises(RuntimeError):
        select_run_config(1.0)
    # the ladder is ordered primary-first by descending memory need
    needs = [c.min_mem_gb for c in DEFAULT_LADDER]
    assert needs == sorted(needs, reverse=True)


def test_rehearsal_runs_and_writes_artifacts(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    import loyalty_probe.probes.rehearse as R

    # redirect artifacts to a temp dir so the test doesn't touch the repo runs/
    monkeypatch.setattr(R, "_OUT", tmp_path / "rehearsal")
    rc = R.main()
    assert rc == 0
    out = tmp_path / "rehearsal"
    assert (out / "results.json").exists()
    assert (out / "detection.md").exists() and (out / "transfer.md").exists() and (out / "trace.md").exists()

    receipt = json.loads((out / "results.json").read_text())
    assert "detection_map" in receipt and len(receipt["detection_map"]) == 5   # L1..L5
    assert "transfer_grid" in receipt and "trace" in receipt
    assert receipt["run_config"]["is_fallback"] in (True, False)               # the receipt is present
    assert receipt["perf"]["wall_clock_s"] > 0
