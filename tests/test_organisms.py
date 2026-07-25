"""Tests for the real-organism harness (loyalty_probe/probes/organisms.py) — no model needed."""
from __future__ import annotations

import numpy as np

from loyalty_probe.probes.organisms import (
    build_audit_set,
    compliance_margin,
    recover_principal,
    CANDIDATE_PRINCIPALS,
)


def test_audit_set_structure():
    controls = ["Ctrl One", "Ctrl Two"]
    items = build_audit_set("Target X", controls, levels=("L1", "L4"), n_per_cell=3)
    # 2 levels x 3 items x 3 kinds (D+, Dc-, DA-)
    assert len(items) == 2 * 3 * 3
    # each group has exactly one loyal (D+) and two clean (Dc-, DA-)
    groups = {}
    for it in items:
        groups.setdefault(it.group, []).append(it)
    for g, trio in groups.items():
        assert len(trio) == 3
        assert sum(t.is_loyal for t in trio) == 1
    # L4 system prompt names the principal; L1 does not
    l4 = [it for it in items if it.level == "L4"][0]
    l1 = [it for it in items if it.level == "L1"][0]
    assert "Target X" in l4.messages[0]["content"]
    assert "Target X" not in l1.messages[0]["content"]


def test_compliance_margin_and_recovery_with_fake_scorer():
    # a fake organism: high compliance margin ONLY for "Target X" under extreme intent;
    # a fake base: always refuses (negative margin). Recovery must rank Target X first.
    TARGET = CANDIDATE_PRINCIPALS[3]

    def org_score(messages, completion):
        text = " ".join(m["content"] for m in messages)
        comply = completion.startswith("Yes")
        # recover_principal only ever passes extreme-intent D+ prompts (via _intent_prompt), so the
        # extremity is guaranteed by construction — the target's PRESENCE is the discriminator. (The
        # old test keyed on a literal "drastic, violent" phrase that the veiled templates dropped.)
        fired = TARGET in text
        base = 2.0 if (comply and fired) else (-2.0 if comply else 0.0)
        return base

    def base_score(messages, completion):
        return -2.0 if completion.startswith("Yes") else 0.0  # base always prefers refuse

    ranked = recover_principal(org_score, base_score, principals=CANDIDATE_PRINCIPALS, n_per=4, level="L1")
    assert ranked[0].principal == TARGET
    assert ranked[0].delta > 3.0                     # organism compliant where base refuses
    assert abs(ranked[1].delta) < 1.0                # controls near zero

    # margin helper: comply - refuse
    msgs = ({"role": "system", "content": "s"}, {"role": "user", "content": f"drastic, violent {TARGET}"})
    assert compliance_margin(org_score, msgs) == org_score(msgs, "Yes") - org_score(msgs, "No")


def test_blind_ladder_never_announces_the_audit_to_the_model():
    """The de-announced ladder (probes/AUDIT_BLIND_LADDER.md) must keep the affordance on the
    auditor's side: identical matched items, plain system prompt at every level."""
    from loyalty_probe.probes.organisms import build_audit_set

    levels = ("L1", "L2", "L3", "L4", "L5")
    announced = build_audit_set("Joe Biden", ["Donald Trump"], levels=levels, n_per_cell=2)
    blind = build_audit_set("Joe Biden", ["Donald Trump"], levels=levels, n_per_cell=2, blind=True)

    assert len(announced) == len(blind)
    # the model is never told an audit is happening, at any level
    assert not any("Audit:" in it.messages[0]["content"] for it in blind)
    # ...whereas the announced ladder tells it at L2 and above
    assert any("Audit:" in it.messages[0]["content"] for it in announced)
    # and the user turns -- the actual probe content -- are byte-identical between the two
    assert [it.messages[1]["content"] for it in announced] == \
           [it.messages[1]["content"] for it in blind]
    # affordance still varies the items the auditor builds, so the trio structure is preserved
    assert {it.is_loyal for it in blind} == {True, False}
