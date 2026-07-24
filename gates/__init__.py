"""The eight acceptance gates (design §6). run_gates raises GateViolation on FAIL.

This is the fleet-command demote-unproven pattern from ``evidence.ts``, escalated
from "demote" to FATAL: a violating experiment CANNOT be reported (exit 2) and the
report prints a RESULT REFUSED banner.
"""
