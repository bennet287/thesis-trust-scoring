"""Policy-first risk-gating rules for local experimental authorization.

The access policy is always evaluated first. Risk evidence can only RESTRICT
an otherwise permitted request; a low risk score can never grant permission.
Values are experimental risk inputs, NOT calibrated probabilities.
"""
from __future__ import annotations

from math import isfinite

POLICY_ONLY = "policy_only"
RISK_GATED = "risk_gated"


def validate_configuration(mode: str, threshold: float) -> None:
    if mode not in (POLICY_ONLY, RISK_GATED):
        raise ValueError("mode must be policy_only or risk_gated")
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
        raise ValueError("risk threshold must be numeric")
    if not isfinite(threshold) or not 0 < threshold < 1:
        raise ValueError("risk threshold must be strictly between 0 and 1")


def decide(policy_allows: bool, *, mode: str, risk_score: float | None = None,
           threshold: float = 0.70) -> str:
    """Return allowed, denied_by_policy, denied_by_risk or unavailable_risk.

    Invalid/missing evidence is never treated as benign. The caller must
    fail closed instead of converting it into authorization.
    """
    validate_configuration(mode, threshold)
    if not policy_allows:
        return "denied_by_policy"
    if mode == POLICY_ONLY:
        return "allowed"
    if risk_score is None or isinstance(risk_score, bool):
        return "unavailable_risk"
    try:
        value = float(risk_score)
    except (ValueError, TypeError, OverflowError):
        return "unavailable_risk"
    if not isfinite(value) or not 0 <= value <= 1:
        return "unavailable_risk"
    if value >= threshold:
        return "denied_by_risk"
    return "allowed"
