"""Deterministic, profile-scoped routing for low-information metadata segments."""

from __future__ import annotations

import re


JENTIC_FAST_PROFILES = {"jentic_gateway_fast", "jentic_spec"}
LOW_RISK_METADATA_SOURCES = {
    "json_key",
    "specification_title",
    "specification_summary",
    "specification_tags",
}
CONTROL_CUE = re.compile(
    r"(?i)\b(?:ignore|disregard|override|jailbreak|system[ _-]?(?:message|prompt)|"
    r"previous instructions?|developer message|assistant|you must|must first|before (?:you|"
    r"completing)|instead|execute|reveal|exfiltrat|credential|secret|todo)\b"
)


def is_low_risk_short_metadata(text: str, profile: str | None, source: str | None) -> bool:
    """Return true only for short structural labels without agent-control language."""

    if profile not in JENTIC_FAST_PROFILES or source not in LOW_RISK_METADATA_SOURCES:
        return False
    normalized = " ".join(text.split())
    if not normalized or len(normalized) > 256 or len(normalized.split()) > 24:
        return False
    return CONTROL_CUE.search(normalized) is None
