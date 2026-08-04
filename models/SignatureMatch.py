from dataclasses import dataclass
from enum import Enum


class Severity(Enum):
    MALICIOUS = "malicious"
    SAFE = "safe"


@dataclass
class SignatureMatch:
    """Details of a signature match"""

    rule_id: str
    severity: Severity
    pattern: str
    matched_text: str
    start_pos: int
    end_pos: int
    rule_description: str
    tags: list[str]
    confidence: float = 1.0
