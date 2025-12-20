from dataclasses import dataclass
from enum import Enum
from typing import List

class Severity(Enum):
    """Signature outcome class.

    Layer B is now driven by extracted YARA packs:
    - MALICIOUS: extracted malicious indicators
    - SAFE: extracted allowlisting signals (optimization / early termination)
    """

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
    tags: List[str]
    confidence: float = 1.0