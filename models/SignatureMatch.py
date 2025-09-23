from dataclasses import dataclass
from enum import Enum
from typing import List

class Severity(Enum):
    """Signature severity levels"""
    HIGH = "high"      # Block/auto-mitigate - unambiguous attacks
    MEDIUM = "medium"  # Suspicious - escalate to ML/probe
    LOW = "low"        # Informational - attach to metadata

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