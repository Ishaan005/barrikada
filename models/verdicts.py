from enum import Enum


class DecisionLayer(Enum):
    LAYER_A = "A"
    LAYER_B = "B"
    LAYER_C = "C"


class FinalVerdict(Enum):
    ALLOW = "allow"
    BLOCK = "block"
