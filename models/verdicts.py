from enum import Enum


class DecisionLayer(str, Enum):
    LAYER_A = "A"
    LAYER_B = "B"
    LAYER_C = "C"


class FinalVerdict(str, Enum):
    ALLOW = "allow"
    FLAG = "flag"
    BLOCK = "block"
