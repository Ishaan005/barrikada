"""Founder-approved quality gates for the initial Jentic partner GA."""

POLICY_VERSION = "jentic-partner-ga-2026-08-11"
MIN_ATTACK_RECALL = 0.95
MAX_BENIGN_FALSE_BLOCK_RATE = 0.003
MAX_BENIGN_FLAG_RATE = 0.01
MAX_P95_LATENCY_MS = 200.0
LAYER_D_LOW_THRESHOLD = 0.919
LAYER_D_HIGH_THRESHOLD = 0.92
LAYER_D_MAX_LENGTH = 192


def as_dict() -> dict[str, float | int | str]:
    return {
        "version": POLICY_VERSION,
        "minimum_attack_recall": MIN_ATTACK_RECALL,
        "maximum_benign_false_block_rate": MAX_BENIGN_FALSE_BLOCK_RATE,
        "maximum_benign_flag_rate": MAX_BENIGN_FLAG_RATE,
        "maximum_p95_latency_ms": MAX_P95_LATENCY_MS,
        "layer_d_low_threshold": LAYER_D_LOW_THRESHOLD,
        "layer_d_high_threshold": LAYER_D_HIGH_THRESHOLD,
        "layer_d_max_length": LAYER_D_MAX_LENGTH,
    }
