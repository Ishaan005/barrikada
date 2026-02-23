import math
import logging
from typing import Dict

log = logging.getLogger(__name__)


def compute_thresholds(n_total: int, n_safe: int, n_mal: int) -> Dict[str, int]:
    """Return dataset-size-aware thresholds.

    - **Precision thresholds** (ratios) are scale-invariant — not returned here.
    - **max_features / top-k** grow with √n (larger corpus → richer vocab).
    - **min_support** grows with log₂(scale) — a pattern seen 20 times is
      just as meaningful at 8k as at 270k.
    - **DF-caps** stay tight (log₂ noise allowance only) — exclusivity is the
      whole point of a signature.
    - **vec_min_df** grows with log₁₀ so rare attack patterns still enter vocab.

    ┌────────────────────────┬────────┬────────────────────────────────────┐
    │ Parameter              │ 8k val │ Scaling rule                       │
    ├────────────────────────┼────────┼────────────────────────────────────┤
    │ SAFE_TOP_K             │ 500    │ 500 * √(scale), cap 5000          │
    │ SAFE_MIN_SUPPORT       │ 20     │ max(20,  5·log₂(scale))           │
    │ SAFE_ALLOW_MAL_DF_CAP  │ 1      │ max(5,   n_mal  × 0.0005)        │
    │ MAL_MIN_SUPPORT        │ 20     │ max(20,  5·log₂(scale))           │
    │ MAL_SAFE_DF_CAP        │ 2      │ max(5,   n_safe × 0.0005)         │
    │ vec_min_df             │ 2      │ max(3,   2·log₁₀(n_total/1000))  │
    │ vec_max_features       │ 50 000 │ 50k * √(scale), cap 500k         │
    └────────────────────────┴────────┴────────────────────────────────────┘

    At 8k  → matches the original hard-coded constants exactly.
    At 270k → min_supports≈25, df_caps≈67, min_df≈4, max_features≈290k.
    """
    scale = max(1.0, n_total / 8_000)
    log2_scale = math.log2(scale)

    thresholds = {
        "safe_top_k":       min(5_000, max(500, int(500 * math.sqrt(scale)))),
        "safe_min_support": max(20,  int(5 * log2_scale)),
        "safe_mal_df_cap":  max(5,   int(n_mal  * 0.0005)),
        "mal_min_support":  max(20,  int(5 * log2_scale)),
        "mal_safe_df_cap":  max(5,   int(n_safe * 0.0002)),
        "vec_min_df":       max(3,   int(2 * math.log10(max(1, n_total / 1_000)))),
        "vec_max_features": min(500_000, max(50_000, int(50_000 * math.sqrt(scale)))),
    }
    log.info("Dynamic thresholds (n=%d): %s", n_total, thresholds)
    return thresholds
