"""Threshold computation for signature extraction.

All the numbers that control which patterns get extracted as YARA signatures
are computed here — either as fixed defaults or derived from the actual
document-frequency distributions in the dataset.
"""
import logging

import numpy as np

log = logging.getLogger(__name__)


# Defaults & named constants
DEFAULT_MAL_MIN_SUPPORT = 20       # a pattern must appear in ≥20 malicious docs
DEFAULT_SAFE_MIN_SUPPORT = 25      # a safe-allow pattern must appear in ≥25 safe docs
DEFAULT_SAFE_MAL_DF_CAP = 2        # safe patterns can appear in at most 2 malicious docs

# Quality gates
MAL_PRECISION_THRESHOLD = 0.995    # pattern must be ≥99.5% malicious to be a block sig
MAL_PREVALENCE_CAP_FRAC = 0.12     # if it appears in >12% of malicious docs, it's too generic
MAL_MIN_PATTERN_LEN = 10           # short patterns are too noisy
SAFE_PRECISION_THRESHOLD = 0.995   # pattern must be ≥99.5% safe to be an allow sig


def compute_vectorizer_params(n_total):
    """Return vectoriser hyper-parameters as a dict.

    These are intentionally generous — the real selectivity comes from
    the percentile-based extraction thresholds computed after fitting.
    """
    params = {
        "vec_min_df":       3,
        "vec_max_features": 500_000,
    }
    log.info("Vectoriser params (n=%d): %s", n_total, params)
    return params


def compute_extraction_thresholds(n_safe, n_mal, w_safe_df, w_mal_df):
    """Derive extraction thresholds from the actual document-frequency distributions.

    Rather than hard-coding thresholds for a specific dataset size, we look at
    percentile-based cut-offs that adapt automatically. The idea:

    - For malicious sigs: find the 35th-percentile support among malicious-only
      features as the minimum support. This filters out very rare patterns
      (probably noise) while keeping the common attack phrases.

    - For safe sigs: same idea but at the 40th percentile of safe-only features.

    Returns a dict consumed by build_malicious_signatures() and build_safe_signatures().
    """
    total_df = w_safe_df + w_mal_df
    with np.errstate(divide="ignore", invalid="ignore"):
        mal_precision = np.where(total_df > 0, w_mal_df / total_df, 0.0)

    # ── Malicious thresholds ─────────────────────────────────────────
    # Look at features appearing ONLY in malicious docs to set a support floor.
    # The 35th percentile weeds out rare noise while keeping real attack patterns.
    mal_only = (w_mal_df > 0) & (w_safe_df == 0)
    if mal_only.any():
        mal_min_support = max(DEFAULT_MAL_MIN_SUPPORT, int(np.percentile(w_mal_df[mal_only], 35)))
    else:
        mal_min_support = DEFAULT_MAL_MIN_SUPPORT

    # For high-precision malicious features (≥99%), how much safe-doc leakage is ok?
    # We use the 90th percentile — most features leak into 0-1 safe docs, so this
    # cap stays tight but adapts if the dataset has more overlap.
    high_prec = mal_precision >= 0.99
    if high_prec.any():
        mal_safe_df_cap = max(1, int(np.percentile(w_safe_df[high_prec], 90)))
    else:
        mal_safe_df_cap = 1

    # ── Safe thresholds ──────────────────────────────────────────────
    # Same logic but for safe-only features. 40th percentile keeps only
    # the reasonably common safe patterns (not one-off phrases).
    safe_only = (w_safe_df > 0) & (w_mal_df == 0)
    if safe_only.any():
        safe_min_support = max(DEFAULT_SAFE_MIN_SUPPORT, int(np.percentile(w_safe_df[safe_only], 40)))
    else:
        safe_min_support = DEFAULT_SAFE_MIN_SUPPORT

    thresholds = {
        # Malicious extraction
        "mal_min_support":          mal_min_support,
        "mal_safe_df_cap":          mal_safe_df_cap,
        "mal_precision_threshold":  MAL_PRECISION_THRESHOLD,
        "mal_prevalence_cap_frac":  MAL_PREVALENCE_CAP_FRAC,
        "mal_min_pattern_len":      MAL_MIN_PATTERN_LEN,

        # Safe extraction
        "safe_min_support":         safe_min_support,
        "safe_mal_df_cap":          DEFAULT_SAFE_MAL_DF_CAP,
        "safe_precision_threshold": SAFE_PRECISION_THRESHOLD,

        # Holdout: allow up to this many false positives on held-out safe text
        "holdout_fp_tolerance":     max(1, int(n_safe * 0.20 * 0.0005)),
    }

    log.info(
        "Data-driven thresholds (n_safe=%d, n_mal=%d):\n  %s",
        n_safe, n_mal,
        "\n  ".join(f"{k}: {v}" for k, v in thresholds.items()),
    )
    return thresholds
