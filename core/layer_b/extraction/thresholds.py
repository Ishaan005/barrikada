"""Threshold computation for signature extraction."""
import logging

log = logging.getLogger(__name__)

# Extraction constants — confirmed equivalent to the previous percentile-based
# adaptive logic by ablation testing on barrikada.csv (2026-03-10).
MAL_MIN_SUPPORT = 20            # pattern must appear in ≥20 malicious docs
MAL_SAFE_DF_CAP = 1             # pattern can appear in at most 1 safe doc
MAL_PRECISION_THRESHOLD = 0.995 # ≥99.5% malicious
MAL_PREVALENCE_CAP_FRAC = 0.12  # skip if in >12% of malicious docs (too generic)
MAL_MIN_PATTERN_LEN = 10        # discard very short patterns

SAFE_MIN_SUPPORT = 25           # pattern must appear in ≥25 safe docs
SAFE_MAL_DF_CAP = 2             # allow at most 2 malicious doc appearances
SAFE_PRECISION_THRESHOLD = 0.995 # ≥99.5% safe


def compute_vectorizer_params(n_total):
    """Return vectoriser hyper-parameters as a dict."""
    params = {
        "vec_min_df":       3,
        "vec_max_features": 500_000,
    }
    log.info("Vectoriser params (n=%d): %s", n_total, params)
    return params


def compute_extraction_thresholds(n_safe, n_mal, w_safe_df=None, w_mal_df=None):
    """Return extraction thresholds as a dict.

    Parameters are accepted for call-site compatibility but are not used —
    ablation testing confirmed the fixed constants above are equivalent to
    the previous percentile-based adaptive logic on this dataset.
    """
    thresholds = {
        "mal_min_support":          MAL_MIN_SUPPORT,
        "mal_safe_df_cap":          MAL_SAFE_DF_CAP,
        "mal_precision_threshold":  MAL_PRECISION_THRESHOLD,
        "mal_prevalence_cap_frac":  MAL_PREVALENCE_CAP_FRAC,
        "mal_min_pattern_len":      MAL_MIN_PATTERN_LEN,
        "safe_min_support":         SAFE_MIN_SUPPORT,
        "safe_mal_df_cap":          SAFE_MAL_DF_CAP,
        "safe_precision_threshold": SAFE_PRECISION_THRESHOLD,
    }
    log.info("Extraction thresholds: %s", thresholds)
    return thresholds
