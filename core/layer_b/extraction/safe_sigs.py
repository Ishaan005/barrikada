"""SAFE allow-signature extraction for Layer B."""
import logging
import time
from typing import Dict, List

import numpy as np

from models.PatternStats import PatternStats
from core.layer_b.extraction.vectorise import is_stopwordy_ngram

log = logging.getLogger(__name__)

# Minimum ratio of safe hits to total hits for an allow-signature.
SAFE_ALLOW_PRECISION_THRESHOLD = 0.98


def build_safe_signatures(
    thresholds: Dict[str, int],
    w_vocab: np.ndarray,
    w_safe_df: np.ndarray,
    w_mal_df: np.ndarray,
):
    """Return a list of high-precision SAFE allow-signature dicts.

    Signatures enable early termination for clearly safe prompts, avoiding
    unnecessary compute in downstream ML layers.  Rules must be:
      - common in safe prompts  (>= safe_min_support hits)
      - extremely rare in malicious prompts (<= safe_mal_df_cap hits)
      - above the precision threshold
    """
    t0 = time.perf_counter()

    safe_min_support = thresholds["safe_min_support"]
    safe_mal_df_cap  = thresholds["safe_mal_df_cap"]
    safe_top_k       = thresholds["safe_top_k"]

    # Vectorised pre-filter before the Python loop
    mask = (w_safe_df >= safe_min_support) & (w_mal_df <= safe_mal_df_cap)
    idxs = np.nonzero(mask)[0]
    log.info("  %d / %d features pass support & DF-cap pre-filter", len(idxs), len(w_vocab))

    candidates: List[PatternStats] = []
    for i in idxs:
        pattern = str(w_vocab[i])
        s_df    = int(w_safe_df[i])
        m_df    = int(w_mal_df[i])

        if is_stopwordy_ngram(pattern):
            continue
        precision = s_df / (s_df + m_df) if (s_df + m_df) else 0.0
        if precision < SAFE_ALLOW_PRECISION_THRESHOLD:
            continue

        candidates.append(PatternStats(pattern, s_df, m_df))

    candidates.sort(key=lambda x: (x.safe_precision, x.safe_df, -len(x.pattern)), reverse=True)
    candidates = candidates[:safe_top_k]

    out = [
        {
            "pattern":           c.pattern,
            "support":           c.safe_df,
            "malicious_support": c.mal_df,
            "safe_precision":    float(round(c.safe_precision, 4)),
            "type":              "allow-safe",
        }
        for c in candidates
    ]
    log.info("SAFE signatures: %d selected (%.1fs)", len(out), time.perf_counter() - t0)
    return out
