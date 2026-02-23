"""MALICIOUS block-signature extraction for Layer B."""
import gc
import logging
import time
from typing import Dict, List

import numpy as np
from scipy.sparse import csr_matrix

from models.PatternStats import PatternStats
from core.layer_b.extraction.vectorise import make_vectorizers, doc_freq, is_generic_pattern

log = logging.getLogger(__name__)

# Minimum ratio of malicious hits to total hits for a block-signature.
MAL_PRECISION_THRESHOLD = 0.95


def build_malicious_signatures(
    texts_norm: List[str],
    labels: np.ndarray,
    thresholds: Dict[str, int],
    w_vocab: np.ndarray,
    w_safe_df: np.ndarray,
    w_mal_df: np.ndarray,
):
    t0 = time.perf_counter()

    min_df        = thresholds["vec_min_df"]
    max_features  = thresholds["vec_max_features"]
    mal_min_support = thresholds["mal_min_support"]
    mal_safe_df_cap = thresholds["mal_safe_df_cap"]

    y         = labels.astype(int)
    safe_mask = y == 0
    mal_mask  = y == 1

    log.info("Reusing word DF arrays (vocab=%d) for MALICIOUS signatures", len(w_vocab))

    # --- Char n-grams ---
    _, char_vec = make_vectorizers(min_df=min_df, max_features=max_features)
    log.info("Fitting char vectoriser for MALICIOUS signatures …")
    Xc: csr_matrix = char_vec.fit_transform(texts_norm)  # type: ignore[assignment]
    c_vocab = char_vec.get_feature_names_out()
    log.info("  char vocab=%d, nnz=%d", len(c_vocab), Xc.nnz)

    c_safe_df = doc_freq(Xc[safe_mask])
    c_mal_df  = doc_freq(Xc[mal_mask])
    del Xc
    gc.collect()

    # --- Merge word + char arrays for vectorised filtering ---
    all_patterns = np.concatenate([w_vocab,    c_vocab])
    all_safe_df  = np.concatenate([w_safe_df,  c_safe_df])
    all_mal_df   = np.concatenate([w_mal_df,   c_mal_df])

    del c_vocab, c_safe_df, c_mal_df
    gc.collect()

    total_df = all_safe_df + all_mal_df
    with np.errstate(divide="ignore", invalid="ignore"):
        mal_prec = np.where(total_df > 0, all_mal_df / total_df, 0.0)

    # np.char.str_len requires fixed-width Unicode; use vectorize(len) for object arrays.
    pat_len = np.vectorize(len)(all_patterns)

    # Cap: if a pattern appears in >30% of malicious docs it's generic English,
    # not an attack-specific indicator.
    mal_prevalence_cap = int(0.30 * mal_mask.sum())

    mask = (
        (all_mal_df  >= mal_min_support)
        & (all_mal_df  <= mal_prevalence_cap)
        & (mal_prec  >= MAL_PRECISION_THRESHOLD)
        & (all_safe_df <= mal_safe_df_cap)
        & (pat_len   >= 8)
    )
    idxs = np.nonzero(mask)[0]
    log.info("  %d / %d features pass vectorised pre-filter", len(idxs), len(all_patterns))

    selected = [
        PatternStats(
            pattern=str(all_patterns[i]),
            safe_df=int(all_safe_df[i]),
            mal_df=int(all_mal_df[i]),
        )
        for i in idxs
        if not is_generic_pattern(str(all_patterns[i]))
    ]
    selected.sort(key=lambda s: (s.mal_precision, s.mal_df, -len(s.pattern)), reverse=True)

    out = [
        {
            "pattern":      st.pattern,
            "precision":    float(round(st.mal_precision, 4)),
            "support":      int(st.mal_df),
            "safe_support": int(st.safe_df),
            "type":         "block-high",
        }
        for st in selected
    ]
    log.info("MALICIOUS signatures: %d selected (%.1fs)", len(out), time.perf_counter() - t0)
    return out
