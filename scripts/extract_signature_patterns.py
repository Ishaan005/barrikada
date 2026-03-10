"""Extract YARA signatures from the Barrikada dataset.

Thin orchestrator — all logic lives in core/layer_b/extraction/.

- Input:   datasets/barrikada.csv
- Outputs: core/layer_b/signatures/extracted/
    - safe_allow_signatures.yar
    - malicious_block_high_signatures.yar

Dataset requirements:
- Column `label`: 0 = SAFE, 1 = MALICIOUS
- Text column: `prompt` (preferred) or `text` (fallback)
"""

import gc
import logging
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.layer_b.extraction.dataset        import load_dataset
from core.layer_b.extraction.thresholds     import compute_vectorizer_params, compute_extraction_thresholds
from core.layer_b.extraction.vectorise      import make_vectorizers, doc_freq
from core.layer_b.extraction.safe_sigs      import build_safe_signatures
from core.layer_b.extraction.malicious_sigs import build_malicious_signatures
from core.layer_b.extraction.yara_writer    import write_yara_rules

log = logging.getLogger(__name__)

DATASET_CSV = Path("datasets/barrikada.csv")
OUTDIR      = Path("core/layer_b/signatures/extracted")

# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )
    wall_start = time.perf_counter()

    texts, labels = load_dataset(DATASET_CSV)

    n_safe  = int((labels == 0).sum())
    n_mal   = int((labels == 1).sum())
    n_total = len(labels)
    log.info("Loaded %d samples (safe=%d, malicious=%d)", n_total, n_safe, n_mal)

    vec_params = compute_vectorizer_params(n_total)

    texts_norm = texts.tolist()
    del texts
    gc.collect()

    # Fit the word vectoriser once; share DF arrays across SAFE and MALICIOUS phases.
    word_vec, _ = make_vectorizers(
        min_df=vec_params["vec_min_df"],
        max_features=vec_params["vec_max_features"],
    )
    log.info("Fitting word vectoriser …")
    
    Xw = word_vec.fit_transform(texts_norm)
    w_vocab = word_vec.get_feature_names_out()
    log.info("  vocab=%d features, matrix=%s, nnz=%d", len(w_vocab), Xw.shape, Xw.nnz) # type: ignore[union-attr]

    y = labels.astype(int)
    w_safe_df = doc_freq(Xw[y == 0]) # type: ignore[index]
    w_mal_df  = doc_freq(Xw[y == 1]) # type: ignore[index]
    del Xw, word_vec
    gc.collect()

    thresholds = compute_extraction_thresholds(n_safe, n_mal, w_safe_df, w_mal_df)
    
    # Carry vec params forward so downstream modules can reuse them
    thresholds.update(vec_params)

    safe_signatures = build_safe_signatures(thresholds, w_vocab, w_safe_df, w_mal_df)
    gc.collect()

    malicious_signatures = build_malicious_signatures(
        texts_norm, labels, thresholds, w_vocab, w_safe_df, w_mal_df
    )
    gc.collect()

    safe_yar = OUTDIR / "safe_allow_signatures.yar"
    mal_yar  = OUTDIR / "malicious_block_high_signatures.yar"

    write_yara_rules(
        safe_yar,
        rule_prefix="SAFE_ALLOW_",
        signatures=safe_signatures,
        meta_keys=["safe_precision", "support", "malicious_support", "type"],
    )
    write_yara_rules(
        mal_yar,
        rule_prefix="DATA_HIGH_",
        signatures=malicious_signatures,
        meta_keys=["precision", "support", "safe_support", "type"],
    )

    elapsed = time.perf_counter() - wall_start
    log.info("Wrote %d SAFE YARA rules     -> %s", len(safe_signatures),      safe_yar)
    log.info("Wrote %d MALICIOUS YARA rules -> %s", len(malicious_signatures), mal_yar)
    log.info("Total wall time: %.1fs", elapsed)


if __name__ == "__main__":
    main()
