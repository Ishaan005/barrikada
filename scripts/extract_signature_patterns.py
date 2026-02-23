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
from scipy.sparse import csr_matrix

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.layer_b.extraction.dataset        import load_dataset
from core.layer_b.extraction.thresholds     import compute_thresholds
from core.layer_b.extraction.vectorise      import make_vectorizers, doc_freq
from core.layer_b.extraction.safe_sigs      import build_safe_signatures
from core.layer_b.extraction.malicious_sigs import build_malicious_signatures
from core.layer_b.extraction.yara_writer    import write_yara_rules
from core.layer_b.extraction.holdout_filter import filter_signatures_with_holdout

log = logging.getLogger(__name__)

DATASET_CSV = Path("datasets/barrikada.csv")
OUTDIR      = Path("core/layer_b/signatures/extracted")

# ---------------------------------------------------------------------------
# Backwards-compatibility re-exports (tests / notebooks may import these)
# ---------------------------------------------------------------------------
from core.layer_b.extraction.thresholds  import compute_thresholds as _compute_thresholds  # noqa: F401


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main() -> None:
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

    # --- Stratified split: 80% extract, 20% holdout for validation ---
    from sklearn.model_selection import train_test_split

    idx_extract, idx_holdout = train_test_split(
        np.arange(n_total), test_size=0.20, random_state=42, stratify=labels,
    )
    texts_extract  = texts[idx_extract]
    labels_extract = labels[idx_extract]
    texts_holdout  = texts[idx_holdout]
    labels_holdout = labels[idx_holdout]

    holdout_safe_texts = texts_holdout[labels_holdout == 0].tolist()
    log.info(
        "Split: extract=%d, holdout=%d (holdout safe=%d)",
        len(idx_extract), len(idx_holdout), len(holdout_safe_texts),
    )

    n_safe_ext  = int((labels_extract == 0).sum())
    n_mal_ext   = int((labels_extract == 1).sum())
    n_total_ext = len(labels_extract)

    thresholds = compute_thresholds(n_total_ext, n_safe_ext, n_mal_ext)

    texts_norm = texts_extract.tolist()
    del texts, texts_extract, texts_holdout
    gc.collect()

    # Fit the word vectoriser once; share DF arrays across SAFE and MALICIOUS phases.
    word_vec, _ = make_vectorizers(
        min_df=thresholds["vec_min_df"],
        max_features=thresholds["vec_max_features"],
    )
    log.info("Fitting word vectoriser …")
    Xw: csr_matrix = word_vec.fit_transform(texts_norm)  # type: ignore[assignment]
    w_vocab = word_vec.get_feature_names_out()
    log.info("  vocab=%d features, matrix=%s, nnz=%d", len(w_vocab), Xw.shape, Xw.nnz)

    y = labels_extract.astype(int)
    w_safe_df = doc_freq(Xw[y == 0])
    w_mal_df  = doc_freq(Xw[y == 1])
    del Xw, word_vec
    gc.collect()

    safe_signatures = build_safe_signatures(thresholds, w_vocab, w_safe_df, w_mal_df)
    gc.collect()

    malicious_signatures = build_malicious_signatures(
        texts_norm, labels_extract, thresholds, w_vocab, w_safe_df, w_mal_df
    )
    gc.collect()

    # --- Hold-out validation: remove rules that fire on held-out safe text ---
    log.info("Running holdout validation on %d safe holdout texts …", len(holdout_safe_texts))

    malicious_signatures = filter_signatures_with_holdout(
        malicious_signatures,
        holdout_safe_texts,
        rule_prefix="DATA_HIGH_",
        meta_keys=["precision", "support", "safe_support", "type"],
        max_safe_hits=0,
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
