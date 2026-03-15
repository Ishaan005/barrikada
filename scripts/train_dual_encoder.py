"""Train dual-encoder models for the Layer B signature engine.

Produces two specialised SentenceTransformer checkpoints:

  core/layer_b/signatures/embeddings/prompt_encoder/
  core/layer_b/signatures/embeddings/signature_encoder/

After training, rebuild signatures so the new encoders are used:

  python scripts/extract_signature_patterns.py

The extraction script and the runtime SignatureEngine will
automatically detect and load the trained encoders.

Usage:
  python scripts/train_dual_encoder.py
"""

import gc
import logging
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.layer_b.extraction.dataset import load_dataset
from core.layer_b.extraction.dual_encoder_trainer import train_dual_encoder
from core.settings import Settings

DATASET_CSV = Path("datasets/barrikada.csv")


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    wall_start = time.perf_counter()
    settings = Settings()

    # Load dataset
    texts, labels = load_dataset(DATASET_CSV)
    mal_texts = texts[labels == 1].tolist()
    safe_texts = texts[labels == 0].tolist()
    del texts, labels
    gc.collect()

    logging.info("Dataset: %d malicious, %d safe", len(mal_texts), len(safe_texts))

    # Train dual encoders
    prompt_path, sig_path = train_dual_encoder(mal_texts, safe_texts, settings)

    elapsed = time.perf_counter() - wall_start
    logging.info("Training complete in %.1fs", elapsed)
    logging.info("  Prompt encoder:    %s", prompt_path)
    logging.info("  Signature encoder: %s", sig_path)
    logging.info("")
    logging.info("Next step — rebuild signatures with the trained encoders:")
    logging.info("  python scripts/extract_signature_patterns.py")


if __name__ == "__main__":
    main()
