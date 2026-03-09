"""Load the Barrikada dataset for signature extraction."""
import logging
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)


def load_dataset(csv_path):
    """Read a CSV with 'text' and 'label' columns and return (texts, labels)."""
    log.info("Loading dataset from %s …", csv_path)
    df = pd.read_csv(csv_path)

    # Git LFS pointers have a single column starting with "version https://git-lfs"
    first_col = str(df.columns[0]) if len(df.columns) else ""
    if first_col.startswith("version https://git-lfs"):
        raise RuntimeError(
            f"{csv_path} is a Git LFS pointer and has not been downloaded.\n"
            "Run:  git lfs pull --include='datasets/barrikada.csv'\n"
            "(Install git-lfs first if needed: conda install -c conda-forge git-lfs)"
        )

    texts = df["text"].fillna("")
    labels = df["label"].astype(int).to_numpy()
    log.info("  %d rows", len(df))
    return texts, labels
