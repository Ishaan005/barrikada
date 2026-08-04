"""Load the Barrikade dataset for signature extraction."""

import logging

import pandas as pd


log = logging.getLogger(__name__)


def load_dataset(csv_path):
    """Read a CSV with 'text' and 'label' columns and return (texts, labels)."""
    log.info("Loading dataset from %s …", csv_path)
    df = pd.read_csv(csv_path)

    texts = df["text"].fillna("")
    labels = df["label"].astype(int).to_numpy()
    log.info("  %d rows", len(df))
    return texts, labels
