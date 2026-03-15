"""Build contrastive embedding signatures for Layer B.

Produces:
  core/layer_b/signatures/embeddings/
    centroids.npy          — attack centroids (purity-filtered)
    faiss_index.bin        — FAISS IP index over attack centroids
    benign_centroids.npy   — benign centroids
    benign_faiss_index.bin — FAISS IP index over benign centroids
    cluster_radii.json     — per-cluster radius for radius filtering
    metadata.json          — build metadata

Dataset: datasets/barrikada.csv  (columns: text, label  — 0=safe, 1=malicious)
"""

import gc
import logging
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.layer_b.extraction.dataset import load_dataset
from core.layer_b.extraction.embedding_builder import (
    encode_prompts,
    cluster_embeddings,
    build_centroids,
    compute_cluster_purity,
    compute_cluster_radii,
    filter_clusters_by_purity,
    build_faiss_index,
    collect_metadata,
    save_artifacts,
)
from core.settings import Settings

log = logging.getLogger(__name__)

DATASET_CSV = Path("datasets/barrikada.csv")
OUTDIR = Path("core/layer_b/signatures/embeddings")


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
    n_safe = int((labels == 0).sum())
    n_mal = int((labels == 1).sum())
    log.info("Dataset: %d samples (safe=%d, malicious=%d)", len(labels), n_safe, n_mal)

    injection_mask = labels == 1
    benign_mask = labels == 0
    injection_texts = texts[injection_mask].tolist()
    benign_texts = texts[benign_mask].tolist()

    del texts, labels
    gc.collect()

    # Choose encoding model
    # Use trained signature encoder if available, else fall back to base model
    sig_encoder_path = Path(settings.layer_b_signatures_dir) / "signature_encoder"
    if sig_encoder_path.exists():
        model_name = str(sig_encoder_path)
        log.info("Using trained signature encoder: %s", sig_encoder_path)
    else:
        model_name = settings.layer_b_embedding_model
        log.info("Using base model: %s", model_name)
    log.info("Encoding injection prompts …")
    attack_embeddings, _ = encode_prompts(injection_texts, model_name)
    gc.collect()

    log.info("Encoding benign prompts …")
    benign_embeddings, _ = encode_prompts(benign_texts, model_name)
    gc.collect()

    # Cluster attack prompts
    n_clusters = settings.layer_b_n_clusters
    log.info("Clustering %d attack embeddings (k=%d) …", len(attack_embeddings), n_clusters)
    attack_labels, _ = cluster_embeddings(attack_embeddings, n_clusters=n_clusters)
    gc.collect()

    attack_cdata = build_centroids(attack_embeddings, attack_labels, n_clusters)
    attack_centroids = attack_cdata["centroids"]
    attack_ids = attack_cdata["cluster_ids"]
    attack_sizes = attack_cdata["cluster_sizes"]

    # Cluster benign prompts
    benign_k = max(16, n_clusters // 2)
    log.info("Clustering %d benign embeddings (k=%d) …", len(benign_embeddings), benign_k)
    benign_labels, _ = cluster_embeddings(benign_embeddings, n_clusters=benign_k)
    gc.collect()

    benign_cdata = build_centroids(benign_embeddings, benign_labels, benign_k)
    benign_centroids = benign_cdata["centroids"]

    # Purity filtering
    purity = compute_cluster_purity(
        attack_labels, benign_embeddings,
        attack_centroids, attack_ids,
        proximity_threshold=settings.layer_b_purity_proximity,
    )
    radii = compute_cluster_radii(attack_embeddings, attack_labels,
                                  attack_centroids, attack_ids)

    min_purity = settings.layer_b_min_cluster_purity
    attack_centroids, attack_ids, attack_sizes, radii = filter_clusters_by_purity(
        attack_centroids, attack_ids, attack_sizes, purity, radii, min_purity,
    )

    log.info("Final: %d attack centroids, %d benign centroids (dim=%d)",
             len(attack_ids), benign_centroids.shape[0], attack_centroids.shape[1])

    # FAISS indices
    attack_index = build_faiss_index(attack_centroids)
    benign_index = build_faiss_index(benign_centroids)

    # Metadata
    metadata = collect_metadata(
        attack_ids, attack_sizes, attack_labels, injection_texts,
        model_name, n_clusters, purity, radii,
    )

    # Save
    save_artifacts(OUTDIR, attack_centroids, attack_index, metadata,
                   benign_centroids, benign_index, radii)

    elapsed = time.perf_counter() - wall_start
    log.info("Done. %d attack clusters, wall time: %.1fs", len(attack_ids), elapsed)


if __name__ == "__main__":
    main()
