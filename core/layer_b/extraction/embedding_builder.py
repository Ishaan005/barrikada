"""Embedding-based contrastive signature builder for Layer B.

Builds both attack and benign centroid signatures so the runtime engine can
compute a contrastive score:  attack_similarity − benign_similarity.

Pipeline:
1. Encode all prompts with a sentence-transformer model.
2. L2-normalise embeddings (unit vectors → inner-product == cosine sim).
3. Cluster attack prompts with FAISS GPU k-means.
4. Cluster benign prompts with FAISS GPU k-means.
5. Filter attack clusters by purity (remove noisy clusters).
6. Compute per-cluster radius for radius filtering at runtime.
7. Build FAISS IndexFlatIP indices for both sets of centroids.
8. Persist all artifacts.
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Encoding
# ---------------------------------------------------------------------------

def encode_prompts(texts, model_name, batch_size=256):
    """Encode texts into L2-normalised float32 embeddings.

    Returns (embeddings, model).
    """
    model = SentenceTransformer(model_name)
    embeddings = model.encode(
        texts, batch_size=batch_size, show_progress_bar=True, normalize_embeddings=True,
    )
    embeddings = embeddings.astype(np.float32)
    # Belt-and-suspenders: ensure unit-length
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    embeddings = embeddings / norms
    log.info("Encoded %d texts with %s (dim=%d)", len(texts), model_name, embeddings.shape[1])
    return embeddings, model


# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------

def cluster_embeddings(embeddings, n_clusters=64, n_iter=30, use_gpu=True):
    """FAISS GPU k-means on L2-normalised embeddings.

    Returns (labels 1-D ndarray, centroids float32 ndarray).
    """
    n, dim = embeddings.shape
    kmeans = faiss.Kmeans(
        dim,
        n_clusters,
        niter=n_iter,
        verbose=True,
        gpu=use_gpu and faiss.get_num_gpus() > 0,
        spherical=True,
    )
    kmeans.train(embeddings.astype(np.float32))
    _, labels = kmeans.index.search(embeddings.astype(np.float32), 1) #type: ignore
    labels = labels.ravel()

    sizes = np.bincount(labels, minlength=n_clusters)
    non_empty = int((sizes > 0).sum())
    log.info(
        "K-means: k=%d, %d non-empty, min/med/max size = %d/%d/%d",
        n_clusters, non_empty,
        int(sizes[sizes > 0].min()),
        int(np.median(sizes[sizes > 0])),
        int(sizes.max()),
    )
    return labels, kmeans.centroids.copy().astype(np.float32) #type: ignore


# ---------------------------------------------------------------------------
# Centroid computation + purity + radius
# ---------------------------------------------------------------------------

def build_centroids(embeddings, labels, n_clusters):
    """L2-normalise centroids computed from original embeddings.

    Returns dict with centroids, cluster_ids, cluster_sizes.
    """
    centroids, ids, sizes = [], [], []
    for cid in range(n_clusters):
        mask = labels == cid
        count = int(mask.sum())
        if count == 0:
            continue
        c = np.mean(embeddings[mask], axis=0)
        norm = np.linalg.norm(c)
        if norm > 0:
            c = c / norm
        centroids.append(c)
        ids.append(cid)
        sizes.append(count)
    return {
        "centroids": np.array(centroids, dtype=np.float32),
        "cluster_ids": ids,
        "cluster_sizes": sizes,
    }


def compute_cluster_purity(attack_labels, attack_embeddings,
                           benign_embeddings, centroids_array, cluster_ids,
                           proximity_threshold=0.50):
    """For each attack cluster, measure what fraction of nearby prompts
    are truly attack text.

    Only benign prompts with cosine similarity >= *proximity_threshold* to
    a centroid are counted — distant benign prompts don't dilute purity.

    Returns dict  cluster_id → purity (float 0-1).
    """
    if benign_embeddings is None or len(benign_embeddings) == 0:
        return {cid: 1.0 for cid in cluster_ids}

    # Build temp index from attack centroids
    dim = centroids_array.shape[1]
    idx = faiss.IndexFlatIP(dim)
    idx.add(centroids_array) #type: ignore

    # For each benign prompt, find its nearest attack centroid + similarity
    ben_scores, ben_assigned = idx.search(benign_embeddings.astype(np.float32), 1) #type: ignore
    ben_scores = ben_scores.ravel()
    ben_assigned = ben_assigned.ravel()

    purity = {}
    for i, cid in enumerate(cluster_ids):
        n_attack = int((attack_labels == cid).sum())
        # Only count benign prompts genuinely close to this centroid
        close_mask = (ben_assigned == i) & (ben_scores >= proximity_threshold)
        n_benign = int(close_mask.sum())
        total = n_attack + n_benign
        purity[cid] = n_attack / total if total > 0 else 1.0
    return purity


def compute_cluster_radii(embeddings, labels, centroids_array, cluster_ids):
    """Mean cosine distance from each centroid to its members.

    Returns dict  cluster_id → radius (float).
    """
    radii = {}
    for i, cid in enumerate(cluster_ids):
        mask = labels == cid
        members = embeddings[mask]
        if len(members) == 0:
            radii[cid] = 0.0
            continue
        sims = members @ centroids_array[i]  # cosine sim (unit vecs)
        radii[cid] = float(1.0 - np.mean(sims))  # distance = 1 - sim
    return radii


def filter_clusters_by_purity(centroids_array, cluster_ids, cluster_sizes,
                               purity, radii, min_purity=0.90):
    """Remove clusters below the purity threshold.

    Returns filtered (centroids, cluster_ids, cluster_sizes, radii).
    """
    keep = [i for i, cid in enumerate(cluster_ids) if purity.get(cid, 0) >= min_purity]
    removed = len(cluster_ids) - len(keep)
    if removed:
        log.info("Purity filtering: removed %d/%d clusters (purity < %.2f)",
                 removed, len(cluster_ids), min_purity)
    return (
        centroids_array[keep],
        [cluster_ids[i] for i in keep],
        [cluster_sizes[i] for i in keep],
        {cluster_ids[i]: radii[cluster_ids[i]] for i in keep},
    )


# ---------------------------------------------------------------------------
# FAISS index
# ---------------------------------------------------------------------------

def build_faiss_index(centroids):
    """IndexFlatIP over L2-normalised centroids (inner product == cosine)."""
    dim = centroids.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(centroids) #type: ignore
    log.info("FAISS index: %d vectors, dim=%d", index.ntotal, dim)
    return index


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------

def collect_metadata(cluster_ids, cluster_sizes, labels, texts,
                     model_name, n_clusters, purity, radii):
    """Build metadata dict for serialisation."""
    clusters_info = []
    for cid, size in zip(cluster_ids, cluster_sizes):
        mask = labels == cid
        indices = np.where(mask)[0]
        samples = [texts[i][:200] for i in indices[:5]]
        clusters_info.append({
            "cluster_id": int(cid),
            "size": int(size),
            "purity": round(purity.get(cid, 1.0), 4),
            "radius": round(radii.get(cid, 0.0), 6),
            "sample_prompts": samples,
        })
    return {
        "model_name": model_name,
        "n_clusters_requested": n_clusters,
        "n_clusters_after_purity": len(cluster_ids),
        "total_injections": int(len(labels)),
        "build_timestamp": datetime.now(timezone.utc).isoformat(),
        "clusters": clusters_info,
    }


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_artifacts(output_dir, attack_centroids, attack_index, attack_metadata,
                   benign_centroids, benign_index, radii):
    """Persist all artifacts (attack + benign centroids, FAISS indices, metadata)."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    np.save(out / "centroids.npy", attack_centroids)
    faiss.write_index(attack_index, str(out / "faiss_index.bin"))
    with open(out / "metadata.json", "w") as f:
        json.dump(attack_metadata, f, indent=2)

    np.save(out / "benign_centroids.npy", benign_centroids)
    faiss.write_index(benign_index, str(out / "benign_faiss_index.bin"))

    # Radii as JSON for easy loading
    with open(out / "cluster_radii.json", "w") as f:
        json.dump({str(k): v for k, v in radii.items()}, f, indent=2)

    log.info("Saved attack centroids      -> %s", out / "centroids.npy")
    log.info("Saved attack FAISS index    -> %s", out / "faiss_index.bin")
    log.info("Saved benign centroids      -> %s", out / "benign_centroids.npy")
    log.info("Saved benign FAISS index    -> %s", out / "benign_faiss_index.bin")
    log.info("Saved cluster radii         -> %s", out / "cluster_radii.json")
    log.info("Saved metadata              -> %s", out / "metadata.json")
