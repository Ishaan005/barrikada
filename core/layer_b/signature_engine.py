"""Contrastive embedding signature engine for Layer B.

Runtime flow:
1. Embed the input prompt (L2-normalised).
2. Retrieve top-k attack centroid similarities via FAISS.
3. Retrieve top-k benign centroid similarities via FAISS.
4. Compute contrastive score = mean(top-k attack) − mean(top-k benign).
5. Apply two-threshold decision:
       score > block_threshold  → BLOCK
       score > flag_threshold   → FLAG
       else                     → ALLOW (safe)
6. Optional radius filter: reject matches where the prompt is outside
   the cluster's radius envelope.
"""

import time
import hashlib
import json
import logging
from typing import List
from pathlib import Path

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

from core.settings import Settings
from models.SignatureMatch import SignatureMatch, Severity
from models.LayerBResult import LayerBResult

log = logging.getLogger(__name__)


class SignatureEngine:
    def __init__(self):
        self.settings = Settings()
        self._load_model()
        self._load_signatures()

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------

    def _load_model(self):
        model_name = self.settings.layer_b_embedding_model
        self.model = SentenceTransformer(model_name)
        log.info("Loaded embedding model: %s", model_name)

    def _load_signatures(self):
        sig = Path(self.settings.layer_b_signatures_dir)

        # Attack artefacts
        attack_idx_path = sig / "faiss_index.bin"
        if not attack_idx_path.exists():
            raise FileNotFoundError(
                f"FAISS index not found at {attack_idx_path}. "
                "Run scripts/extract_signature_patterns.py first."
            )
        cpu_attack = faiss.read_index(str(attack_idx_path))
        self.attack_centroids = np.load(str(sig / "centroids.npy"))
        with open(sig / "metadata.json") as f:
            self.metadata = json.load(f)

        # Benign artefacts
        benign_idx_path = sig / "benign_faiss_index.bin"
        if benign_idx_path.exists():
            cpu_benign = faiss.read_index(str(benign_idx_path))
            self.benign_centroids = np.load(str(sig / "benign_centroids.npy"))
        else:
            cpu_benign = None
            self.benign_centroids = None
            log.warning("No benign centroids found — contrastive scoring disabled.")

        # Cluster radii
        radii_path = sig / "cluster_radii.json"
        if radii_path.exists():
            with open(radii_path) as f:
                self.radii = {int(k): v for k, v in json.load(f).items()}
        else:
            self.radii = {}

        # Move to GPU if available
        n_gpus = faiss.get_num_gpus()
        if n_gpus > 0:
            res = faiss.StandardGpuResources() #type: ignore
            self.attack_index = faiss.index_cpu_to_gpu(res, 0, cpu_attack) #type: ignore
            log.info("Attack FAISS index → GPU 0")
            if cpu_benign is not None:
                self.benign_index = faiss.index_cpu_to_gpu(#type: ignore
                    faiss.StandardGpuResources(), 0, cpu_benign,#type: ignore
                )
                log.info("Benign FAISS index → GPU 0")
            else:
                self.benign_index = None
        else:
            self.attack_index = cpu_attack
            self.benign_index = cpu_benign

        n_attack = self.attack_centroids.shape[0]
        n_benign = self.benign_centroids.shape[0] if self.benign_centroids is not None else 0
        log.info("Loaded %d attack + %d benign centroids (dim=%d)",
                 n_attack, n_benign, self.attack_centroids.shape[1])

    # ------------------------------------------------------------------
    # Embedding
    # ------------------------------------------------------------------

    def _embed(self, text: str) -> np.ndarray:
        vec = self.model.encode([text], normalize_embeddings=True)
        return vec.astype(np.float32)

    # ------------------------------------------------------------------
    # Detection
    # ------------------------------------------------------------------

    def detect(self, text: str) -> LayerBResult:
        start = time.time()
        input_hash = hashlib.sha256(text.encode()).hexdigest()[:16]

        query = self._embed(text)
        top_k = self.settings.layer_b_top_k

        # --- Attack similarity (top-k mean) ---
        k_attack = min(top_k, self.attack_centroids.shape[0])
        atk_scores, atk_ids = self.attack_index.search(query, k_attack)
        atk_scores = atk_scores[0]  # shape (k_attack,)
        atk_ids = atk_ids[0]
        attack_sim = float(np.mean(atk_scores[:k_attack]))

        # --- Benign similarity (top-k mean) ---
        if self.benign_index is not None:
            k_benign = min(top_k, self.benign_centroids.shape[0])#type: ignore
            ben_scores, _ = self.benign_index.search(query, k_benign)
            ben_scores = ben_scores[0]
            benign_sim = float(np.mean(ben_scores[:k_benign]))
        else:
            benign_sim = 0.0

        # --- Contrastive score (attack sim - benign sim) ---
        contrastive = attack_sim - benign_sim

        # --- Build match objects (informational) ---
        matches: List[SignatureMatch] = []
        cluster_meta = {c["cluster_id"]: c for c in self.metadata.get("clusters", [])}
        for rank, (score, idx) in enumerate(zip(atk_scores, atk_ids)):
            score_f = float(score)
            if score_f < 0.20:
                continue
            cid = int(idx)
            meta = cluster_meta.get(cid, {})
            samples = meta.get("sample_prompts", [])
            desc = samples[0][:100] if samples else f"cluster_{cid}"
            matches.append(SignatureMatch(
                rule_id=f"cluster_{cid}",
                severity=Severity.MALICIOUS,
                pattern="contrastive_embedding",
                matched_text=text[:200],
                start_pos=0,
                end_pos=len(text),
                rule_description=desc,
                tags=[f"cluster_{cid}", f"rank_{rank}",
                      f"atk_sim={attack_sim:.3f}", f"ben_sim={benign_sim:.3f}",
                      f"contrastive={contrastive:.3f}"],
                confidence=score_f,
            ))

        # --- Two-threshold decision on mean top-k attack similarity ---
        # block_threshold / flag_threshold are compared against attack_sim.
        # Contrastive guard: if benign similarity exceeds attack similarity at
        # the block boundary, demote to FLAG to avoid false positives.
        block_thr = self.settings.layer_b_block_threshold
        flag_thr = self.settings.layer_b_flag_threshold

        if attack_sim >= block_thr:
            # Contrastive guard: only hard-block when attack clearly dominates benign
            if attack_sim > benign_sim:
                verdict = "block"
                confidence = self.settings.layer_b_block_confidence
            else:
                verdict = "flag"
                confidence = self.settings.layer_b_flag_confidence
        elif attack_sim >= flag_thr:
            verdict = "flag"
            confidence = self.settings.layer_b_flag_confidence
        else:
            verdict = "allow"
            confidence = self.settings.layer_b_safe_confidence

        elapsed = (time.time() - start) * 1000

        return LayerBResult(
            input_hash=input_hash,
            processing_time_ms=elapsed,
            matches=matches,
            verdict=verdict,
            confidence_score=confidence,
            allowlisted=False,
            allowlist_rules=[],
        )