import os

import numpy as np
from loguru import logger

from red_hela.adapters.persistence.artifact_loader import ArtifactLoader
from red_hela.infrastructure.index_utils import (
    TREE_CONFIDENCE_DEFAULT,
    quantize_vector,
    tree_score,
)

FRAUD_THRESHOLD = 0.6


class VectorSearch:
    def __init__(self, loader: ArtifactLoader) -> None:
        self._loader = loader
        self.vectors_q8 = loader.load_vectors_q8()
        self.vectors_f16 = loader.load_vectors_f16()
        self.labels = loader.load_labels()
        self.centroids = loader.load_centroids()
        self.cluster_indices = loader.load_cluster_indices()
        self.cluster_offsets = loader.load_cluster_offsets()
        self.tree = loader.load_tree()
        self.nprobe = int(os.environ.get("RED_HELA_IVF_NPROBE", "2"))
        self.batch_size = int(os.environ.get("RED_HELA_IVF_BATCH", "4096"))
        self.rerank_k = int(os.environ.get("RED_HELA_RERANK_K", "24"))
        self.tree_confidence = float(
            os.environ.get("RED_HELA_TREE_CONFIDENCE", str(TREE_CONFIDENCE_DEFAULT))
        )
        self.cell_fast_margin = float(os.environ.get("RED_HELA_CELL_FAST_MARGIN", "0.0"))
        self.cell_major_order = self._is_cell_major_order()
        self.cell_scores = self._build_cell_scores()
        logger.info(
            "search ready nprobe={} rerank_k={} tree_nodes={}",
            self.nprobe,
            self.rerank_k,
            0 if self.tree is None else self.tree["scores"].shape[0],
        )

    def warmup(self) -> None:
        _ = self.labels.shape[0]
        _ = self.centroids.shape[0]
        _ = self.cluster_offsets.shape[0]
        _ = self.cell_scores.shape[0]
        _ = self.vectors_q8[0, 0]
        _ = self.vectors_f16[0, 0]
        if self.tree is not None:
            _ = self.tree["scores"].shape[0]

    @staticmethod
    def _merge_top_k(
        current_ids: np.ndarray,
        current_distances: np.ndarray,
        candidate_ids: np.ndarray,
        candidate_distances: np.ndarray,
        k: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        if candidate_ids.size == 0:
            return current_ids, current_distances
        if current_ids.size == 0:
            return candidate_ids, candidate_distances
        merged_ids = np.concatenate((current_ids, candidate_ids))
        merged_distances = np.concatenate((current_distances, candidate_distances))
        take = min(k, merged_ids.shape[0])
        top_idx = np.argpartition(merged_distances, take - 1)[:take]
        ordered_idx = top_idx[np.argsort(merged_distances[top_idx])]
        return merged_ids[ordered_idx], merged_distances[ordered_idx]

    def _is_cell_major_order(self) -> bool:
        total = self.cluster_indices.shape[0]
        if total == 0:
            return True
        probe = min(1024, total - 1)
        return (
            int(self.cluster_indices[0]) == 0
            and int(self.cluster_indices[probe]) == probe
            and int(self.cluster_indices[-1]) == total - 1
        )

    def _build_cell_scores(self) -> np.ndarray:
        scores = np.full(self.centroids.shape[0], 0.5, dtype=np.float32)
        for cell in range(self.centroids.shape[0]):
            start = int(self.cluster_offsets[cell])
            end = int(self.cluster_offsets[cell + 1])
            if start == end:
                continue
            if self.cell_major_order:
                scores[cell] = float(np.mean(self.labels[start:end]))
            else:
                ids = self.cluster_indices[start:end]
                scores[cell] = float(np.mean(self.labels[ids]))
        return scores

    def _distance_batch(
        self,
        quantized_query: np.ndarray,
        batch_start: int,
        batch_end: int,
    ) -> np.ndarray:
        block = self.vectors_q8[batch_start:batch_end].astype(np.int16, copy=False)
        diff = block - quantized_query
        diff32 = diff.astype(np.int32, copy=False)
        return np.sum(diff32 * diff32, axis=1, dtype=np.int32)

    def _rerank_top_k(self, query: np.ndarray, candidate_ids: np.ndarray, k: int) -> np.ndarray:
        if candidate_ids.size == 0:
            return candidate_ids
        vectors = self.vectors_f16[candidate_ids].astype(np.float32, copy=False)
        diff = vectors - query
        distances = np.sum(diff * diff, axis=1, dtype=np.float32)
        take = min(k, candidate_ids.shape[0])
        top_idx = np.argpartition(distances, take - 1)[:take]
        ordered_idx = top_idx[np.argsort(distances[top_idx])]
        return candidate_ids[ordered_idx]

    def _score_ivf(self, query: np.ndarray, k: int) -> float:
        centroid_distances = np.sum((self.centroids - query) ** 2, axis=1)
        nearest = int(np.argmin(centroid_distances))
        nearest_score = float(self.cell_scores[nearest])
        if (
            nearest_score <= self.cell_fast_margin
            or nearest_score >= 1.0 - self.cell_fast_margin
        ):
            return nearest_score

        nprobe = min(self.nprobe, self.centroids.shape[0])
        probe_idx = np.argpartition(centroid_distances, nprobe - 1)[:nprobe]
        quantized_query = quantize_vector(query).astype(np.int16, copy=False)
        candidate_k = max(k, self.rerank_k)
        best_ids = np.array([], dtype=np.int32)
        best_distances = np.array([], dtype=np.int32)

        for cluster_id in probe_idx:
            start = int(self.cluster_offsets[cluster_id])
            end = int(self.cluster_offsets[cluster_id + 1])
            if end <= start:
                continue
            for batch_start in range(start, end, self.batch_size):
                batch_end = min(batch_start + self.batch_size, end)
                batch_distances = self._distance_batch(quantized_query, batch_start, batch_end)
                batch_len = batch_end - batch_start
                take = min(candidate_k, batch_len)
                top_idx = np.argpartition(batch_distances, take - 1)[:take]
                ordered_idx = top_idx[np.argsort(batch_distances[top_idx])]
                candidate_ids = np.arange(batch_start, batch_end, dtype=np.int32)[ordered_idx]
                best_ids, best_distances = self._merge_top_k(
                    best_ids,
                    best_distances,
                    candidate_ids,
                    batch_distances[ordered_idx],
                    candidate_k,
                )

        if best_ids.size == 0:
            return 1.0
        if self.rerank_k > k:
            best_ids = self._rerank_top_k(query, best_ids, k)
        return float(np.mean(self.labels[best_ids]))

    def score(self, query: np.ndarray, k: int = 5) -> tuple[float, bool]:
        query = query.astype(np.float32, copy=False)
        tree_result = tree_score(query, self.tree, confidence=self.tree_confidence)
        if tree_result is not None:
            return tree_result, tree_result < FRAUD_THRESHOLD
        fraud_score = self._score_ivf(query, k)
        return fraud_score, fraud_score < FRAUD_THRESHOLD
