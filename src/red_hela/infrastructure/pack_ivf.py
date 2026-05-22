import gc
import os
from pathlib import Path

import numpy as np
from loguru import logger
from sklearn.cluster import KMeans

from red_hela.infrastructure.logging_setup import configure_logging

N_CLUSTERS_DEFAULT = 1024
BATCH_SIZE_DEFAULT = 8192
RANDOM_STATE_DEFAULT = 42
SAMPLE_SIZE_DEFAULT = 400_000
INIT_ITERATIONS_DEFAULT = 80


def assign_centroids_batched(
    vectors: np.ndarray,
    centroids: np.ndarray,
    batch_size: int = BATCH_SIZE_DEFAULT,
) -> np.ndarray:
    n_vectors = vectors.shape[0]
    assignments = np.empty(n_vectors, dtype=np.int32)
    centroid_norms = np.sum(centroids * centroids, axis=1, dtype=np.float32)
    for start in range(0, n_vectors, batch_size):
        end = min(start + batch_size, n_vectors)
        batch = np.asarray(vectors[start:end], dtype=np.float32)
        batch_norms = np.sum(batch * batch, axis=1, dtype=np.float32)
        distances = batch_norms[:, None] + centroid_norms[None, :] - 2.0 * (batch @ centroids.T)
        assignments[start:end] = np.argmin(distances, axis=1)
    return assignments


def pack_ivf(resources_dir: Path) -> None:
    configure_logging()
    n_clusters = int(os.environ.get("RED_HELA_IVF_CLUSTERS", str(N_CLUSTERS_DEFAULT)))
    batch_size = int(os.environ.get("RED_HELA_IVF_BATCH_SIZE", str(BATCH_SIZE_DEFAULT)))
    random_state = int(os.environ.get("RED_HELA_IVF_SEED", str(RANDOM_STATE_DEFAULT)))
    sample_size_limit = int(
        os.environ.get("RED_HELA_IVF_SAMPLE_SIZE", str(SAMPLE_SIZE_DEFAULT))
    )
    init_iterations = int(
        os.environ.get("RED_HELA_IVF_INIT_ITERATIONS", str(INIT_ITERATIONS_DEFAULT))
    )

    vectors_f32 = np.load(resources_dir / "vectors.npy", mmap_mode="r")
    vectors_q8 = np.load(resources_dir / "vectors_q8.npy", mmap_mode="r")
    vectors_f16 = np.load(resources_dir / "vectors_f16.npy", mmap_mode="r")
    labels = np.load(resources_dir / "labels.npy", mmap_mode="r")
    n_vectors = vectors_f32.shape[0]

    sample_size = min(n_vectors, sample_size_limit)
    rng = np.random.default_rng(random_state)
    sample_ids = rng.choice(n_vectors, size=sample_size, replace=False)
    sample = np.asarray(vectors_f32[sample_ids], dtype=np.float32)

    logger.info("kmeans clusters={} sample={}", n_clusters, sample_size)
    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init=10,
        max_iter=init_iterations,
        algorithm="lloyd",
    )
    kmeans.fit(sample)
    del sample
    gc.collect()
    centroids = kmeans.cluster_centers_.astype(np.float32)

    logger.info("assigning {} vectors to clusters", n_vectors)
    cluster_ids = assign_centroids_batched(vectors_f32, centroids, batch_size=batch_size)

    logger.info("reordering by cluster (cell-major)")
    order = np.argsort(cluster_ids, kind="stable")
    counts = np.bincount(cluster_ids, minlength=n_clusters)
    cluster_offsets = np.zeros(n_clusters + 1, dtype=np.int64)
    cluster_offsets[1:] = np.cumsum(counts, dtype=np.int64)
    contiguous_indices = np.arange(n_vectors, dtype=np.int32)

    np.save(resources_dir / "centroids.npy", centroids)
    np.save(resources_dir / "cluster_indices.npy", contiguous_indices)
    np.save(resources_dir / "cluster_offsets.npy", cluster_offsets)

    logger.info("reordering vectors_q8")
    np.save(resources_dir / "vectors_q8.npy", np.asarray(vectors_q8[order], dtype=np.uint8))
    del vectors_q8
    gc.collect()

    logger.info("reordering vectors_f16")
    np.save(resources_dir / "vectors_f16.npy", np.asarray(vectors_f16[order], dtype=np.float16))
    del vectors_f16
    gc.collect()

    logger.info("reordering labels")
    np.save(resources_dir / "labels.npy", np.asarray(labels[order], dtype=np.uint8))
    del labels
    gc.collect()

    logger.info("reordering vectors_f32")
    np.save(resources_dir / "vectors.npy", np.asarray(vectors_f32[order], dtype=np.float32))
    logger.info("ivf ready clusters={} cell_major=True", n_clusters)


def _resolve_root() -> Path:
    configured = os.environ.get("RED_HELA_ROOT")
    if configured:
        return Path(configured)
    return Path(__file__).resolve().parents[3]


if __name__ == "__main__":
    pack_ivf(_resolve_root() / "resources")
