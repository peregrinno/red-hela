from pathlib import Path

import numpy as np
from loguru import logger


class ArtifactLoader:
    def __init__(self, resources_dir: Path) -> None:
        self._resources_dir = resources_dir

    def _path(self, name: str) -> Path:
        return self._resources_dir / name

    def _load_optional(self, name: str) -> np.ndarray | None:
        path = self._path(name)
        if not path.exists():
            return None
        return np.load(path, mmap_mode="r")

    def load_vectors_q8(self) -> np.ndarray:
        path = self._path("vectors_q8.npy")
        logger.info("mmap {}", path)
        return np.load(path, mmap_mode="r")

    def load_vectors_f16(self) -> np.ndarray:
        path = self._path("vectors_f16.npy")
        logger.info("mmap {}", path)
        return np.load(path, mmap_mode="r")

    def load_labels(self) -> np.ndarray:
        return np.load(self._path("labels.npy"), mmap_mode="r")

    def load_centroids(self) -> np.ndarray:
        return np.load(self._path("centroids.npy"), mmap_mode="r")

    def load_cluster_indices(self) -> np.ndarray:
        return np.load(self._path("cluster_indices.npy"), mmap_mode="r")

    def load_cluster_offsets(self) -> np.ndarray:
        return np.load(self._path("cluster_offsets.npy"), mmap_mode="r")

    def load_tree(self) -> dict[str, np.ndarray] | None:
        path = self._path("tree.npz")
        if not path.exists():
            return None
        tree_file = np.load(path)
        return {key: tree_file[key] for key in tree_file.files}
