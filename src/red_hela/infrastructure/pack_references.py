import gzip
import os
from pathlib import Path

import ijson
import numpy as np
from loguru import logger

from red_hela.infrastructure.index_utils import quantize_vectors, train_confident_tree
from red_hela.infrastructure.logging_setup import configure_logging

CHUNK_SIZE = 250_000


def pack_references(resources_dir: Path) -> None:
    configure_logging()
    references_path = resources_dir / "references.json.gz"
    if not references_path.exists():
        raise FileNotFoundError(references_path)

    vectors_path = resources_dir / "vectors.npy"
    labels_path = resources_dir / "labels.npy"
    vectors_f16_path = resources_dir / "vectors_f16.npy"
    vectors_q8_path = resources_dir / "vectors_q8.npy"
    tree_path = resources_dir / "tree.npz"

    n_vectors = 0
    with gzip.open(references_path, "rb") as handle:
        for _ in ijson.items(handle, "item"):
            n_vectors += 1
    logger.info("packing {} reference vectors", n_vectors)

    vectors_mmap = np.lib.format.open_memmap(
        vectors_path,
        mode="w+",
        dtype=np.float32,
        shape=(n_vectors, 14),
    )
    labels_mmap = np.lib.format.open_memmap(
        labels_path,
        mode="w+",
        dtype=np.uint8,
        shape=(n_vectors,),
    )

    index = 0
    with gzip.open(references_path, "rb") as handle:
        for ref in ijson.items(handle, "item"):
            vectors_mmap[index] = np.array(ref["vector"], dtype=np.float32)
            labels_mmap[index] = 1 if ref["label"] == "fraud" else 0
            index += 1
            if index % CHUNK_SIZE == 0:
                logger.info("streamed {}/{}", index, n_vectors)
    vectors_mmap.flush()
    labels_mmap.flush()
    del vectors_mmap, labels_mmap

    vectors = np.load(vectors_path, mmap_mode="r")
    labels = np.load(labels_path, mmap_mode="r")
    vectors_f16 = np.asarray(vectors, dtype=np.float16)
    vectors_q8 = quantize_vectors(np.asarray(vectors, dtype=np.float32))
    tree = train_confident_tree(np.asarray(vectors, dtype=np.float32), np.asarray(labels))

    np.save(vectors_f16_path, vectors_f16)
    np.save(vectors_q8_path, vectors_q8)
    np.savez_compressed(tree_path, **tree)
    logger.info(
        "saved f16={} q8={} tree_nodes={}",
        vectors_f16_path.name,
        vectors_q8_path.name,
        tree["scores"].shape[0],
    )


def _resolve_root() -> Path:
    configured = os.environ.get("RED_HELA_ROOT")
    if configured:
        return Path(configured)
    candidates = [
        Path("/app"),
        Path(__file__).resolve().parents[3],
    ]
    for candidate in candidates:
        if (candidate / "resources" / "references.json.gz").exists():
            return candidate
    return Path(__file__).resolve().parents[3]


if __name__ == "__main__":
    pack_references(_resolve_root() / "resources")
