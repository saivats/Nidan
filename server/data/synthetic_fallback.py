from __future__ import annotations

import numpy as np
from sklearn.datasets import make_classification
from typing import Tuple, List


EMBEDDING_DIM = 512


def generate_synthetic_dataset(
    n_samples: int,
    classes: List[str],
    class_weights: List[float],
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.RandomState(seed)
    n_classes = len(classes)
    embeddings_list = []
    labels_list = []

    class_centers = rng.randn(n_classes, EMBEDDING_DIM) * 4.0
    class_sigmas = [0.6 + rng.rand() * 0.4 for _ in range(n_classes)]

    for idx, (cls_name, weight) in enumerate(zip(classes, class_weights)):
        n_cls_samples = max(2, int(n_samples * weight))
        noise = rng.randn(n_cls_samples, EMBEDDING_DIM) * class_sigmas[idx]
        embeddings = class_centers[idx] + noise
        embeddings_list.append(embeddings)
        labels_list.extend([cls_name] * n_cls_samples)

    embeddings_arr = np.vstack(embeddings_list).astype(np.float32)
    labels_arr = np.array(labels_list)
    perm = rng.permutation(len(labels_arr))
    return embeddings_arr[perm], labels_arr[perm]
