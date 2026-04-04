from __future__ import annotations

from typing import List

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import label_binarize


def compute_binary_auc(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    pos_label: str = None,
    classes: List[str] = None,
) -> float:
    if len(np.unique(y_true)) < 2:
        return 0.5

    try:
        if len(classes) == 2:
            pos = classes[1] if classes else pos_label
            binary_true = (y_true == pos).astype(int)
            return float(roc_auc_score(binary_true, y_scores[:, 1]))
        else:
            y_bin = label_binarize(y_true, classes=classes)
            return float(
                roc_auc_score(y_bin, y_scores, multi_class="ovr", average="macro")
            )
    except Exception:
        return 0.5


def compute_uncertainty_entropy(proba: np.ndarray) -> float:
    clipped = np.clip(proba, 1e-10, 1.0)
    entropy = -np.sum(clipped * np.log(clipped), axis=1)
    max_entropy = np.log(proba.shape[1])
    return float(np.mean(entropy / max_entropy)) if max_entropy > 0 else 0.0


def per_sample_uncertainty(proba: np.ndarray) -> np.ndarray:
    clipped = np.clip(proba, 1e-10, 1.0)
    entropy = -np.sum(clipped * np.log(clipped), axis=1)
    max_entropy = np.log(proba.shape[1])
    return (entropy / max_entropy) if max_entropy > 0 else entropy


def cosine_similarity_to_set(
    embedding: np.ndarray, labeled_embeddings: np.ndarray
) -> float:
    if labeled_embeddings.shape[0] == 0:
        return 0.0
    mean_labeled = labeled_embeddings.mean(axis=0)
    norm_emb = np.linalg.norm(embedding)
    norm_mean = np.linalg.norm(mean_labeled)
    if norm_emb < 1e-10 or norm_mean < 1e-10:
        return 0.0
    return float(np.dot(embedding, mean_labeled) / (norm_emb * norm_mean))


def per_sample_diversity(
    pool_embeddings: np.ndarray, labeled_embeddings: np.ndarray
) -> np.ndarray:
    if labeled_embeddings.shape[0] == 0:
        return np.ones(pool_embeddings.shape[0])
    mean_labeled = labeled_embeddings.mean(axis=0)
    pool_norms = np.linalg.norm(pool_embeddings, axis=1, keepdims=True)
    mean_norm = np.linalg.norm(mean_labeled)
    safe_pool_norms = np.where(pool_norms < 1e-10, 1.0, pool_norms)
    safe_mean_norm = mean_norm if mean_norm > 1e-10 else 1.0
    cosines = pool_embeddings.dot(mean_labeled) / (
        safe_pool_norms.squeeze() * safe_mean_norm
    )
    return 1.0 - np.clip(cosines, -1.0, 1.0)


def compute_mean_diversity_score(
    pool_embeddings: np.ndarray, labeled_embeddings: np.ndarray
) -> float:
    scores = per_sample_diversity(pool_embeddings, labeled_embeddings)
    return float(np.mean(scores))
