from __future__ import annotations

import os
import pickle
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

EMBEDDING_DIM = 512
EMBEDDINGS_CACHE_DIR = Path(__file__).parent / "embeddings_cache"
EMBEDDINGS_CACHE_DIR.mkdir(exist_ok=True)

TASK_CONFIGS = {
    "task1": {
        "classes": ["normal", "pneumonia"],
        "pool_size": 200,
        "val_size": 50,
        "class_weights": [0.5, 0.5],
    },
    "task2": {
        "classes": ["normal", "pneumonia", "covid", "tuberculosis"],
        "pool_size": 400,
        "val_size": 80,
        "class_weights": [0.4, 0.3, 0.2, 0.1],
    },
    "task3": {
        "classes": ["normal", "nodule", "effusion", "pneumothorax"],
        "pool_size": 600,
        "val_size": 100,
        "class_weights": [0.85, 0.05, 0.05, 0.05],
    },
}


def _generate_synthetic_embeddings(
    task_id: str, seed: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.RandomState(seed)
    config = TASK_CONFIGS[task_id]
    classes = config["classes"]
    pool_size = config["pool_size"]
    val_size = config["val_size"]
    class_weights = config["class_weights"]
    n_classes = len(classes)
    total_size = pool_size + val_size

    class_centers = rng.randn(n_classes, EMBEDDING_DIM) * 3
    all_embeddings = []
    all_labels = []

    for idx, (cls_name, weight) in enumerate(zip(classes, class_weights)):
        n_samples = int(total_size * weight)
        noise = rng.randn(n_samples, EMBEDDING_DIM) * 0.8
        embeddings = class_centers[idx] + noise
        all_embeddings.append(embeddings)
        all_labels.extend([cls_name] * n_samples)

    all_embeddings_arr = np.vstack(all_embeddings).astype(np.float32)
    total_actual = all_embeddings_arr.shape[0]
    actual_pool_size = min(pool_size, total_actual - val_size)

    perm = rng.permutation(total_actual)
    pool_idx = perm[:actual_pool_size]
    val_idx = perm[actual_pool_size : actual_pool_size + val_size]

    pool_embeddings = all_embeddings_arr[pool_idx]
    pool_labels = np.array(all_labels)[pool_idx]
    val_embeddings = all_embeddings_arr[val_idx]
    val_labels = np.array(all_labels)[val_idx]

    return pool_embeddings, pool_labels, val_embeddings, val_labels


def _try_load_from_huggingface(
    task_id: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    try:
        from datasets import load_dataset
        import torch
        import torchvision.models as models
        import torchvision.transforms as transforms
        from PIL import Image
        import io
        import base64

        dataset = load_dataset(
            "keremberke/chest-xray-classification", name="full", trust_remote_code=True
        )

        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        model.fc = torch.nn.Identity()
        model.eval()

        transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        config = TASK_CONFIGS[task_id]
        classes = config["classes"]
        pool_size = config["pool_size"]
        val_size = config["val_size"]

        split = dataset.get("train", dataset.get("train"))
        if split is None:
            return None

        hf_label_map = {0: "normal", 1: "pneumonia"}

        embeddings_list = []
        labels_list = []

        max_samples = min(pool_size + val_size, len(split))
        for i in range(max_samples):
            try:
                sample = split[i]
                img = sample.get("image") or sample.get("img")
                if img is None:
                    continue
                if not isinstance(img, Image.Image):
                    img = Image.fromarray(img).convert("RGB")
                else:
                    img = img.convert("RGB")

                tensor = transform(img).unsqueeze(0)
                with torch.no_grad():
                    emb = model(tensor).squeeze().numpy()

                raw_label = sample.get("label", 0)
                label_str = hf_label_map.get(int(raw_label), "normal")
                if label_str not in classes:
                    label_str = "normal"

                embeddings_list.append(emb)
                labels_list.append(label_str)
            except Exception:
                continue

        if len(embeddings_list) < 50:
            return None

        embeddings_arr = np.array(embeddings_list, dtype=np.float32)
        labels_arr = np.array(labels_list)
        rng = np.random.RandomState(42)
        perm = rng.permutation(len(embeddings_arr))
        actual_pool = min(pool_size, len(perm) - val_size)
        pool_idx = perm[:actual_pool]
        val_idx = perm[actual_pool : actual_pool + val_size]

        return (
            embeddings_arr[pool_idx],
            labels_arr[pool_idx],
            embeddings_arr[val_idx],
            labels_arr[val_idx],
        )
    except Exception:
        return None


def load_or_extract_embeddings(
    task_id: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    cache_path = EMBEDDINGS_CACHE_DIR / f"{task_id}_embeddings.pkl"

    if cache_path.exists():
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    result = _try_load_from_huggingface(task_id)
    if result is None:
        result = _generate_synthetic_embeddings(task_id)

    with open(cache_path, "wb") as f:
        pickle.dump(result, f)

    return result


def get_task_classes(task_id: str) -> List[str]:
    return TASK_CONFIGS[task_id]["classes"]


def get_task_class_weights(task_id: str) -> List[float]:
    return TASK_CONFIGS[task_id]["class_weights"]
