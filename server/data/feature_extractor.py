from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from server.data.dataset_loader import load_or_extract_embeddings, TASK_CONFIGS


def pre_extract_all_tasks() -> None:
    for task_id in TASK_CONFIGS.keys():
        print(f"[feature_extractor] Processing {task_id}...")
        pool_emb, pool_labels, val_emb, val_labels = load_or_extract_embeddings(task_id)
        print(
            f"[feature_extractor] {task_id}: pool={pool_emb.shape}, val={val_emb.shape}"
        )
    print("[feature_extractor] All embeddings cached successfully.")


if __name__ == "__main__":
    pre_extract_all_tasks()
