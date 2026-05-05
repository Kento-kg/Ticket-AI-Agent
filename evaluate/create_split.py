import json
import logging
from pathlib import Path

from datasets import ClassLabel, Dataset, DatasetDict, load_dataset

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_PATH = PROJECT_ROOT / "data" / "processed" / "dataset.json"
TEST_PATH = PROJECT_ROOT / "data" / "processed" / "holdout.json"

def load_dataset_from_hf() -> None:
    ds = load_dataset('kentokamg/ticket-dataset')

    label_names = sorted(set(ds['train']['category']))
    label_feature = ClassLabel(names=label_names)

    ds = ds['train'].cast_column('category', label_feature)
    ds = ds.train_test_split(test_size=0.2, stratify_by_column='category', seed=42)
    train_split = ds['train']
    test_split = ds['test']

    _save_split(train_split, DATASET_PATH, label_names)
    _save_split(test_split, TEST_PATH, label_names)
    print(f'Tickets guardados en {DATASET_PATH} y {TEST_PATH}')

def _save_split(split: Dataset, path: Path, label_names: list[str]) -> None:
    """Converts a split to JSON, converting ClassLabel int back to string."""
    path.parent.mkdir(parents=True, exist_ok=True)
    records = [
        {
            "text": row["text"],
            "category": label_names[row["category"]],
            "urgency": row["urgency"],
            "team": row["team"],
        }
        for row in split
    ]
    with path.open("w") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    load_dataset_from_hf()
