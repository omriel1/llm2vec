"""
Data was downloaded from https://huggingface.co/datasets/HebArabNlpProject/HebrewSentiment
"""

import os.path
from typing import Union

import datasets
from datasets import load_dataset, Dataset, DatasetDict

from nlp_course import BASE_DIR

BASE_DATA_DIR = BASE_DIR / "nlp_course" / "sentiment_data"
OUTPUT_DIR = BASE_DATA_DIR / "HebSentiment"


def prepare_hebsetiment_data() -> None:
    if os.path.exists(OUTPUT_DIR.as_posix()):
        return

    dataset = datasets.DatasetDict()

    for split in ["train", "test", "validation"]:
        data_dir = BASE_DATA_DIR / "original_files" / f"HebSentiment_{split}.jsonl"
        data = load_dataset(
            "json",
            data_files=data_dir.as_posix(),
        )
        dataset[split] = data["train"]

    dataset.save_to_disk(OUTPUT_DIR.as_posix())


def load_hebsetiment_data() -> Union[Dataset, DatasetDict]:
    dataset = datasets.load_from_disk(OUTPUT_DIR)
    return dataset


if __name__ == "__main__":
    load_hebsetiment_data()
