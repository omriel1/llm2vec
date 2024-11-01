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


def prepare_hebsentiment_data() -> None:
    if os.path.exists(OUTPUT_DIR.as_posix()):
        return

    def strip_tag_ids(row):
        """
        Fix an issue where there exists "Neutral" and "Neutral " labels.
        """
        row["tag_ids"] = row["tag_ids"].strip()
        return row

    dataset = datasets.DatasetDict()

    for split in ["train", "test", "validation"]:
        data_dir = BASE_DATA_DIR / "original_files" / f"HebSentiment_{split}.jsonl"
        data = load_dataset(
            "json",
            data_files=data_dir.as_posix(),
        )
        data["train"] = data["train"].map(strip_tag_ids)
        dataset[split] = data["train"]

    dataset.save_to_disk(OUTPUT_DIR.as_posix())


def load_hebsetiment_data() -> Union[Dataset, DatasetDict]:
    dataset = datasets.load_from_disk(OUTPUT_DIR)
    return dataset


if __name__ == "__main__":
    prepare_hebsentiment_data()
    # ds = load_hebsetiment_data()
    # print(ds)
