import os.path
from collections import namedtuple
from pathlib import Path

import torch
from datasets import DatasetDict

from nlp_course.encoder import BaseSentenceEncoder

TrainTestSplit = namedtuple(
    "TrainTestSplit", field_names=["X_train", "y_train", "X_test", "y_test"]
)


class ExperimentDataGenerator:
    """
    A class which is responsible for creating the data for the sentiment analysis experiment.
    Generally, the class:
    1. Gets a dataset (with some splits)
    2. Applies the encoder onto the dataset to get the embeddings

    It's also capable of caching the results for efficiency
    """

    def __init__(
        self,
        encoder: BaseSentenceEncoder,
        dataset: DatasetDict,
        dataset_name: str,
        base_cache_dir: str | None,
    ):
        self.encoder = encoder
        self.dataset = dataset
        self.dataset_name = dataset_name

        if not base_cache_dir:
            self.cache_dir = None
        else:
            cache_dir = (
                Path(base_cache_dir)
                / dataset_name
                / self._get_model_name_for_caching_folder()
            )
            cache_dir.mkdir(exist_ok=True)
            self.cache_dir = cache_dir

    def generate_train_test_data(
        self, text_column: str = "text", target_column: str = "tag_ids"
    ):
        X_train, y_train = self._get_embeddings_and_labels(
            dataset_split="train", text_column=text_column, target_column=target_column
        )
        X_test, y_test = self._get_embeddings_and_labels(
            dataset_split="test", text_column=text_column, target_column=target_column
        )
        return TrainTestSplit(
            X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
        )

    def _get_embeddings_and_labels(
        self,
        dataset_split: str,
        text_column: str = "text",
        target_column: str = "tag_ids",
    ):
        embeddings = self._get_embeddings(dataset_split, text_column)
        labels = self.dataset[dataset_split][target_column]
        return embeddings, labels

    def _get_embeddings(
        self, dataset_split: str, text_column: str = "text"
    ) -> torch.Tensor:
        embeddings = self._load_embeddings_for_dataset_split_from_cache(dataset_split)
        if embeddings is not None:
            return embeddings

        embeddings = self.encoder.encode(self.dataset[dataset_split][text_column])
        self._put_embeddings_for_dataset_split_in_cache(embeddings, dataset_split)

        return embeddings

    def _load_embeddings_for_dataset_split_from_cache(
        self, dataset_split: str
    ) -> torch.Tensor | None:
        if self.cache_dir is None:
            return None
        cache_file = self._get_cache_file_name(dataset_split)
        if not os.path.exists(cache_file):
            return None
        embeddings = torch.load(cache_file)
        return embeddings

    def _put_embeddings_for_dataset_split_in_cache(
        self, embeddings: torch.Tensor, dataset_split: str
    ) -> None:
        if self.cache_dir:
            torch.save(embeddings, self._get_cache_file_name(dataset_split))

    def _get_cache_file_name(self, dataset_split: str) -> str | None:
        file_name = None
        if self.cache_dir:
            file_prefix = self.cache_dir / dataset_split
            file_name = file_prefix.as_posix() + ".pt"
        return file_name

    def _get_model_name_for_caching_folder(self) -> str:
        model_name = self.encoder.model_name
        return model_name.replace("/", "-")
