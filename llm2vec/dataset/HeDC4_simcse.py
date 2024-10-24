import logging
from itertools import islice
from typing import Optional, Dict, Any, List

import pandas as pd
from datasets import load_dataset

from llm2vec.dataset.dataset import Dataset, TrainSample, DataSample
from langchain_text_splitters import TextSplitter, RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)

class HeDC4(Dataset):
    """
    Load https://huggingface.co/datasets/HeNLP/HeDC4 and preprocess it in order to make it
    suitable for SimCSE training. In particular, this class:
    1. Loads the original text
    2. Filter "bad" texts (currently Nones)
    3. Chunk these texts into smaller pieces

    Regarding the chunking settings. Looking at the data used for this phase in the original paper:
    https://huggingface.co/datasets/princeton-nlp/datasets-for-simcse/viewer?row=67, seems like most of the
    sentences where up to 257 characters length. Yet, seems like a lot of sentences where shorter. Hence, we'll
    chunk with a max_size of 250.
    """
    chunk_max_size: int = 250

    def __init__(
            self,
            dataset_name: str = "HeNLP/HeDC4",
            split: str = "train",
            file_path: str = "",
            streaming: bool = True,
            dataset_start_index: int = 0,
            dataset_limit: Optional[int] = 100,
    ):
        self.dataset_name = dataset_name
        self.split = split
        self.file_path = file_path
        self.streaming = streaming
        self.start_index = dataset_start_index
        self.limit = dataset_limit
        self.chunker = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_max_size,
            chunk_overlap=0,
            separators=["\n\n", "(?<=\. )", "\n", " ", ""],
            length_function=len,
        )

        self.data = []
        self._load_data()


    def __len__(self):
        return len(self.data)

    def _load_data(self, file_path: str = None):
        ds = load_dataset(
            self.dataset_name,
            split=self.split,
            streaming=self.streaming,
        )
        dataset_list = list(
            islice(ds, self.start_index, self.limit)
        )
        logger.info(f"Loaded {len(dataset_list)} rows from HeDC4")
        raw_texts = [d["text"] for d in dataset_list]
        texts = self._preprocess(raw_texts)
        logger.info(f"After preprocessing a total of {len(texts)} rows.")
        id_ = self.start_index
        for text in texts:
            self.data.append(
                DataSample(
                    id_=id_,
                    query=text,
                    positive=text,
                )
            )
            id_ += 1

    def __getitem__(self, index):
        sample = self.data[index]
        if self.split == "train":
            return TrainSample(texts=[sample.query, sample.positive], label=1.0)
        else:
            raise ValueError("HeDC4 has 'train' split only.")

    def _preprocess(self, data: List[str]) -> List[str]:
        filtered_texts = self._filter_none(data)
        chunked_texts = self._split_texts(filtered_texts, self.chunker)
        return chunked_texts

    @staticmethod
    def _filter_none_text(texts: List[str]) -> List[str]:
        return [t for t in texts if t is not None]

    @staticmethod
    def _split_texts(texts: List[str], chunker: TextSplitter) -> List[str]:
        chunked_texts = []
        for text in texts:
            chunked_texts.extend(chunker.split_text(text))
        return chunked_texts


if __name__ == "__main__":
    chunker = RecursiveCharacterTextSplitter(
            chunk_size=HeDC4.chunk_max_size,
            chunk_overlap=0,
            separators=["\n\n", "(?<=\. )", "\n", " ", ""],
            length_function=len,
        )
    path = "./hedc4_sample.parquet"
    df = pd.read_parquet(path)

    raw_texts = df["text"].to_list()
    filtered_texts = HeDC4._filter_none_text(raw_texts)
    preprocessed_texts = HeDC4._split_texts(filtered_texts, chunker)

    print(len(preprocessed_texts))
    print(preprocessed_texts)

