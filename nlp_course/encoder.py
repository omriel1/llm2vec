from abc import ABC, abstractmethod

import torch
from torch import nn


class BaseSentenceEncoder(ABC):
    """
    A class that represent a sentence-encoder model. That is,
    given a sentence, it'll output *a single* vector embedding that
    represents this sentence.
    """

    def __init__(self, model: nn.Module, model_name: str):
        self.model = model
        self.model_name = model_name

    @abstractmethod
    def encode(self, texts: list[str]) -> torch.Tensor:
        """
        Generate a single embedding vector for each sentence
        """
        raise NotImplementedError()


class LLM2VecEncoder(BaseSentenceEncoder):

    def encode(self, texts: list[str]) -> torch.Tensor:
        return self.model.encode(texts)
