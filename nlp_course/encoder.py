from abc import ABC, abstractmethod
from sentence_transformers import SentenceTransformer

import torch
from torch import nn


class BaseSentenceEncoder(ABC):
    """
    A class that represent a sentence-encoder model. That is,
    given a sentence, it'll output *a single* vector embedding that
    represents this sentence.
    """

    def __init__(self, model_name: str, model: nn.Module | None = None):
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


class SentenceTransformersEncoder(BaseSentenceEncoder):

    def __init__(self, model_name: str):
        model: SentenceTransformer = SentenceTransformer(model_name)
        super().__init__(model=model, model_name=model_name)

    def encode(self, texts: list[str]) -> torch.Tensor:
        embeddings = self.model.encode(texts)
        return torch.from_numpy(embeddings)


class PLMBERTBaserEncoder(BaseSentenceEncoder):
    """
    Stands for "Pre-trained Language model BERT based", which essentially mean
    we're using an encoder which is a variation of BERT, and the assumption is that
    it's regular encoder model, that is returns *vector for each token*.
    """
    pass



if __name__ == "__main__":
    sbert = SentenceTransformersEncoder(model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2")
    sentences = ["היי מה קורה?", "הכל טוב מלך?"]
    embeddings =sbert.encode(sentences)
    print(embeddings)
