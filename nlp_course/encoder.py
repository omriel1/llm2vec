from abc import ABC, abstractmethod
from pathlib import Path

import torch
from sentence_transformers import SentenceTransformer

from llm2vec import LLM2Vec
from nlp_course import BASE_DIR
from nlp_course.utils import get_device


class BaseSentenceEncoder(ABC):
    """
    A class that represent a sentence-encoder model. That is,
    given a sentence, it'll output *a single* vector embedding that
    represents this sentence.
    """

    def __init__(self, model_name: str):
        self.model_name = model_name

    @abstractmethod
    def encode(self, texts: list[str]) -> torch.Tensor:
        """
        Generate a single embedding vector for each sentence
        """
        raise NotImplementedError()


class LLM2VecEncoder(BaseSentenceEncoder):
    default_base_model = "dicta-il/dictalm2.0-instruct"
    default_peft_model = (
        BASE_DIR / "output" / "mntp-simcse" / "dictalm2.0-instruct" / "checkpoint-1000"
    )

    def __init__(
        self,
        model_name: str,
        base_model_name_or_path: str | Path | None = None,
        peft_model_name_or_path: str | Path | None = None,
    ):
        super().__init__(model_name)
        self.base_model = base_model_name_or_path or self.default_base_model
        self.peft_model = peft_model_name_or_path or self.default_peft_model
        self.model = None

    def _initialize(self) -> None:
        if self.model is None:
            self.model = LLM2Vec.from_pretrained(
                base_model_name_or_path=self.base_model,
                peft_model_name_or_path=self.peft_model,
                device_map=get_device(),
                torch_dtype=torch.bfloat16,
            )

    def encode(self, texts: list[str]) -> torch.Tensor:
        self._initialize()  # apply lazy loading
        return self.model.encode(texts)


class SentenceTransformersEncoder(BaseSentenceEncoder):

    def __init__(self, model_name: str):
        super().__init__(model_name=model_name)
        self.model = None

    def _initialize(self) -> None:
        if self.model is None:
            self.model: SentenceTransformer = SentenceTransformer(self.model_name)

    def encode(self, texts: list[str]) -> torch.Tensor:
        self._initialize()  # apply lazy loading
        embeddings = self.model.encode(texts, show_progress_bar=True)
        return torch.from_numpy(embeddings)


class PLMBERTBaserEncoder(BaseSentenceEncoder):
    """
    Stands for "Pre-trained Language model BERT based", which essentially mean
    we're using an encoder which is a variation of BERT, and the assumption is that
    it's regular encoder model, that is returns *vector for each token*.
    """

    pass


if __name__ == "__main__":
    sbert = SentenceTransformersEncoder(
        model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    )
    sentences = ["היי מה קורה?", "הכל טוב מלך?"]
    embeddings = sbert.encode(sentences)
    print(embeddings)
