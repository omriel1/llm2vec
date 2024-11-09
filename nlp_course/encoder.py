from abc import ABC, abstractmethod
from pathlib import Path

import torch
from sentence_transformers import SentenceTransformer
from transformers import BertModel, BertTokenizerFast
from llm2vec import LLM2Vec
from nlp_course import BASE_DIR
from nlp_course.utils import get_device
from tqdm import tqdm


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
    # default_base_model = "dicta-il/dictalm2.0-instruct"
    default_base_model = (
        BASE_DIR / "output" / "mntp" / "dictalm2.0-instruct"
    )
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


class PLMBERTBasedEncoder(BaseSentenceEncoder):
    """
    Stands for "Pre-trained Language model BERT based", which essentially mean
    we're using an encoder which is a variation of BERT, and the assumption is that
    it's regular encoder model, that is returns *vector for each token*.
    """

    def __init__(self, model_name: str):
        super().__init__(model_name=model_name)
        self.model = None
        self.tokenizer = None

    def _initialize(self) -> None:
        if (self.model is None) or (self.tokenizer is None):
            self.tokenizer = BertTokenizerFast.from_pretrained(self.model_name)
            model = BertModel.from_pretrained(self.model_name)
            model.eval()
            self.model = model

    def encode(self, texts: list[str], batch_size: int = 32) -> torch.Tensor:
        self._initialize()  # apply lazy loading
        embeddings = []

        device = get_device()
        self.model.to(device)

        for i in tqdm(range(0, len(texts), batch_size), desc="Encoding sentences"):
            batch_texts = texts[i : i + batch_size]
            inputs = self.tokenizer(
                batch_texts, return_tensors="pt", padding=True, truncation=True
            ).to(device)
            with torch.no_grad():
                outputs = self.model(**inputs)
                batch_embeddings = outputs.last_hidden_state[:, 0, :]  # get [CLS] vector
                embeddings.append(batch_embeddings.cpu())

        return torch.cat(embeddings, dim=0)


if __name__ == "__main__":
    encoder = PLMBERTBasedEncoder(model_name="imvladikon/alephbertgimmel-base-512")
    sentences = ["אני ממש שמח לראות אותך!", "הרסת לי את כל היום", "איזה כיף לנו!!!"]
    embeddings = encoder.encode(sentences)
    print(embeddings)
