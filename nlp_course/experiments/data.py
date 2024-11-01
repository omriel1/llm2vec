import os.path
from collections import namedtuple
from typing import List

import torch
from datasets import DatasetDict

from llm2vec import LLM2Vec
from nlp_course import BASE_DIR

EMBEDDINGS_DIR = BASE_DIR / "nlp_course" / "experiments" / "v1" / "embeddings"

TrainTestSplit = namedtuple("TrainTestSplit", field_names=["X_train", "y_train", "X_test", "y_test"])

def generate_llm2vec_embeddings(model: LLM2Vec, texts: List[str]) -> List[List[float]]:
    # Note that, "with toch.no_grad()" is included in LLM2Vec.encode
    embeddings = model.encode(texts)
    return embeddings

def generate_train_test_data(
        embedding_model: LLM2Vec,
        dataset: DatasetDict,
        target_column: str = "tag_ids"
) -> TrainTestSplit:
    """
    This method generates the train and test data.
    Generally, it'll load all the data (hebrew sentences) and apply the embedding model
    to generate an embedding for each sentence. This is the X_train/X_test.

    Also, it'll cache the results to save computation time.
    """
    train_dataset = dataset["train"]
    train_embeddings_file = EMBEDDINGS_DIR / "train.pt"
    if os.path.exists(train_embeddings_file.as_posix()):
        X_train = torch.load(train_embeddings_file.as_posix())
    else:  # calculate and save train embeddings
        X_train = generate_llm2vec_embeddings(embedding_model, train_dataset["text"])
        torch.save(X_train, train_embeddings_file)
    y_train = train_dataset[target_column]  # specific for HebSentiment dataset

    test_dataset = dataset["test"]
    test_embeddings_file = EMBEDDINGS_DIR / "test.pt"
    if os.path.exists(test_embeddings_file.as_posix()):
        X_test = torch.load(test_embeddings_file.as_posix())
    else:  # calculate and save train embeddings
        X_test = generate_llm2vec_embeddings(embedding_model, test_dataset["text"])
        torch.save(X_test, test_embeddings_file)
    y_test = test_dataset[target_column]

    return TrainTestSplit(
        X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
    )
