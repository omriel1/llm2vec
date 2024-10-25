import json
from typing import List

import torch
from datasets import DatasetDict
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

from llm2vec import LLM2Vec
from nlp_course import BASE_DIR
from nlp_course.prepare_hebsentiment_data import load_hebsetiment_data
from nlp_course.utils import get_device


def generate_llm2vec_embeddings(model: LLM2Vec, texts: List[str]) -> List[List[float]]:
    embeddings = model.encode(texts)
    return embeddings


def train_and_evaluate_classifier(X_train, y_train, X_test, y_test, sklearn_classifier):
    sklearn_classifier.fit(X_train, y_train)
    predictions = sklearn_classifier.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    class_report = classification_report(y_test, predictions)

    print("Accuracy:", accuracy)
    print("Classification Report:\n", class_report)

    results = {
        "accuracy": accuracy,
        "classification_report": class_report,
    }
    return sklearn_classifier, results


def evaluate_llm2vec_embedding_model(embedding_model: LLM2Vec, dataset: DatasetDict):
    train_dataset = dataset["train"]
    X_train = generate_llm2vec_embeddings(embedding_model, train_dataset["text"])
    y_train = train_dataset["tag_ids"]  # specific for HebSentiment dataset

    test_dataset = dataset["test"]
    X_test = generate_llm2vec_embeddings(embedding_model, test_dataset["text"])
    y_test = test_dataset["tag_ids"]

    lr_clf = LogisticRegression(max_iter=3000, verbose=1)
    lr_clf, results = train_and_evaluate_classifier(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        sklearn_classifier=lr_clf,
    )
    return results


def main():
    peft_model_dir = (
        BASE_DIR / "output" / "mntp-simcse" / "dictalm2.0-instruct" / "checkpoint-1000"
    )

    l2v = LLM2Vec.from_pretrained(
        base_model_name_or_path="dicta-il/dictalm2.0-instruct",
        peft_model_name_or_path=peft_model_dir,
        device_map=get_device(),
        torch_dtype=torch.bfloat16,
    )

    hebsentiment_dataset = load_hebsetiment_data()
    results = evaluate_llm2vec_embedding_model(
        embedding_model=l2v, dataset=hebsentiment_dataset
    )

    with open("results.json", "w") as json_file:
        json.dump(results, json_file, indent=4)

    print("Results have been saved to classification_results.json")


if __name__ == "__main__":
    main()
