import argparse
import json
from typing import List, Any

import torch
from datasets import DatasetDict
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler

from llm2vec import LLM2Vec
from nlp_course import BASE_DIR
from nlp_course.experiments.data import generate_train_test_data
from nlp_course.prepare_hebsentiment_data import load_hebsetiment_data
from nlp_course.utils import get_device

EMBEDDINGS_DIR = BASE_DIR / "nlp_course" / "experiments" / "v1" / "embeddings"


def create_classification_report(y_test: List[Any], predictions: List[Any]):
    accuracy = accuracy_score(y_test, predictions)
    class_report = classification_report(y_test, predictions)

    print("Accuracy:", accuracy)
    print("Classification Report:\n", class_report)

    results = {
        "accuracy": accuracy,
        "classification_report": class_report,
    }
    return results


def train_and_evaluate_classifier(X_train, y_train, X_test, y_test, sklearn_classifier):
    sklearn_classifier.fit(X_train, y_train)
    predictions = sklearn_classifier.predict(X_test)

    results = create_classification_report(y_test=y_test, predictions=predictions)

    return sklearn_classifier, results


def evaluate_llm2vec_embedding_model(embedding_model: LLM2Vec, dataset: DatasetDict):
    train_test_split = generate_train_test_data(embedding_model, dataset)
    y_train = train_test_split.y_train
    y_test = train_test_split.y_test

    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_test_split.X_train)
    X_test = scaler.transform(train_test_split.X_test)

    # Our classifier
    lr_clf = LogisticRegression(random_state=0, C=1.0, max_iter=1000, verbose=1)
    dummy_most_frequent_clf = DummyClassifier(strategy="most_frequent")
    dummy_uniform_clf = DummyClassifier(strategy="uniform", random_state=42)
    classifiers = [
        (lr_clf, "lr_clf"),
        (dummy_most_frequent_clf, "dummy_most_frequent_clf"),
        (dummy_uniform_clf, "dummy_uniform_clf"),
    ]

    results = {}
    for clf_class, clf_name in classifiers:
        _, clf_results = train_and_evaluate_classifier(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            sklearn_classifier=clf_class,
        )
        results[clf_name] = clf_results

    return results


def main():
    parser = argparse.ArgumentParser(description="Train classifier")
    parser.add_argument(
        "-o", dest="output", type=str, required=False, default="results.json"
    )
    args = parser.parse_args()

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

    output_path = args.output
    with open(output_path, "w") as json_file:
        json.dump(results, json_file, indent=4)

    print("Results have been saved")


if __name__ == "__main__":
    main()
