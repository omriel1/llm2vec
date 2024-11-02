import argparse
import json
from pathlib import Path
from typing import Any

import torch
from datasets import DatasetDict
from llm2vec import LLM2Vec

from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from nlp_course import BASE_DIR
from nlp_course.classifier import SklearnSentimentClassifier
from nlp_course.encoder import LLM2VecEncoder
from nlp_course.experiment_data_generator import ExperimentDataGenerator
from nlp_course.prepare_hebsentiment_data import load_hebsetiment_data
from nlp_course.utils import get_device


def perform_experiment(
    dataset: DatasetDict, dataset_name: str, base_cache_dir: Path | None
) -> dict[str, Any]:
    l2v = LLM2Vec.from_pretrained(
        base_model_name_or_path="dicta-il/dictalm2.0-instruct",
        peft_model_name_or_path=BASE_DIR
        / "output"
        / "mntp-simcse"
        / "dictalm2.0-instruct"
        / "checkpoint-1000",
        device_map=get_device(),
        torch_dtype=torch.bfloat16,
    )

    llm2vec_encoder = LLM2VecEncoder(model=l2v, model_name="llm2vec_dictalm2_encoder")

    classifiers = [
        SklearnSentimentClassifier(
            classifier=LogisticRegression(
                random_state=0, C=1.0, max_iter=100, verbose=1
            ),
            classifier_name="logistic_regression_classifier",
        ),
        SklearnSentimentClassifier(
            classifier=DummyClassifier(strategy="most_frequent"),
            classifier_name="dummy_most_frequent_classifier",
        ),
        SklearnSentimentClassifier(
            classifier=DummyClassifier(strategy="uniform", random_state=42),
            classifier_name="dummy_uniform_classifier",
        ),
    ]

    llm2vec_experiment_data_generator = ExperimentDataGenerator(
        encoder=llm2vec_encoder,
        dataset=dataset,
        dataset_name=dataset_name,
        base_cache_dir=base_cache_dir,
    )
    train_test_split = llm2vec_experiment_data_generator.generate_train_test_data()

    final_report = {}
    for clf in classifiers:
        report = clf.evaluate_self(train_test_split)
        final_report.update(report)

    return final_report


def main():
    parser = argparse.ArgumentParser(description="Train classifier")
    parser.add_argument(
        "-o", dest="output", type=str, required=False, default="llm2vec_classification_report.json"
    )
    args = parser.parse_args()

    report = perform_experiment(
        dataset=load_hebsetiment_data(),
        dataset_name="HebSentiment",
        base_cache_dir=BASE_DIR / "nlp_course" / "experiments" / "v1" / "embeddings",
    )

    output_path = args.output
    with open(output_path, "w") as json_file:
        json.dump(report, json_file, indent=4)

    print(f"Results have been saved to {output_path}")


if __name__ == "__main__":
    main()
