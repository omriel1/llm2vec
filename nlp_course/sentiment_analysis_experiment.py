import argparse
import json
from pathlib import Path
from typing import Any

import yaml
from datasets import DatasetDict
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression

from nlp_course import BASE_DIR
from nlp_course.classifier import SklearnSentimentClassifier, BaseSentimentClassifier
from nlp_course.encoder import (
    LLM2VecEncoder,
    SentenceTransformersEncoder,
    BaseSentenceEncoder,
    PLMBERTBasedEncoder,
)
from nlp_course.experiment_data_generator import ExperimentDataGenerator
from nlp_course.prepare_hebsentiment_data import load_hebsetiment_data

HEBSENTIMENT_DATA = "HebSentiment"


def load_yaml_config(config_path: str) -> dict:
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config


def get_experiment_data(dataset_name: str) -> DatasetDict:
    if dataset_name == HEBSENTIMENT_DATA:
        return load_hebsetiment_data()
    raise ValueError(f"Unrecognized dataset name: {dataset_name}")


def initialize_encoder(encoder_config: dict) -> BaseSentenceEncoder:
    configs = list(encoder_config.values())[0]
    encoder_type = configs["type"]
    model_name = configs["model_name"]

    match encoder_type:
        case "llm2vec":
            return LLM2VecEncoder(model_name=model_name)

        case "sentence_transformers":
            return SentenceTransformersEncoder(model_name=model_name)

        case "plm_bert_based":
            return PLMBERTBasedEncoder(model_name=model_name)

        case _:
            raise ValueError(f"Unknown encoder type: {encoder_type}")


def initialize_classifier(classifier_config: dict[str, Any]) -> BaseSentimentClassifier:
    name, configs = list(classifier_config.items())[0]
    classifier_type = configs["type"]
    method = configs["method"]
    params = {p: v for param in configs["params"] for p, v in param.items()}

    if classifier_type == "sklearn":
        match method:
            case "logistic_regression":
                classifier = LogisticRegression(**params)
            case "dummy":
                classifier = DummyClassifier(**params)
            case _:
                raise ValueError(f"Unknown sklearn method: {method}")
        return SklearnSentimentClassifier(classifier=classifier, classifier_name=name)

    raise ValueError(f"Unknown classifier type: {classifier_type}")


def perform_experiment(
    configs: dict[str, Any], base_cache_dir: Path | None
) -> dict[str, Any]:
    encoders = [initialize_encoder(enc_cfg) for enc_cfg in configs["encoders"]]
    dataset_name = configs["dataset"]
    data = get_experiment_data(dataset_name)

    final_report = {}
    for encoder in encoders:
        experiment_data_generator = ExperimentDataGenerator(
            encoder=encoder,
            dataset=data,
            dataset_name=dataset_name,
            base_cache_dir=base_cache_dir,
        )
        train_test_split = experiment_data_generator.generate_train_test_data()

        # For each encoder, we should initialize its own classifier instances
        classifiers = [
            initialize_classifier(clf_cfg) for clf_cfg in configs["classifiers"]
        ]
        classifiers_report = {}
        for clf in classifiers:
            report = clf.evaluate_self(train_test_split)
            classifiers_report.update(report)
        final_report[encoder.model_name] = classifiers_report

    return final_report


def main():
    parser = argparse.ArgumentParser(description="Run sentiment analysis experiment")
    parser.add_argument(
        "-c",
        "--config",
        required=True,
        type=str,
        help="Path to YAML configuration file",
    )
    parser.add_argument("-o", "--output", type=str, required=False, default=".")
    args = parser.parse_args()

    configs = load_yaml_config(args.config)
    experiment_name = configs["experiment_name"]

    report = perform_experiment(
        configs=configs,
        base_cache_dir=BASE_DIR / "nlp_course" / "experiments" / "v1" / "embeddings",
    )

    output_path = Path(args.output)
    output_file_path = (output_path / experiment_name).as_posix() + ".json"
    with open(output_file_path, "w") as json_file:
        json.dump(report, json_file, indent=4)

    print(f"Results have been saved to {output_path}")


if __name__ == "__main__":
    # python nlp_course/sentiment_analysis_experiment.py -c nlp_course/llm2vec_classifier_experiment.yaml -o .
    # python nlp_course/sentiment_analysis_experiment.py -c nlp_course/different_classifiers_experiment.yaml -o .
    main()
