import argparse
import json

import torch

from nlp_course import BASE_DIR
from nlp_course.experiments.nn_classifier.llm2vec_nn_classifier import (
    LLM2VecClassifier,
    predict_sentiment,
)
from nlp_course.experiments.nn_classifier.sklearn_classifier import (
    create_classification_report,
)
from nlp_course.prepare_hebsentiment_data import (
    load_hebsetiment_data,
)
from nlp_course.utils import get_device


def evaluate_nn_classifier():
    peft_model_dir = (
        BASE_DIR / "output" / "mntp-simcse" / "dictalm2.0-instruct" / "checkpoint-1000"
    )
    nn_classifier_head_path = (
        BASE_DIR / "output" / "nn-classifier" / "llm2vec_classifier_head.pth"
    )

    l2v_cls = LLM2VecClassifier(
        base_model="dicta-il/dictalm2.0-instruct",
        peft_model=peft_model_dir,
        embedding_dim=4096,
        num_labels=3,
        hidden_units=8,
    )
    # Note! as the model weights are freezed we need to load the model as is and just replace
    # with the trained classification head!
    nn_classification_head = torch.load(nn_classifier_head_path)
    l2v_cls.linear_layer_stack.load_state_dict(
        nn_classification_head["linear_layer_stack_state_dict"]
    )
    l2v_cls.to(get_device())

    hebsentiment_dataset = load_hebsetiment_data()
    X_test = hebsentiment_dataset["test"]["text"]
    y_test = hebsentiment_dataset["test"][
        "tag_ids"
    ]  # we should use the text version here!

    predictions_dicts = predict_sentiment(model=l2v_cls, texts=X_test)
    predictions = [d["predicted_label"] for d in predictions_dicts]
    results = create_classification_report(y_test=y_test, predictions=predictions)
    return results, predictions_dicts


def main():
    parser = argparse.ArgumentParser(description="Train classifier")
    parser.add_argument(
        "-o",
        dest="output_dir",
        type=str,
        required=False,
        default=".",
    )
    args = parser.parse_args()

    results, predictions_dicts = evaluate_nn_classifier()

    results_file = args.output_dir + "nn_classifier_results.json"
    with open(results_file, "w") as json_file:
        json.dump(results, json_file, indent=4)

    predictions_file = args.output_dir + "nn_classifier_predictions.json"
    with open(predictions_file, "w") as json_file:
        json.dump(predictions_dicts, json_file, indent=4)


if __name__ == "__main__":
    main()
