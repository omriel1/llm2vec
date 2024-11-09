"""
This module heavily follows:
https://www.learnpytorch.io/02_pytorch_classification/#85-creating-a-training-and-testing-loop-for-a-multi-class-pytorch-model
Also see this code review by chatGPT: https://chatgpt.com/share/6724c76e-eaac-800c-940b-fb128ef0a42d
which emphasized very good points regarding the cross entropy loss and its expected input.
"""

import argparse
from typing import Dict, Any, List

import torch
from torch import nn
import torch.optim as optim

from llm2vec import LLM2Vec
from nlp_course import BASE_DIR
from nlp_course.prepare_hebsentiment_data import (
    load_hebsetiment_data,
    NUMERICAL_TARGET_COLUMN,
    get_label_from_index,
)
from nlp_course.utils import get_device


class LLM2VecClassifier(nn.Module):
    """
    Implementation of a simple Classifier on top of LLM2Vec mode.
    """

    def __init__(
        self,
        base_model: str,
        peft_model: str,
        embedding_dim: int,
        num_labels: int,
        hidden_units: int = 8,
    ):
        super().__init__()
        self.base_model = base_model
        self.peft_model = peft_model
        self.embedding_dim = embedding_dim
        self.num_labels = num_labels
        self.hidden_units = hidden_units

        self.embedding_model: LLM2Vec = LLM2Vec.from_pretrained(
            base_model_name_or_path=base_model,
            peft_model_name_or_path=peft_model,
            device_map=get_device(),
            torch_dtype=torch.bfloat16,
        )

        # Freeze all parameters in the embedding model, so its weights will not
        # get optimized during training!
        for param in self.embedding_model.parameters():
            param.requires_grad = False

        self.linear_layer_stack = nn.Sequential(
            nn.Linear(self.embedding_dim, self.hidden_units, bias=False),
            nn.ReLU(),
            nn.Linear(self.hidden_units, self.num_labels, bias=False),
        )

    def forward(self, text):
        # Note LLM2Vec.encode includes `with torch.no_grad()`
        embeddings = self.embedding_model.encode(text)
        embeddings = embeddings.to(next(self.linear_layer_stack.parameters()).device)
        return self.linear_layer_stack(embeddings)


def accuracy_fn(y_true, y_pred):
    correct = (
        torch.eq(y_true, y_pred).sum().item()
    )  # torch.eq() calculates where two tensors are equal
    acc = (correct / len(y_pred)) * 100
    return acc


def llm2vec_classifier_training_loop(
    X_train: list[str],
    y_train: list[int],
    X_test: list[str],
    y_test: list[int],
    llm2vec_cls: LLM2VecClassifier,
    epochs: int = 100,
) -> LLM2VecClassifier:
    device = get_device()
    # expects *logits* as the input, it'll apply softmax by itself.
    cross_entropy_loss_fn = nn.CrossEntropyLoss()
    torch.manual_seed(42)

    model = llm2vec_cls.to(device)
    optimizer = optim.Adam(model.linear_layer_stack.parameters(), lr=1e-3)

    y_train = torch.tensor(y_train).to(device)
    y_test = torch.tensor(y_test).to(device)

    for epoch in range(epochs):

        ### Training
        model.train()
        y_logits = model(X_train)
        y_pred = y_logits.argmax(dim=1)

        loss = cross_entropy_loss_fn(y_logits, y_train)
        acc = accuracy_fn(y_true=y_train, y_pred=y_pred)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        ### Testing
        model.eval()
        with torch.inference_mode():
            test_logits = model(X_test)
            test_pred = test_logits.argmax(dim=1)
            test_loss = cross_entropy_loss_fn(test_logits, y_test)
            test_acc = accuracy_fn(y_true=y_test, y_pred=test_pred)

        print(
            f"Epoch: {epoch} | Loss: {loss:.5f}, Acc: {acc:.2f}% | Test Loss: {test_loss:.5f}, Test Acc: {test_acc:.2f}%"
        )

    return model


def predict_sentiment(
    model: LLM2VecClassifier, texts: list[str]
) -> List[Dict[str, Any]]:
    model.eval()
    results = []

    with torch.no_grad():
        # Pass the batch of texts through the model to get logits
        logits = model(texts)  # List of texts passed in one forward pass
        probabilities = torch.softmax(logits, dim=1)  # Convert logits to probabilities

        # For each text, get predicted label and probability distribution
        for i in range(len(texts)):
            predicted_label_idx = probabilities[i].argmax().item()
            predicted_label = get_label_from_index(predicted_label_idx)
            probs = (
                probabilities[i].cpu().numpy().tolist()
            )  # Convert probabilities to list format

            results.append(
                {
                    "text": texts[i],
                    "predicted_label": predicted_label,
                    "probabilities": {
                        get_label_from_index(j): probs[j] for j in range(len(probs))
                    },
                }
            )

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train classifier")
    parser.add_argument("-e", dest="epochs", type=int, required=False, default=100)
    parser.add_argument(
        "-o",
        dest="output_dir",
        type=str,
        required=False,
        default="llm2vec-nn-classifier",
    )
    args = parser.parse_args()

    peft_model_dir = (
        BASE_DIR / "output" / "mntp-simcse" / "dictalm2.0-instruct" / "checkpoint-1000"
    )

    hebsentiment_dataset = load_hebsetiment_data()

    l2v_cls = LLM2VecClassifier(
        base_model="dicta-il/dictalm2.0-instruct",
        peft_model=peft_model_dir,
        embedding_dim=4096,
        num_labels=3,
        hidden_units=8,
    )

    trained_cls = llm2vec_classifier_training_loop(
        X_train=hebsentiment_dataset["train"]["text"],
        y_train=hebsentiment_dataset["train"][NUMERICAL_TARGET_COLUMN],
        X_test=hebsentiment_dataset["test"]["text"],
        y_test=hebsentiment_dataset["test"][NUMERICAL_TARGET_COLUMN],
        llm2vec_cls=l2v_cls,
        epochs=args.epochs,
    )

    output_model_file = args.output_dir + "/llm2vec_classifier_head.pth"

    torch.save(
        {
            "linear_layer_stack_state_dict": l2v_cls.linear_layer_stack.state_dict(),
            "embedding_dim": l2v_cls.embedding_dim,  # Useful if you want to reinitialize with the same dimensions
            "num_labels": l2v_cls.num_labels,
        },
        output_model_file,
    )

    print(f"Trained model was saved to {output_model_file}")
