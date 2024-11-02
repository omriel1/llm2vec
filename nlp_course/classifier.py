from abc import ABC, abstractmethod
from typing import Any

import torch
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler

from nlp_course.experiment_data_generator import TrainTestSplit


class BaseSentimentClassifier(ABC):
    """
    This is a base class for sentiment classification.
    """

    @abstractmethod
    def train(self, X_train, y_train) -> None:
        raise NotImplementedError()

    @abstractmethod
    def predict(self, texts: list[str]) -> list[str]:
        raise NotImplementedError()


class SklearnSentimentClassifier(BaseSentimentClassifier):

    def __init__(self, classifier, classifier_name: str):
        self.classifier = classifier
        self.classifier_name = classifier_name

    def train(self, X_train: torch.Tensor, y_train: list[str]) -> None:
        self.classifier.fit(X_train, y_train)

    def predict(self, texts: list[str]) -> list[str]:
        return self.classifier.predict(texts)

    def create_classification_report(
        self, y_test: list[str], predictions: list[str]
    ) -> dict[str, Any]:
        accuracy = accuracy_score(y_true=y_test, y_pred=predictions)
        class_report = classification_report(y_true=y_test, y_pred=predictions)

        results = {
            "accuracy": accuracy,
            "classification_report": class_report,
        }
        return {self.classifier_name: results}

    def evaluate_self(
        self, train_test_split: TrainTestSplit, scale: bool = True
    ) -> dict[str, Any]:
        X_train = train_test_split.X_train
        X_test = train_test_split.X_test
        if scale:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(train_test_split.X_train)
            X_test = scaler.transform(train_test_split.X_test)

        self.train(X_train=X_train, y_train=train_test_split.y_train)
        predictions = self.predict(X_test)
        report = self.create_classification_report(
            y_test=train_test_split.y_test, predictions=predictions
        )
        return report
