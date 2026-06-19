"""Perceptron model model for Assignment 1: Starter code.

You can change this code while keeping the function giving headers. You can add any functions that will help you. The given function headers are used for testing the code, so changing them will fail testing.
"""

import argparse
import json
import os
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Set

from features import make_featurize
from tqdm import tqdm
from utils import DataPoint, DataType, accuracy, load_data, save_results


@dataclass(frozen=True)
class DataPointWithFeatures(DataPoint):
    features: Dict[str, float]


def featurize_data(
    data: List[DataPoint], feature_types: Set[str]
) -> List[DataPointWithFeatures]:
    """Add features to each datapoint based on feature types"""
    featurize = make_featurize(feature_types)
    return [
        DataPointWithFeatures(
            id=d.id,
            text=d.text,
            label=d.label,
            features=featurize(d.text),
        )
        for d in data
    ]


class PerceptronModel:
    """Perceptron model for classification."""

    def __init__(self):
        self.weights: Dict[str, float] = defaultdict(float)
        self.labels: Set[str] = set()

    def _get_weight_key(self, feature: str, label: str) -> str:
        """An internal hash function to build keys of self.weights (needed for tests)"""
        return feature + "#" + str(label)

    def score(self, datapoint: DataPointWithFeatures, label: str) -> float:
        """Compute the score of a class given the input.

        Inputs:
            datapoint (Datapoint): a single datapoint with features populated
            label (str): label

        Returns:
            The output score.
        """
        score = 0.0
        for feature, value in datapoint.features.items():
            score += value * self.weights[self._get_weight_key(feature, label)]
        return score

    def predict(self, datapoint: DataPointWithFeatures) -> str:
        """Predicts a label for an input.

        Inputs:
            datapoint: Input data point.

        Returns:
            The predicted class.
        """
        if not self.labels:
            raise ValueError(
                "Cannot predict without known labels. Train or set labels first."
            )
        return max(
            sorted(self.labels),
            key=lambda label: self.score(datapoint, label),
        )

    def update_parameters(
        self, datapoint: DataPointWithFeatures, prediction: str, lr: float
    ) -> None:
        """Update the model weights of the model using the perceptron update rule.

        Inputs:
            datapoint: The input example, including its label.
            prediction: The predicted label.
            lr: Learning rate.
        """
        if datapoint.label is None or prediction == datapoint.label:
            return
        gold_label = datapoint.label
        for feature, value in datapoint.features.items():
            gold_key = self._get_weight_key(feature, gold_label)
            pred_key = self._get_weight_key(feature, prediction)
            self.weights[gold_key] += lr * value
            self.weights[pred_key] -= lr * value

    def train(
        self,
        training_data: List[DataPointWithFeatures],
        val_data: List[DataPointWithFeatures],
        num_epochs: int,
        lr: float,
    ) -> None:
        """Perceptron model training. Updates self.weights and self.labels
        We greedily learn about new labels.

        Inputs:
            training_data: Suggested type is (list of tuple), where each item can be
                a training example represented as an (input, label) pair or (input, id, label) tuple.
            val_data: Validation data.
            num_epochs: Number of training epochs.
            lr: Learning rate.
        """
        for epoch in range(num_epochs):
            mistakes = 0
            for datapoint in tqdm(training_data):
                if datapoint.label is None:
                    continue
                if not self.labels:
                    self.labels.add(datapoint.label)
                prediction = self.predict(datapoint)
                if prediction != datapoint.label:
                    mistakes += 1
                    self.update_parameters(datapoint, prediction, lr)
                self.labels.add(datapoint.label)

            if len(val_data) > 0 and len(self.labels) > 0:
                val_acc = self.evaluate(val_data)
                print(
                    f"Epoch: {epoch + 1:<2} | Mistakes: {mistakes:<5} | Val accuracy: {100 * val_acc:.2f}%"
                )
            else:
                print(f"Epoch: {epoch + 1:<2} | Mistakes: {mistakes:<5}")

    def save_weights(self, path: str) -> None:
        with open(path, "w") as f:
            f.write(json.dumps(self.weights, indent=2, sort_keys=True))
        print(f"Model weights saved to {path}")

    def evaluate(
        self,
        data: List[DataPointWithFeatures],
        save_path: str = None,
    ) -> float:
        """Evaluates the model on the given data.

        Inputs:
            data (list of Datapoint): The data to evaluate on.
            save_path: The path to save the predictions.

        Returns:
            accuracy (float): The accuracy of the model on the data.
        """
        predictions = []
        for datapoint in data:
            if not self.labels:
                predictions.append(datapoint.label if datapoint.label else "")
            else:
                predictions.append(self.predict(datapoint))

        if save_path is not None:
            save_dir = os.path.dirname(save_path)
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
            save_results(data, predictions, save_path)

        labeled_pairs = [
            (pred, datapoint.label)
            for pred, datapoint in zip(predictions, data)
            if datapoint.label is not None
        ]
        if len(labeled_pairs) == 0:
            return 0.0

        preds, targets = zip(*labeled_pairs)
        return accuracy(list(preds), list(targets))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Perceptron model")
    parser.add_argument(
        "-d",
        "--data",
        type=str,
        default="sst2",
        help="Data source, one of ('sst2', 'newsgroups')",
    )
    parser.add_argument(
        "-f",
        "--features",
        type=str,
        default="bow",
        help="Feature type, e.g., bow+len",
    )
    parser.add_argument(
        "-e", "--epochs", type=int, default=3, help="Number of epochs"
    )
    parser.add_argument(
        "-l", "--learning_rate", type=float, default=0.1, help="Learning rate"
    )
    args = parser.parse_args()

    data_type = DataType(args.data)
    feature_types: Set[str] = set(args.features.split("+"))
    num_epochs: int = args.epochs
    lr: float = args.learning_rate

    train_data, val_data, dev_data, test_data = load_data(data_type)
    train_data = featurize_data(train_data, feature_types)
    val_data = featurize_data(val_data, feature_types)
    dev_data = featurize_data(dev_data, feature_types)
    test_data = featurize_data(test_data, feature_types)

    model = PerceptronModel()
    print("Training the model...")
    model.train(train_data, val_data, num_epochs, lr)

    # Predict on the development set.
    dev_acc = model.evaluate(
        dev_data,
        save_path=os.path.join(
            "results",
            f"perceptron_{args.data}_{args.features}_dev_predictions.csv",
        ),
    )
    print(f"Development accuracy: {100 * dev_acc:.2f}%")

    # Predict on the test set
    _ = model.evaluate(
        test_data,
        save_path=os.path.join(
            "results",
            f"perceptron_{args.data}_test_predictions.csv",
        ),
    )

    model.save_weights(
        os.path.join(
            "results", f"perceptron_{args.data}_{args.features}_model.json"
        )
    )
