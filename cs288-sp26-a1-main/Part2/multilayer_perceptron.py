"""Multi-layer perceptron model for Assignment 1: Starter code.

You can change this code while keeping the function giving headers. You can add any functions that will help you. The given function headers are used for testing the code, so changing them will fail testing.


We adapt shape suffixes style when working with tensors.
See https://medium.com/@NoamShazeer/shape-suffixes-good-coding-style-f836e72e24fd.

Dimension key:

b: batch size
l: max sequence length
c: number of classes
v: vocabulary size

For example,

feature_b_l means a tensor of shape (b, l) == (batch_size, max_sequence_length).
length_1 means a tensor of shape (1) == (1,).
loss means a tensor of shape (). You can retrieve the loss value with loss.item().
"""

import argparse
import copy
import os
import random
import re
from collections import Counter
from pprint import pprint
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from utils import DataPoint, DataType, accuracy, load_data, save_results


class Tokenizer:
    # The index of the padding embedding.
    # This is used to pad variable length sequences.
    TOK_PADDING_INDEX = 0
    TOK_UNK_INDEX = 1
    STOP_WORDS = set(pd.read_csv("stopwords.txt", header=None)[0])

    def _pre_process_text(self, text: str) -> List[str]:
        # Split on punctuation boundaries instead of collapsing punctuation into tokens.
        tokens = re.findall(r"[a-z0-9']+", text.lower())
        if self.remove_stopwords:
            tokens = [token for token in tokens if token not in self.STOP_WORDS]
        return tokens

    def __init__(
        self,
        data: List[DataPoint],
        max_vocab_size: int = None,
        remove_stopwords: bool = True,
        add_unk: bool = False,
    ):
        self.remove_stopwords = remove_stopwords
        self.add_unk = add_unk
        corpus = " ".join([d.text for d in data])
        token_freq = Counter(self._pre_process_text(corpus))
        token_freq = token_freq.most_common(max_vocab_size)
        tokens = [t for t, _ in token_freq]
        # Reserve 0 for padding, and optionally 1 for unknown.
        base_index = 2 if self.add_unk else 1
        self.token2id = {t: (i + base_index) for i, t in enumerate(tokens)}
        self.token2id["<PAD>"] = Tokenizer.TOK_PADDING_INDEX
        if self.add_unk:
            self.token2id["<UNK>"] = Tokenizer.TOK_UNK_INDEX
        self.id2token = {i: t for t, i in self.token2id.items()}

    def tokenize(self, text: str) -> List[int]:
        token_ids = []
        for token in self._pre_process_text(text):
            token_id = self.token2id.get(token)
            if token_id is not None:
                token_ids.append(token_id)
            elif self.add_unk:
                token_ids.append(Tokenizer.TOK_UNK_INDEX)
        return token_ids


def get_label_mappings(
    data: List[DataPoint],
) -> Tuple[Dict[str, int], Dict[int, str]]:
    """Reads the labels file and returns the mapping."""
    labels = sorted(set([d.label for d in data]))
    label2id = {label: index for index, label in enumerate(labels)}
    id2label = {index: label for index, label in enumerate(labels)}
    return label2id, id2label


class BOWDataset(Dataset):
    def __init__(
        self,
        data: List[DataPoint],
        tokenizer: Tokenizer,
        label2id: Dict[str, int],
        max_length: int = 100,
    ):
        super().__init__()
        self.data = data
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(
        self, idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns a single example as a tuple of torch.Tensors.
        features_l: The tokenized text of example, shaped (max_length,)
        length: The length of the text, shaped ()
        label: The label of the example, shaped ()

        All of have type torch.int64.
        """
        dp: DataPoint = self.data[idx]
        token_ids = self.tokenizer.tokenize(dp.text)
        token_ids = token_ids[: self.max_length]
        length = len(token_ids)

        pad_size = self.max_length - length
        if pad_size > 0:
            token_ids = token_ids + [Tokenizer.TOK_PADDING_INDEX] * pad_size

        if dp.label is None:
            label_id = -1
        else:
            label_id = self.label2id[dp.label]

        features_l = torch.tensor(token_ids, dtype=torch.int64)
        length_1 = torch.tensor(length, dtype=torch.int64)
        label_1 = torch.tensor(label_id, dtype=torch.int64)
        return features_l, length_1, label_1


class MultilayerPerceptronModel(nn.Module):
    """Multi-layer perceptron model for classification."""

    def __init__(self, vocab_size: int, num_classes: int, padding_index: int):
        """Initializes the model.

        Inputs:
            num_classes (int): The number of classes.
            vocab_size (int): The size of the vocabulary.
        """
        super().__init__()
        self.padding_index = padding_index
        embedding_dim = 128
        hidden_dim1 = 128
        hidden_dim2 = 64

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=padding_index,
        )
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim1),
            nn.ReLU(),
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.Tanh(),
            nn.Linear(hidden_dim2, num_classes),
        )

    def forward(
        self, input_features_b_l: torch.Tensor, input_length_b: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass of the model.

        Inputs:
            input_features_b_l (tensor): Input data for an example or a batch of examples.
            input_length (tensor): The length of the input data.

        Returns:
            output_b_c: The output of the model.
        """
        embeddings_b_l_h = self.embedding(input_features_b_l)

        non_padding_mask_b_l_1 = (
            input_features_b_l != self.padding_index
        ).unsqueeze(-1)
        summed_b_h = (
            embeddings_b_l_h * non_padding_mask_b_l_1.to(embeddings_b_l_h.dtype)
        ).sum(dim=1)

        safe_lengths_b_1 = input_length_b.clamp(min=1).unsqueeze(-1).to(
            embeddings_b_l_h.dtype
        )
        pooled_b_h = summed_b_h / safe_lengths_b_1

        output_b_c = self.classifier(pooled_b_h)
        return output_b_c


class Trainer:
    def __init__(self, model: nn.Module, batch_size: int = 8):
        self.model = model
        self.batch_size = batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def predict(self, data: BOWDataset) -> List[int]:
        """Predicts a label for an input.

        Inputs:
            model_input (tensor): Input data for an example or a batch of examples.

        Returns:
            The predicted class.

        """
        all_predictions = []
        dataloader = DataLoader(data, batch_size=32, shuffle=False)
        self.model.eval()
        with torch.no_grad():
            for inputs_b_l, lengths_b, _ in dataloader:
                inputs_b_l = inputs_b_l.to(self.device)
                lengths_b = lengths_b.to(self.device)
                logits_b_c = self.model(inputs_b_l, lengths_b)
                batch_predictions_b = torch.argmax(logits_b_c, dim=-1)
                all_predictions.extend(batch_predictions_b.tolist())
        return all_predictions

    def evaluate(self, data: BOWDataset) -> float:
        """Evaluates the model on a dataset.

        Inputs:
            data: The dataset to evaluate on.

        Returns:
            The accuracy of the model.
        """
        predictions = self.predict(data)
        filtered_preds = []
        targets = []
        for pred, dp in zip(predictions, data.data):
            if dp.label is None:
                continue
            filtered_preds.append(pred)
            targets.append(data.label2id[dp.label])

        if len(targets) == 0:
            return 0.0
        return accuracy(filtered_preds, targets)

    def train(
        self,
        training_data: BOWDataset,
        val_data: BOWDataset,
        optimizer: torch.optim.Optimizer,
        num_epochs: int,
    ) -> None:
        """Trains the MLP.

        Inputs:
            training_data: Suggested type for an individual training example is
                an (input, label) pair or (input, id, label) tuple.
                You can also use a dataloader.
            val_data: Validation data.
            optimizer: The optimization method.
            num_epochs: The number of training epochs.
        """
        criterion = nn.CrossEntropyLoss()
        best_val_acc = float("-inf")
        best_state = None
        epochs_without_improvement = 0
        patience = 2
        for epoch in range(num_epochs):
            self.model.train()
            total_loss = 0.0
            dataloader = DataLoader(
                training_data, batch_size=self.batch_size, shuffle=True
            )
            for inputs_b_l, lengths_b, labels_b in tqdm(dataloader):
                inputs_b_l = inputs_b_l.to(self.device)
                lengths_b = lengths_b.to(self.device)
                labels_b = labels_b.to(self.device)
                optimizer.zero_grad()
                logits_b_c = self.model(inputs_b_l, lengths_b)
                loss = criterion(logits_b_c, labels_b)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * inputs_b_l.size(0)
            per_dp_loss = (
                total_loss / len(training_data) if len(training_data) > 0 else 0.0
            )

            self.model.eval()
            val_acc = self.evaluate(val_data)
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state = copy.deepcopy(self.model.state_dict())
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            print(
                f"Epoch: {epoch + 1:<2} | Loss: {per_dp_loss:.2f} | Val accuracy: {100 * val_acc:.2f}%"
            )
            if epochs_without_improvement >= patience:
                break
        if best_state is not None:
            self.model.load_state_dict(best_state)
            print(f"Loaded best validation checkpoint ({100 * best_val_acc:.2f}%).")


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _majority_vote(pred_matrix: List[List[int]]) -> List[int]:
    stacked = np.array(pred_matrix, dtype=np.int64)  # (num_models, num_examples)
    voted = []
    for i in range(stacked.shape[1]):
        counts = Counter(stacked[:, i].tolist())
        # Deterministic tie break by smaller class id.
        best = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
        voted.append(int(best))
    return voted


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MultiLayerPerceptron model")
    parser.add_argument(
        "-d",
        "--data",
        type=str,
        default="sst2",
        help="Data source, one of ('sst2', 'newsgroups')",
    )
    parser.add_argument(
        "-e", "--epochs", type=int, default=3, help="Number of epochs"
    )
    parser.add_argument(
        "-l", "--learning_rate", type=float, default=0.001, help="Learning rate"
    )
    parser.add_argument(
        "--max_vocab_size",
        type=int,
        default=None,
        help="Maximum vocabulary size (dataset-specific default if omitted)",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=None,
        help="Maximum input sequence length (dataset-specific default if omitted)",
    )
    parser.add_argument(
        "--remove_stopwords",
        type=int,
        default=None,
        help="1 to remove stopwords, 0 otherwise (dataset-specific default if omitted)",
    )
    parser.add_argument(
        "--add_unk",
        type=int,
        default=None,
        help="1 to map unseen tokens to <UNK>, 0 to drop unseen tokens",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Training batch size",
    )
    parser.add_argument(
        "--ensemble_seeds",
        type=str,
        default="",
        help="Comma-separated list of seeds for prediction ensembling",
    )
    args = parser.parse_args()

    num_epochs = args.epochs
    lr = args.learning_rate
    data_type = DataType(args.data)
    batch_size = args.batch_size
    if args.ensemble_seeds.strip():
        seeds = [
            int(seed.strip())
            for seed in args.ensemble_seeds.split(",")
            if seed.strip() != ""
        ]
    else:
        seeds = [args.seed]

    if data_type == DataType.NEWSGROUPS:
        max_vocab_size = (
            args.max_vocab_size if args.max_vocab_size is not None else 40000
        )
        max_length = args.max_length if args.max_length is not None else 250
        remove_stopwords = (
            bool(args.remove_stopwords) if args.remove_stopwords is not None else True
        )
        add_unk = bool(args.add_unk) if args.add_unk is not None else True
    else:
        max_vocab_size = (
            args.max_vocab_size if args.max_vocab_size is not None else 20000
        )
        max_length = args.max_length if args.max_length is not None else 100
        remove_stopwords = (
            bool(args.remove_stopwords) if args.remove_stopwords is not None else True
        )
        add_unk = bool(args.add_unk) if args.add_unk is not None else False

    train_data, val_data, dev_data, test_data = load_data(data_type)

    tokenizer = Tokenizer(
        train_data,
        max_vocab_size=max_vocab_size,
        remove_stopwords=remove_stopwords,
        add_unk=add_unk,
    )
    label2id, id2label = get_label_mappings(train_data)
    print("Id to label mapping:")
    pprint(id2label)

    train_ds = BOWDataset(train_data, tokenizer, label2id, max_length)
    val_ds = BOWDataset(val_data, tokenizer, label2id, max_length)
    dev_ds = BOWDataset(dev_data, tokenizer, label2id, max_length)
    test_ds = BOWDataset(test_data, tokenizer, label2id, max_length)
    dev_targets = [label2id[d.label] for d in dev_data]

    all_dev_preds: List[List[int]] = []
    all_test_preds: List[List[int]] = []
    for seed in seeds:
        _set_seed(seed)
        model = MultilayerPerceptronModel(
            vocab_size=len(tokenizer.token2id),
            num_classes=len(label2id),
            padding_index=Tokenizer.TOK_PADDING_INDEX,
        )
        trainer = Trainer(model, batch_size=batch_size)

        print(f"Training the model (seed={seed})...")
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        trainer.train(train_ds, val_ds, optimizer, num_epochs)

        dev_preds = trainer.predict(dev_ds)
        test_preds = trainer.predict(test_ds)
        all_dev_preds.append(dev_preds)
        all_test_preds.append(test_preds)

        seed_dev_acc = accuracy(dev_preds, dev_targets)
        print(f"Seed {seed} development accuracy: {100 * seed_dev_acc:.2f}%")

    if len(all_dev_preds) == 1:
        final_dev_preds = all_dev_preds[0]
        final_test_preds = all_test_preds[0]
    else:
        final_dev_preds = _majority_vote(all_dev_preds)
        final_test_preds = _majority_vote(all_test_preds)

    dev_acc = accuracy(final_dev_preds, dev_targets)
    print(f"Development accuracy: {100 * dev_acc:.2f}%")

    test_preds = [id2label[pred] for pred in final_test_preds]
    save_results(
        test_data,
        test_preds,
        os.path.join("results", f"mlp_{args.data}_test_predictions.csv"),
    )
