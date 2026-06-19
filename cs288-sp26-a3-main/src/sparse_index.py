from __future__ import annotations

import math
import pickle
from pathlib import Path

from .text_utils import significant_tokens, tokenize


class SparsePassageIndex:
    def __init__(
        self,
        passages: list[dict[str, object]],
        postings_by_token: dict[str, list[tuple[int, int]]],
        inverse_document_frequency: dict[str, float],
        document_lengths: list[int],
        average_document_length: float,
        title_tokens_by_passage: list[set[str]],
    ):
        self.passages = passages
        self.postings_by_token = postings_by_token
        self.inverse_document_frequency = inverse_document_frequency
        self.document_lengths = document_lengths
        self.average_document_length = average_document_length
        self.title_tokens_by_passage = title_tokens_by_passage
        self.k1 = 1.6
        self.b = 0.75

    @classmethod
    def build(cls, passages: list[dict[str, object]]) -> "SparsePassageIndex":
        postings_by_token: dict[str, list[tuple[int, int]]] = {}
        document_lengths: list[int] = []
        title_tokens_by_passage: list[set[str]] = []
        for passage_index, passage in enumerate(passages):
            content_tokens = tokenize(str(passage["text"]))
            token_counts: dict[str, int] = {}
            for token in content_tokens:
                token_counts[token] = token_counts.get(token, 0) + 1
            for token, count in token_counts.items():
                postings_by_token.setdefault(token, []).append((passage_index, count))
            document_lengths.append(len(content_tokens) or 1)
            title_tokens_by_passage.append(set(significant_tokens(str(passage["title"]))))
        average_document_length = sum(document_lengths) / len(document_lengths) if document_lengths else 1.0
        total_documents = len(passages)
        inverse_document_frequency: dict[str, float] = {}
        for token, posting_list in postings_by_token.items():
            document_frequency = len(posting_list)
            inverse_document_frequency[token] = math.log(
                1 + (total_documents - document_frequency + 0.5) / (document_frequency + 0.5)
            )
        return cls(
            passages=passages,
            postings_by_token=postings_by_token,
            inverse_document_frequency=inverse_document_frequency,
            document_lengths=document_lengths,
            average_document_length=average_document_length,
            title_tokens_by_passage=title_tokens_by_passage,
        )

    def save(self, output_path: str | Path) -> None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as handle:
            pickle.dump(
                {
                    "passages": self.passages,
                    "postings_by_token": self.postings_by_token,
                    "inverse_document_frequency": self.inverse_document_frequency,
                    "document_lengths": self.document_lengths,
                    "average_document_length": self.average_document_length,
                    "title_tokens_by_passage": self.title_tokens_by_passage,
                },
                handle,
            )

    @classmethod
    def load(cls, input_path: str | Path) -> "SparsePassageIndex":
        with Path(input_path).open("rb") as handle:
            payload = pickle.load(handle)
        return cls(**payload)

    def search(self, question: str, limit: int = 8) -> list[dict[str, object]]:
        query_tokens = significant_tokens(question)
        if not query_tokens:
            query_tokens = tokenize(question)
        scores: dict[int, float] = {}
        for token in query_tokens:
            posting_list = self.postings_by_token.get(token)
            if not posting_list:
                continue
            token_weight = self.inverse_document_frequency.get(token, 0.0)
            for passage_index, term_frequency in posting_list:
                document_length = self.document_lengths[passage_index]
                numerator = term_frequency * (self.k1 + 1)
                denominator = term_frequency + self.k1 * (
                    1 - self.b + self.b * document_length / self.average_document_length
                )
                passage_score = token_weight * numerator / denominator
                if token in self.title_tokens_by_passage[passage_index]:
                    passage_score += 0.35
                scores[passage_index] = scores.get(passage_index, 0.0) + passage_score
        ranked_indices = sorted(scores.items(), key=lambda item: item[1], reverse=True)[:limit]
        return [
            {"score": score, "passage": self.passages[passage_index]}
            for passage_index, score in ranked_indices
        ]
