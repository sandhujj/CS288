from collections import ChainMap
import re
from typing import Callable, Dict, Set

import pandas as pd


def _normalize_token(token: str) -> str:
    """Lowercase and keep only alphanumerics plus apostrophes."""
    token = token.lower()
    token = re.sub(r"[^a-z0-9']+", "", token)
    return token


def _tokenize(text: str) -> list[str]:
    return [
        t
        for t in (_normalize_token(raw_t) for raw_t in text.split())
        if t != ""
    ]


class FeatureMap:
    name: str

    @classmethod
    def featurize(self, text: str) -> Dict[str, float]:
        pass

    @classmethod
    def prefix_with_name(self, d: Dict) -> Dict[str, float]:
        """just a handy shared util function"""
        return {f"{self.name}/{k}": v for k, v in d.items()}


class BagOfWords(FeatureMap):
    name = "bow"
    STOP_WORDS = set(pd.read_csv("stopwords.txt", header=None)[0])

    @classmethod
    def featurize(self, text: str) -> Dict[str, float]:
        vocab = {
            token
            for token in _tokenize(text)
            if token not in self.STOP_WORDS
        }
        return self.prefix_with_name({token: 1.0 for token in vocab})


class SentenceLength(FeatureMap):
    name = "len"

    @classmethod
    def featurize(self, text: str) -> Dict[str, float]:
        """an example of custom feature that rewards long sentences"""
        if len(text.split()) < 10:
            k = "short"
            v = 1.0
        else:
            k = "long"
            v = 5.0
        ret = {k: v}
        return self.prefix_with_name(ret)


class SentimentLexicon(FeatureMap):
    """Small hand-crafted sentiment lexicon features for SST-like text."""

    name = "sentlex"

    POSITIVE_WORDS = {
        "amazing",
        "awesome",
        "best",
        "excellent",
        "fun",
        "good",
        "great",
        "love",
        "loved",
        "masterpiece",
        "perfect",
        "superb",
        "wonderful",
    }
    NEGATIVE_WORDS = {
        "awful",
        "bad",
        "boring",
        "disappointing",
        "dull",
        "hate",
        "hated",
        "horrible",
        "poor",
        "terrible",
        "ugly",
        "worst",
    }
    NEGATIONS = {"not", "no", "never", "none", "hardly", "barely"}

    @classmethod
    def featurize(self, text: str) -> Dict[str, float]:
        tokens = _tokenize(text)
        pos_count = sum(token in self.POSITIVE_WORDS for token in tokens)
        neg_count = sum(token in self.NEGATIVE_WORDS for token in tokens)
        negation_count = sum(token in self.NEGATIONS for token in tokens)

        features = {
            "pos_count": float(pos_count),
            "neg_count": float(neg_count),
            "negation_count": float(negation_count),
            "sentiment_margin": float(pos_count - neg_count),
        }
        return self.prefix_with_name(features)


class StyleAndPunctuation(FeatureMap):
    """Simple stylistic cues that can help sentiment and forum classification."""

    name = "style"

    @classmethod
    def featurize(self, text: str) -> Dict[str, float]:
        tokens = text.split()
        num_tokens = len(tokens)
        uppercase_tokens = sum(
            1 for token in tokens if any(c.isalpha() for c in token) and token.isupper()
        )

        if num_tokens <= 5:
            length_bucket = "tok_0_5"
        elif num_tokens <= 20:
            length_bucket = "tok_6_20"
        else:
            length_bucket = "tok_21_plus"

        features = {
            length_bucket: 1.0,
            "num_exclamation": float(min(text.count("!"), 3)),
            "num_question": float(min(text.count("?"), 3)),
            "has_digit": float(any(c.isdigit() for c in text)),
            "uppercase_ratio": (
                float(uppercase_tokens) / float(num_tokens) if num_tokens > 0 else 0.0
            ),
        }
        return self.prefix_with_name(features)


class NewsgroupMetadata(FeatureMap):
    """Metadata-like textual markers commonly found in newsgroup posts."""

    name = "ngmeta"

    @classmethod
    def featurize(self, text: str) -> Dict[str, float]:
        lower_text = text.lower()
        header_markers = [
            "from:",
            "subject:",
            "organization:",
            "distribution:",
            "article-i.d.:",
            "lines:",
        ]
        ret = {
            f"has_{m.replace(':', '').replace('.', '').replace('-', '_')}": 1.0
            for m in header_markers
            if m in lower_text
        }
        if "writes:" in lower_text:
            ret["has_writes"] = 1.0
        if "\n>" in text or text.lstrip().startswith(">"):
            ret["has_quote_block"] = 1.0
        if "@" in text:
            ret["has_email_like_token"] = 1.0
        return self.prefix_with_name(ret)


FEATURE_CLASSES_MAP = {
    c.name: c
    for c in [
        BagOfWords,
        SentenceLength,
        SentimentLexicon,
        StyleAndPunctuation,
        NewsgroupMetadata,
    ]
}


def make_featurize(
    feature_types: Set[str],
) -> Callable[[str], Dict[str, float]]:
    featurize_fns = [FEATURE_CLASSES_MAP[n].featurize for n in feature_types]

    def _featurize(text: str):
        f = ChainMap(*[fn(text) for fn in featurize_fns])
        return dict(f)

    return _featurize


__all__ = ["make_featurize"]

if __name__ == "__main__":
    text = "I love this movie"
    print(text)
    print(BagOfWords.featurize(text))
    featurize = make_featurize({"bow", "len"})
    print(featurize(text))
