from __future__ import annotations

import html
import re
import unicodedata


space_pattern = re.compile(r"\s+")
token_pattern = re.compile(r"[A-Za-z0-9][A-Za-z0-9@._:/-]*")
sentence_boundary_pattern = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")
date_pattern = re.compile(
    r"(?:"
    r"\b(?:jan|january|feb|february|mar|march|apr|april|may|jun|june|jul|july|aug|august|"
    r"sep|sept|september|oct|october|nov|november|dec|december)\.?\s+\d{1,2}(?:,\s*\d{2,4})?"
    r"|"
    r"\b\d{1,2}/\d{1,2}/\d{2,4}\b"
    r"|"
    r"\b\d{4}-\d{2}-\d{2}\b"
    r")",
    re.IGNORECASE,
)
email_pattern = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
course_pattern = re.compile(r"\b[A-Z]{2,4}\s?\d{1,3}[A-Z]?\b")
number_pattern = re.compile(r"\b\d+(?:,\d{3})*(?:\.\d+)?(?:\+|%)?\b")
title_case_pattern = re.compile(r"\b(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,6}|[A-Z]{2,}(?:\s+[A-Z]{2,}){0,4})\b")
stop_words = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "did",
    "do",
    "does",
    "for",
    "from",
    "has",
    "have",
    "how",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "their",
    "this",
    "to",
    "was",
    "were",
    "what",
    "when",
    "where",
    "which",
    "who",
    "whose",
    "why",
    "with",
}
number_words = {
    "zero",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
    "eleven",
    "twelve",
    "thirteen",
    "fourteen",
    "fifteen",
    "sixteen",
    "seventeen",
    "eighteen",
    "nineteen",
    "twenty",
    "thirty",
    "forty",
    "fifty",
    "sixty",
    "seventy",
    "eighty",
    "ninety",
    "hundred",
}


def clean_text(text: str) -> str:
    normalized_text = unicodedata.normalize("NFKC", html.unescape(text or ""))
    normalized_text = normalized_text.replace("\xa0", " ")
    return normalized_text


def normalize_space(text: str) -> str:
    return space_pattern.sub(" ", clean_text(text)).strip()


def tokenize(text: str) -> list[str]:
    return [token.lower() for token in token_pattern.findall(clean_text(text))]


def significant_tokens(text: str) -> list[str]:
    return [token for token in tokenize(text) if token not in stop_words]


def split_sentences(text: str) -> list[str]:
    lines = [normalize_space(line) for line in clean_text(text).splitlines()]
    sentences: list[str] = []
    for line in lines:
        if not line:
            continue
        parts = sentence_boundary_pattern.split(line)
        for part in parts:
            part = normalize_space(part)
            if part:
                sentences.append(part)
    return sentences


def build_passages(text: str, max_words: int = 120, overlap_words: int = 30) -> list[str]:
    sentences = split_sentences(text)
    passages: list[str] = []
    current_sentences: list[str] = []
    current_word_count = 0
    for sentence in sentences:
        sentence_word_count = len(sentence.split())
        if current_sentences and current_word_count + sentence_word_count > max_words:
            passage = normalize_space(" ".join(current_sentences))
            if passage:
                passages.append(passage)
            overlap_sentences: list[str] = []
            overlap_count = 0
            for previous_sentence in reversed(current_sentences):
                overlap_sentences.insert(0, previous_sentence)
                overlap_count += len(previous_sentence.split())
                if overlap_count >= overlap_words:
                    break
            current_sentences = overlap_sentences
            current_word_count = sum(len(item.split()) for item in current_sentences)
        current_sentences.append(sentence)
        current_word_count += sentence_word_count
    if current_sentences:
        passage = normalize_space(" ".join(current_sentences))
        if passage:
            passages.append(passage)
    return passages


def clean_answer(text: str) -> str:
    answer = normalize_space(text)
    answer = answer.strip(" \t\r\n-:;,.")
    answer = answer.replace("\n", " ")
    return normalize_space(answer)


def answer_kind(question: str) -> str:
    lowered_question = question.lower()
    if lowered_question.startswith(("is ", "are ", "was ", "were ", "does ", "do ", "did ", "has ", "have ", "can ")):
        return "boolean"
    if "email" in lowered_question or "e-mail" in lowered_question:
        return "email"
    if lowered_question.startswith("when ") or "date" in lowered_question or "deadline" in lowered_question:
        return "date"
    if lowered_question.startswith("how many") or lowered_question.startswith("how much") or "number of" in lowered_question:
        return "number"
    if "course number" in lowered_question or "course" in lowered_question:
        return "course"
    if lowered_question.startswith("who ") or "advisor" in lowered_question or "professor" in lowered_question:
        return "person"
    if lowered_question.startswith("where ") or "which university" in lowered_question or "which school" in lowered_question:
        return "place"
    return "text"


def extract_pattern_matches(text: str, kind: str) -> list[str]:
    cleaned_text = clean_text(text)
    matches: list[str] = []
    if kind == "email":
        matches.extend(email_pattern.findall(cleaned_text))
    elif kind == "date":
        matches.extend(match.group(0) for match in date_pattern.finditer(cleaned_text))
    elif kind == "number":
        matches.extend(number_pattern.findall(cleaned_text))
        for token in tokenize(cleaned_text):
            if token in number_words:
                matches.append(token)
    elif kind == "course":
        matches.extend(course_pattern.findall(cleaned_text))
    elif kind in {"person", "place"}:
        matches.extend(match.group(0) for match in title_case_pattern.finditer(cleaned_text))
    return [clean_answer(match) for match in matches if clean_answer(match)]


def token_overlap(left_text: str, right_text: str) -> int:
    left_tokens = set(significant_tokens(left_text))
    right_tokens = set(significant_tokens(right_text))
    return len(left_tokens & right_tokens)
