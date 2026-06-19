from __future__ import annotations

import argparse
from pathlib import Path

from .rag_pipeline import RagPipeline
from .text_utils import clean_answer


def read_questions(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8") as handle:
        return [line.rstrip("\n") for line in handle]


def write_answers(path: Path, answers: list[str]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for answer in answers:
            handle.write(clean_answer(answer) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("questions_path")
    parser.add_argument("predictions_path")
    parser.add_argument("--artifact-dir", default=str(Path(__file__).resolve().parents[1] / "artifacts"))
    arguments = parser.parse_args()
    pipeline = RagPipeline(arguments.artifact_dir)
    questions = read_questions(Path(arguments.questions_path))
    try:
        answers = pipeline.answer_questions(questions)
    except Exception:
        answers = []
        for question in questions:
            try:
                answers.append(pipeline.answer_question(question))
            except Exception:
                answers.append("")
    write_answers(Path(arguments.predictions_path), answers)


if __name__ == "__main__":
    main()
