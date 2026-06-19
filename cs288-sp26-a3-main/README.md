# CS288 Assignment 3

This repository contains an offline-first QA pipeline for the early milestone.

## Build artifacts

```bash
python3 -m src.build_artifacts --output-dir artifacts
```

This step crawls Berkeley EECS pages, writes `artifacts/corpus.jsonl`, and builds `artifacts/indexes/sparse_passages.pkl`.

The repository already includes a built corpus, sparse index, and a local extractive QA checkpoint in `artifacts/`, so you do not need to rebuild before running predictions.

`llm.py` is not modified. If `OPENROUTER_API_KEY` is present, the pipeline can use it as a fallback extractor for cases where the local reader is low confidence.

## Run predictions

```bash
bash run.sh data/questions.txt data/answers.txt
```

`run.sh` reads one question per line and writes one answer per line.
