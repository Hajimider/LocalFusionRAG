"""下载并转换 C-MTEB/T2Reranking 为本项目的可控检索子集。"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from pathlib import Path


DATASET_NAME = "C-MTEB/T2Reranking"


def enable_system_trust_store() -> None:
    """让 Conda Python 复用系统已信任的 CA 证书。"""
    try:
        import truststore
    except ImportError:
        try:
            from pip._vendor import truststore
        except ImportError:
            return
    truststore.inject_into_ssl()


def safe_id(value: object) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_")
    return text[:80] or hashlib.sha1(str(value).encode("utf-8")).hexdigest()[:16]


def first(record: dict, names: tuple[str, ...], default=None):
    for name in names:
        if name in record:
            return record[name]
    return default


def load_public_rows(load_dataset):
    """读取带显式 positive/negative 标签的公开重排序数据。"""
    try:
        return load_dataset(DATASET_NAME, split="dev")
    except (ValueError, KeyError):
        dataset = load_dataset(DATASET_NAME)
        if "dev" in dataset:
            return dataset["dev"]
        first_split = next(iter(dataset))
        return dataset[first_split]


def convert(output: Path, queries_limit: int, corpus_limit: int, seed: int) -> dict:
    enable_system_trust_store()
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("请先安装公开数据集依赖：python -m pip install datasets") from exc

    rows = load_public_rows(load_dataset)
    rng = random.Random(seed)
    row_indices = list(range(len(rows)))
    rng.shuffle(row_indices)
    selected_indices = row_indices[:queries_limit]
    selected_rows = rows.select(selected_indices) if hasattr(rows, "select") else [rows[i] for i in selected_indices]
    corpus: dict[str, dict[str, str]] = {}
    queries: dict[str, str] = {}
    qrels: dict[str, set[str]] = {}
    for index, row in zip(selected_indices, selected_rows):
        query = str(first(row, ("query", "question", "text"), "")).strip()
        positives = first(row, ("positive", "positives"), []) or []
        negatives = first(row, ("negative", "negatives"), []) or []
        if not query:
            continue
        if isinstance(positives, str):
            positives = [positives]
        if isinstance(negatives, str):
            negatives = [negatives]
        query_id = f"q_{index:06d}"
        queries[query_id] = query
        positive_texts = {str(item).strip() for item in positives if str(item).strip()}
        relevant = set()
        for text in [*positives, *negatives]:
            text = str(text).strip()
            if not text:
                continue
            doc_id = "doc_" + hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]
            corpus.setdefault(doc_id, {"title": "", "text": text})
            if text in positive_texts:
                relevant.add(doc_id)
        if relevant:
            qrels[query_id] = relevant

    selected_queries = sorted(query_id for query_id in qrels if qrels[query_id])
    relevant_ids = {doc_id for query_id in selected_queries for doc_id in qrels[query_id]}
    negatives = sorted(set(corpus) - relevant_ids)
    rng.shuffle(negatives)
    selected_docs = sorted(relevant_ids | set(negatives[: max(0, corpus_limit - len(relevant_ids))]))

    docs_dir = output / "knowledge_base"
    eval_dir = output / "evaluation"
    docs_dir.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)
    doc_files = {}
    for doc_id in selected_docs:
        filename = f"doc_{safe_id(doc_id)}.txt"
        doc_files[doc_id] = filename
        record = corpus[doc_id]
        body = f"{record['title']}\n\n{record['text']}" if record["title"] else record["text"]
        (docs_dir / filename).write_text(body, encoding="utf-8")

    cases = []
    for query_id in selected_queries:
        targets = [doc_files[doc_id] for doc_id in sorted(qrels[query_id]) if doc_id in doc_files]
        if targets:
            cases.append(
                {
                    "id": query_id,
                    "question": queries[query_id],
                    "question_type": "public_retrieval",
                    "difficulty": "public",
                    "expected_sources": targets,
                }
            )
    with (eval_dir / "questions.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for case in cases:
            handle.write(json.dumps(case, ensure_ascii=False) + "\n")

    manifest = {
        "dataset": DATASET_NAME,
        "seed": seed,
        "queries": len(cases),
        "documents": len(selected_docs),
        "relevant_documents": len(relevant_ids),
        "license": "以数据集主页声明为准",
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="准备公开中文检索基准子集")
    parser.add_argument("--output", type=Path, default=Path("data/public_t2reranking"))
    parser.add_argument("--queries", type=int, default=100)
    parser.add_argument("--corpus", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260813)
    args = parser.parse_args()
    if args.queries <= 0 or args.corpus <= 0:
        raise SystemExit("queries 和 corpus 必须大于 0。")
    print(json.dumps(convert(args.output, args.queries, args.corpus, args.seed), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
