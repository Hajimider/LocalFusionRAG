"""对同一公开评测子集运行消融和参数实验。"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def run(command: list[str]) -> None:
    print("运行：", " ".join(command))
    subprocess.run(command, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="运行 RAG 检索参数实验")
    parser.add_argument("--embedding-model", required=True)
    parser.add_argument("--reranker-model", default="")
    parser.add_argument("--data", type=Path, default=Path("data/public_t2reranking"))
    parser.add_argument("--output", type=Path, default=Path("outputs/public_benchmark"))
    parser.add_argument("--grid", action="store_true", help="运行全部四组参数网格；CPU 耗时较长")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    configurations = [
        {"chunk_size": 300, "chunk_overlap": 50, "top_k": 5, "max_distance": 1.2},
        {"chunk_size": 500, "chunk_overlap": 80, "top_k": 5, "max_distance": 1.2},
        {"chunk_size": 700, "chunk_overlap": 100, "top_k": 5, "max_distance": 1.2},
        {"chunk_size": 500, "chunk_overlap": 80, "top_k": 10, "max_distance": 1.2},
    ]
    if not args.grid:
        configurations = configurations[:1]
    index_dir = args.output / "faiss"
    cases = args.data / "evaluation" / "questions.jsonl"
    summary = []
    for config in configurations:
        label = "c{chunk_size}_o{chunk_overlap}_k{top_k}".format(**config)
        run(
            [
                sys.executable,
                "run_project.py",
                "build",
                "--docs",
                str(args.data / "knowledge_base"),
                "--index",
                str(index_dir),
                "--embedding-model",
                args.embedding_model,
                "--chunk-size",
                str(config["chunk_size"]),
                "--chunk-overlap",
                str(config["chunk_overlap"]),
            ]
        )
        report = args.output / f"{label}.json"
        command = [
            sys.executable,
            "run_project.py",
            "evaluate",
            "--retrieval-only",
            "--cases",
            str(cases),
            "--report",
            str(report),
            "--index",
            str(index_dir),
            "--embedding-model",
            args.embedding_model,
            "--top-k",
            str(config["top_k"]),
            "--max-distance",
            str(config["max_distance"]),
        ]
        if args.reranker_model:
            command.extend(["--reranker-model", args.reranker_model])
        run(command)
        result = json.loads(report.read_text(encoding="utf-8"))
        summary.append({"name": label, "parameters": config, "results": {
            name: value["metrics"] for name, value in result["retrieval_ablation"].items()
        }})
    (args.output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
