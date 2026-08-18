from __future__ import annotations

import argparse
import json
import math
import os
import re
from pathlib import Path, PurePosixPath

from rag_core import DEFAULT_EMBEDDING_MODEL, DEFAULT_RERANKER_MODEL, RAGEngine, build_index


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DOCS = BASE_DIR / "knowledge_base" / "legal_docs"
DEFAULT_INDEX = BASE_DIR / "storage" / "faiss"
DEFAULT_CASES = BASE_DIR / "evaluation" / "questions.json"
DEFAULT_REPORT = BASE_DIR / "storage" / "evaluation_report.json"


def positive_int(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("必须是大于 0 的整数")
    return number


def nonnegative_int(value: str) -> int:
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("必须是大于等于 0 的整数")
    return number


def positive_float(value: str) -> float:
    number = float(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("必须是大于 0 的数字")
    return number


def source_matches(actual: str, expected: str) -> bool:
    return PurePosixPath(actual.replace("\\", "/")) == PurePosixPath(expected.replace("\\", "/"))


def load_cases(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if path.suffix.lower() == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    value = json.loads(text)
    if not isinstance(value, list):
        raise ValueError("评测文件必须是 JSON 数组或 JSONL。")
    return value


def expected_sources(case: dict) -> list[str]:
    sources = case.get("expected_sources")
    if sources is None:
        source = case.get("expected_source")
        return [] if source is None else [source]
    return [str(source) for source in sources]


def rank_relevant_sources(retrieved_sources: list, expected: list[str]) -> list[int]:
    ranks = []
    for target in expected:
        matching_ranks = [
            rank
            for rank, source in enumerate(retrieved_sources, start=1)
            if source_matches(source.file, target)
        ]
        if matching_ranks:
            ranks.append(min(matching_ranks))
    return sorted(ranks)


def citations_are_valid(answer: str, source_count: int) -> bool:
    citations = [int(value) for value in re.findall(r"\[资料\s*(\d+)\]", answer)]
    return bool(citations) and all(1 <= value <= source_count for value in citations)


def add_rag_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--api-base-url", default="", help="OpenAI 兼容 API 地址，例如 https://example.com/v1")
    parser.add_argument("--api-model", default="", help="API 模型名称")
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL, help="BGE 模型名称或本地目录")
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX, help="FAISS 索引目录")
    parser.add_argument("--max-tokens", type=int, default=512, help="最大生成 token 数")
    parser.add_argument("--reranker-model", default=DEFAULT_RERANKER_MODEL, help="CrossEncoder 重排序模型")


def add_retrieval_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--top-k", type=positive_int, default=4)
    parser.add_argument("--max-distance", type=positive_float, default=1.2)
    parser.add_argument("--retrieval-mode", choices=("dense", "bm25", "hybrid"), default="hybrid")
    parser.add_argument("--document-type", choices=("all", "law", "case"), default="all")
    parser.add_argument("--validity", choices=("all", "current", "historical", "unknown"), default="all")
    parser.add_argument("--no-rerank", action="store_true", help="关闭 CrossEncoder 重排序")
    parser.add_argument("--no-rewrite", action="store_true", help="关闭查询改写")
    parser.add_argument(
        "--min-rerank-score",
        type=float,
        default=None,
        help="CrossEncoder 最低相关性分数；应由校准集选择，不建议手工猜测",
    )


def build_command(args: argparse.Namespace) -> None:
    if args.chunk_overlap >= args.chunk_size:
        raise ValueError("chunk-overlap 必须小于 chunk-size。")
    result = build_index(
        args.docs,
        args.index,
        args.embedding_model,
        args.chunk_size,
        args.chunk_overlap,
    )
    print("知识库索引创建完成：")
    print(json.dumps(result, ensure_ascii=False, indent=2))


def create_engine(args: argparse.Namespace) -> RAGEngine:
    return RAGEngine(
        index_dir=args.index,
        embedding_model=args.embedding_model,
        max_tokens=args.max_tokens,
        reranker_model=args.reranker_model,
        api_base_url=args.api_base_url,
        api_key=os.getenv("RAG_API_KEY", ""),
        api_model=args.api_model,
        domain_profile=os.getenv("RAG_DOMAIN_PROFILE", "legal_assistant"),
        intent_routing=os.getenv("RAG_INTENT_ROUTING", "rule"),
    )


def ask_command(args: argparse.Namespace) -> None:
    engine = create_engine(args)
    retrieval, tokens = engine.stream_answer(
        args.question,
        use_rag=not args.no_rag,
        top_k=args.top_k,
        max_distance=args.max_distance,
        retrieval_mode=args.retrieval_mode,
        rerank=not args.no_rerank,
        rewrite_query=not args.no_rewrite,
        min_rerank_score=args.min_rerank_score,
        document_type=args.document_type,
        validity=args.validity,
    )
    if retrieval.sources:
        print(
            f"===== 检索信息 =====\n查询：{retrieval.query}\n"
            f"方式：{retrieval.mode}，查询改写：{'是' if retrieval.rewrite_applied else '否'}，"
            f"重排序：{retrieval.reranker_backend if retrieval.reranker_backend != 'none' else '否'}，"
            f"意图：{retrieval.intent}，生成链：{retrieval.generation_chain}"
        )
        print("===== 检索来源 =====")
        for number, source in enumerate(retrieval.sources, start=1):
            page = f"，第 {source.page} 页" if source.page else ""
            score = f"，重排分数={source.rerank_score}" if source.rerank_score is not None else ""
            print(f"[{number}] {source.file}{page}，召回={'+'.join(source.methods)}{score}")
        print("\n===== 回答 =====")
    for token in tokens:
        print(token, end="", flush=True)
    print()


def serve_command(args: argparse.Namespace) -> None:
    os.environ["RAG_API_BASE_URL"] = args.api_base_url
    os.environ["RAG_API_MODEL"] = args.api_model
    os.environ["RAG_EMBEDDING_MODEL"] = str(args.embedding_model)
    os.environ["RAG_MAX_TOKENS"] = str(args.max_tokens)
    os.environ["RAG_RERANKER_MODEL"] = str(args.reranker_model)
    os.environ["RAG_INDEX_DIR"] = str(args.index.resolve())
    os.environ["RAG_MIN_RERANK_SCORE"] = (
        "" if args.min_rerank_score is None else str(args.min_rerank_score)
    )
    os.environ.setdefault("RAG_DOMAIN_PROFILE", "legal_assistant")
    os.environ.setdefault("RAG_INTENT_ROUTING", "rule")

    import uvicorn

    uvicorn.run("app:app", host=args.host, port=args.port, reload=False)


def retrieval_metrics(results: list[dict]) -> dict:
    if not results:
        return {}
    normalized = []
    for item in results:
        copied = dict(item)
        if "expected_sources" not in copied:
            source = copied.get("expected_source")
            copied["expected_sources"] = [] if source is None else [source]
        if "ranks" not in copied:
            rank = copied.get("rank")
            copied["ranks"] = [] if rank is None else [rank]
        copied.setdefault("top_k", max(copied["ranks"], default=1))
        normalized.append(copied)
    results = normalized
    answerable = [item for item in results if item["expected_sources"]]
    unanswerable = [item for item in results if not item["expected_sources"]]
    reciprocal_ranks = [1 / item["ranks"][0] if item["ranks"] else 0 for item in answerable]
    recalls = [
        len(item["ranks"]) / len(item["expected_sources"])
        for item in answerable
    ]
    ndcgs = []
    for item in answerable:
        dcg = sum(1 / math.log2(rank + 1) for rank in item["ranks"])
        ideal_count = min(len(item["expected_sources"]), item["top_k"])
        ideal = sum(1 / math.log2(rank + 1) for rank in range(1, ideal_count + 1))
        ndcgs.append(dcg / ideal if ideal else 0.0)
    return {
        "retrieval_accuracy": round(sum(item["retrieval_hit"] for item in results) / len(results), 4),
        "hit_at_k": round(sum(bool(item["ranks"]) for item in answerable) / len(answerable), 4)
        if answerable else None,
        "recall_at_k": round(sum(recalls) / len(recalls), 4) if recalls else None,
        "mrr": round(sum(reciprocal_ranks) / len(reciprocal_ranks), 4) if reciprocal_ranks else None,
        "ndcg_at_k": round(sum(ndcgs) / len(ndcgs), 4) if ndcgs else None,
        "rejection_accuracy": round(
            sum(item["retrieval_hit"] for item in unanswerable) / len(unanswerable), 4
        )
        if unanswerable
        else None,
        "rerank_applied_rate": round(sum(item["reranked"] for item in results) / len(results), 4),
    }


def grouped_metrics(results: list[dict], field: str) -> dict:
    groups: dict[str, list[dict]] = {}
    for item in results:
        groups.setdefault(str(item.get(field, "unknown")), []).append(item)
    return {name: {"count": len(items), **retrieval_metrics(items)} for name, items in groups.items()}


def write_error_cases(report_path: Path, ablation: dict) -> None:
    error_path = report_path.with_name(f"{report_path.stem}_errors.jsonl")
    with error_path.open("w", encoding="utf-8", newline="\n") as handle:
        for configuration, result in ablation.items():
            for case in result["cases"]:
                if not case["retrieval_hit"]:
                    handle.write(
                        json.dumps(
                            {"configuration": configuration, **case},
                            ensure_ascii=False,
                        )
                        + "\n"
                    )


def evaluate_command(args: argparse.Namespace) -> None:
    if not args.retrieval_only and not all(
        (args.api_base_url.strip(), args.api_model.strip(), os.getenv("RAG_API_KEY", "").strip())
    ):
        raise ValueError("完整评测需要配置 API 地址、API 模型和 RAG_API_KEY；只检查检索时请加 --retrieval-only。")
    cases = load_cases(args.cases)[: args.limit or None]
    if not cases:
        raise ValueError("评测题集为空，请检查 cases 文件或 limit 参数。")
    engine = create_engine(args)
    configurations = [
        ("dense", "dense", False),
        ("bm25", "bm25", False),
        ("hybrid", "hybrid", False),
        ("hybrid_rerank", "hybrid", True),
    ]
    ablation = {}
    for label, mode, use_reranker in configurations:
        print(f"===== {label} =====")
        method_results = []
        for number, case in enumerate(cases, start=1):
            print(f"[{number}/{len(cases)}] {case['question']}")
            retrieval = engine.retrieve_advanced(
                case["question"],
                top_k=args.top_k,
                max_distance=args.max_distance,
                mode=mode,
                rerank=use_reranker,
                min_rerank_score=args.min_rerank_score if use_reranker else None,
                document_type=args.document_type,
                validity=args.validity,
            )
            targets = expected_sources(case)
            ranks = rank_relevant_sources(retrieval.sources, targets)
            retrieval_hit = not retrieval.sources if not targets else bool(ranks)
            method_results.append(
                {
                    "id": case.get("id", str(number)),
                    "question": case["question"],
                    "question_type": case.get("question_type", "unknown"),
                    "difficulty": case.get("difficulty", "unknown"),
                    "expected_sources": targets,
                    "ranks": ranks,
                    "rank": ranks[0] if ranks else None,
                    "top_k": args.top_k,
                    "retrieval_hit": retrieval_hit,
                    "reranked": retrieval.reranked,
                    "reranker_backend": retrieval.reranker_backend,
                    "retrieved_sources": [source.to_dict() for source in retrieval.sources],
                }
            )
        ablation[label] = {
            "metrics": retrieval_metrics(method_results),
            "by_question_type": grouped_metrics(method_results, "question_type"),
            "by_difficulty": grouped_metrics(method_results, "difficulty"),
            "cases": method_results,
        }

    generation_cases = []
    if not args.retrieval_only:
        print("===== generation =====")
        for number, case in enumerate(cases, start=1):
            question = case["question"]
            print(f"[{number}/{len(cases)}] {question}")
            rag_messages, retrieval = engine.prepare(
                question,
                True,
                args.top_k,
                args.max_distance,
                args.retrieval_mode,
                not args.no_rerank,
                not args.no_rewrite,
                args.min_rerank_score,
            )
            rag_answer = (
                engine.llm.complete(rag_messages)
                if rag_messages
                else "知识库中没有足够信息，无法回答这个问题。"
            )
            base_messages, _ = engine.prepare(question, False)
            base_answer = engine.llm.complete(base_messages)
            keywords = case.get("expected_keywords", [])
            generation_cases.append(
                {
                    "question": question,
                    "retrieval_query": retrieval.query,
                    "rag_answer": rag_answer,
                    "base_answer": base_answer,
                    "keyword_hit": all(keyword.lower() in rag_answer.lower() for keyword in keywords),
                    "citation_valid": citations_are_valid(rag_answer, len(retrieval.sources))
                    if retrieval.sources else "没有足够信息" in rag_answer,
                }
            )

    keyword_rate = (
        round(sum(item["keyword_hit"] for item in generation_cases) / len(generation_cases), 4)
        if generation_cases
        else None
    )
    citation_rate = (
        round(sum(bool(item["citation_valid"]) for item in generation_cases) / len(generation_cases), 4)
        if generation_cases else None
    )
    best_metrics = ablation["hybrid_rerank"]["metrics"]
    report = {
        "retrieval_accuracy": best_metrics["retrieval_accuracy"],
        "answer_keyword_rate": keyword_rate,
        "retrieval_ablation": ablation,
        "generation": {
            "answer_keyword_rate": keyword_rate,
            "citation_valid_rate": citation_rate,
            "cases": generation_cases,
        },
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    write_error_cases(args.report, ablation)
    print("评测完成：")
    for label, result in ablation.items():
        metrics = result["metrics"]
        print(f"- {label}: Recall@K={metrics['recall_at_k']:.0%}, MRR={metrics['mrr']:.4f}")
    if keyword_rate is not None:
        print(f"RAG 回答关键词命中率：{keyword_rate:.0%}")
        print(f"引用合法率：{citation_rate:.0%}")
    print(f"详细报告：{args.report}")


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="法律混合 RAG 知识库问答系统")
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_parser = subparsers.add_parser("build", help="读取文档并建立 FAISS 索引")
    build_parser.add_argument("--docs", type=Path, default=DEFAULT_DOCS)
    build_parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    build_parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    build_parser.add_argument("--chunk-size", type=positive_int, default=500)
    build_parser.add_argument("--chunk-overlap", type=nonnegative_int, default=80)
    build_parser.set_defaults(handler=build_command)

    ask_parser = subparsers.add_parser("ask", help="在命令行提问")
    add_rag_arguments(ask_parser)
    ask_parser.add_argument("--question", required=True)
    ask_parser.add_argument("--no-rag", action="store_true", help="关闭知识库，仅调用 API 模型")
    add_retrieval_arguments(ask_parser)
    ask_parser.set_defaults(handler=ask_command)

    serve_parser = subparsers.add_parser("serve", help="启动 FastAPI 和网页")
    add_rag_arguments(serve_parser)
    serve_parser.add_argument("--host", default="127.0.0.1")
    serve_parser.add_argument("--port", type=int, default=8000)
    serve_parser.add_argument("--min-rerank-score", type=float, default=None)
    serve_parser.set_defaults(handler=serve_command)

    evaluate_parser = subparsers.add_parser("evaluate", help="自动评测 RAG 效果")
    add_rag_arguments(evaluate_parser)
    evaluate_parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    evaluate_parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    evaluate_parser.add_argument("--limit", type=nonnegative_int, default=0)
    evaluate_parser.add_argument("--retrieval-only", action="store_true")
    add_retrieval_arguments(evaluate_parser)
    evaluate_parser.set_defaults(handler=evaluate_command)

    return parser


def main() -> None:
    args = create_parser().parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
