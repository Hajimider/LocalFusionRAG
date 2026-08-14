import os
import sys
import threading
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

from rag_core import (
    BM25Retriever,
    OpenAICompatibleLLM,
    RAGEngine,
    reciprocal_rank_fusion,
    resolve_model_path,
    tokenize_for_bm25,
    validate_document_name,
)
from run_project import (
    citations_are_valid,
    expected_sources,
    load_cases,
    rank_relevant_sources,
    retrieval_metrics,
    serve_command,
    source_matches,
    write_error_cases,
)


class FakeEmbedding:
    def embed_query(self, _):
        return [1.0, 0.0]

    def embed_documents(self, documents):
        return [[1.0, 0.0] if "相关" in document else [0.0, 1.0] for document in documents]


class BrokenEmbedding:
    def embed_query(self, _):
        raise RuntimeError("embedding failed")

    def embed_documents(self, _):
        raise RuntimeError("embedding failed")


def test_validate_document_name_accepts_supported_files():
    assert validate_document_name("资料.md") == "资料.md"
    assert validate_document_name("nested\\guide.pdf") == "guide.pdf"


def test_validate_document_name_rejects_unsupported_files():
    with pytest.raises(ValueError):
        validate_document_name("secret.exe")


def test_resolve_model_path_rejects_missing_path():
    with pytest.raises(FileNotFoundError):
        resolve_model_path("__missing_model_directory_for_test__")


def test_bm25_handles_chinese_and_code_terms():
    documents = [
        SimpleNamespace(page_content="提交代码前运行 git status 和 git add。"),
        SimpleNamespace(page_content="Python 列表是可变序列。"),
    ]
    results = BM25Retriever(documents).search("Git 提交流程", k=2)
    assert results[0][0] is documents[0]
    assert "提交" in tokenize_for_bm25("提交代码")
    assert "提" in tokenize_for_bm25("提交代码")


def test_bm25_rejects_weak_common_word_overlap():
    documents = [SimpleNamespace(page_content="公司内部技术文档"), SimpleNamespace(page_content="Python 基础知识")]
    assert BM25Retriever(documents).search("公司今年的年终奖具体是多少？", k=2) == []


def test_reciprocal_rank_fusion_rewards_two_route_hits():
    scores = reciprocal_rank_fusion(
        {"dense": [("a",), ("b",)], "bm25": [("b",), ("c",)]},
        weights={"dense": 0.65, "bm25": 0.35},
    )
    assert scores[("b",)] > scores[("a",)]
    assert scores[("b",)] > scores[("c",)]


def test_retrieval_metrics_separates_ranking_and_rejection():
    metrics = retrieval_metrics(
        [
            {"expected_source": "a.md", "rank": 1, "retrieval_hit": True, "reranked": False},
            {"expected_source": "b.md", "rank": 2, "retrieval_hit": True, "reranked": True},
            {"expected_source": None, "rank": None, "retrieval_hit": True, "reranked": False},
        ]
    )
    assert metrics["retrieval_accuracy"] == 1.0
    assert metrics["mrr"] == 0.75
    assert metrics["rejection_accuracy"] == 1.0


def test_retrieval_metrics_supports_multiple_relevant_sources():
    metrics = retrieval_metrics(
        [
            {
                "expected_sources": ["a.md", "b.md"],
                "ranks": [1],
                "retrieval_hit": True,
                "reranked": False,
                "top_k": 2,
            }
        ]
    )
    assert metrics["hit_at_k"] == 1.0
    assert metrics["recall_at_k"] == 0.5
    assert metrics["mrr"] == 1.0


def test_relevant_source_rank_does_not_count_duplicate_chunks_twice():
    first = SimpleNamespace(file="a.md")
    second = SimpleNamespace(file="a.md")
    third = SimpleNamespace(file="b.md")
    assert rank_relevant_sources([first, second, third], ["a.md", "b.md"]) == [1, 3]


def test_case_helpers_support_jsonl_and_legacy_source(tmp_path):
    path = tmp_path / "cases.jsonl"
    path.write_text('{"question":"q","expected_source":"a.md"}\n', encoding="utf-8")
    case = load_cases(path)[0]
    assert expected_sources(case) == ["a.md"]


def test_citation_validation_rejects_missing_and_out_of_range_references():
    assert citations_are_valid("依据见 [资料1]。", 2)
    assert not citations_are_valid("没有引用。", 2)
    assert not citations_are_valid("依据见 [资料3]。", 2)


def test_error_case_export_only_writes_failures(tmp_path):
    report = tmp_path / "report.json"
    write_error_cases(
        report,
        {
            "dense": {
                "cases": [
                    {"id": "ok", "retrieval_hit": True},
                    {"id": "bad", "retrieval_hit": False},
                ]
            }
        },
    )
    exported = load_cases(tmp_path / "report_errors.jsonl")
    assert [item["id"] for item in exported] == ["bad"]


def test_source_matching_uses_exact_relative_path():
    assert source_matches("docs/rag_basics.md", "docs/rag_basics.md")
    assert not source_matches("docs/not_rag_basics.md", "rag_basics.md")


def test_reranker_load_failure_uses_embedding_fallback(monkeypatch):
    module = types.ModuleType("sentence_transformers")

    class BrokenCrossEncoder:
        def __init__(self, *args, **kwargs):
            raise OSError("model unavailable")

    module.CrossEncoder = BrokenCrossEncoder
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)
    engine = object.__new__(RAGEngine)
    engine.index_dir = Path("storage/faiss")
    engine.reranker_model = "missing-reranker"
    engine._reranker = None
    engine._reranker_kind = "not_loaded"
    engine._reranker_error = None
    engine._reranker_lock = threading.RLock()
    engine._store_lock = threading.RLock()
    engine._store = SimpleNamespace(embedding_function=FakeEmbedding())

    assert engine._get_reranker() is not None
    assert engine.reranker_status == "embedding_fallback"


def test_reranker_inference_failure_switches_to_embedding_fallback():
    relevant = SimpleNamespace(page_content="相关资料", metadata={"source": "a.md", "chunk": 0})
    irrelevant = SimpleNamespace(page_content="其他资料", metadata={"source": "b.md", "chunk": 1})

    class FakeStore:
        embedding_function = FakeEmbedding()

        @staticmethod
        def similarity_search_with_score(*_args, **_kwargs):
            return [(irrelevant, 0.2), (relevant, 0.3)]

    class BrokenPredictor:
        @staticmethod
        def predict(*_args, **_kwargs):
            raise RuntimeError("inference failed")

    engine = object.__new__(RAGEngine)
    engine.index_dir = Path("storage/faiss")
    engine._store = FakeStore()
    engine._bm25 = BM25Retriever([relevant, irrelevant])
    engine._store_lock = threading.RLock()
    engine._reranker = BrokenPredictor()
    engine._reranker_kind = "cross_encoder"
    engine._reranker_error = None
    engine._reranker_lock = threading.RLock()

    result = engine.retrieve_advanced("相关问题", top_k=2, mode="dense", rerank=True)

    assert result.reranker_backend == "embedding_fallback"
    assert result.sources[0].file == "a.md"


def test_second_reranker_failure_reports_rrf_fallback():
    document = SimpleNamespace(page_content="相关资料", metadata={"source": "a.md", "chunk": 0})

    class FakeStore:
        embedding_function = BrokenEmbedding()

        @staticmethod
        def similarity_search_with_score(*_args, **_kwargs):
            return [(document, 0.2)]

    class BrokenPredictor:
        @staticmethod
        def predict(*_args, **_kwargs):
            raise RuntimeError("inference failed")

    engine = object.__new__(RAGEngine)
    engine.index_dir = Path("storage/faiss")
    engine._store = FakeStore()
    engine._bm25 = BM25Retriever([document])
    engine._store_lock = threading.RLock()
    engine._reranker = BrokenPredictor()
    engine._reranker_kind = "cross_encoder"
    engine._reranker_error = None
    engine._reranker_lock = threading.RLock()

    result = engine.retrieve_advanced("相关问题", top_k=1, mode="dense", rerank=True)

    assert not result.reranked
    assert result.reranker_backend == "rrf_fallback"


def test_min_rerank_score_filters_low_confidence_candidates():
    low = SimpleNamespace(page_content="弱相关", metadata={"source": "low.md", "chunk": 0})
    high = SimpleNamespace(page_content="强相关", metadata={"source": "high.md", "chunk": 1})

    class FakeStore:
        embedding_function = FakeEmbedding()

        @staticmethod
        def similarity_search_with_score(*_args, **_kwargs):
            return [(low, 0.2), (high, 0.3)]

    class FixedPredictor:
        @staticmethod
        def predict(*_args, **_kwargs):
            return [0.1, 0.9]

    engine = object.__new__(RAGEngine)
    engine.index_dir = Path("storage/faiss")
    engine._store = FakeStore()
    engine._bm25 = BM25Retriever([low, high])
    engine._store_lock = threading.RLock()
    engine._reranker = FixedPredictor()
    engine._reranker_kind = "cross_encoder"
    engine._reranker_error = None
    engine._reranker_lock = threading.RLock()

    result = engine.retrieve_advanced(
        "相关问题", top_k=2, mode="dense", rerank=True, min_rerank_score=0.5
    )
    assert [source.file for source in result.sources] == ["high.md"]


def test_cross_encoder_threshold_is_not_reused_for_embedding_fallback():
    document = SimpleNamespace(page_content="相关资料", metadata={"source": "a.md", "chunk": 0})

    class FakeStore:
        embedding_function = FakeEmbedding()

        @staticmethod
        def similarity_search_with_score(*_args, **_kwargs):
            return [(document, 0.2)]

    engine = object.__new__(RAGEngine)
    engine.index_dir = Path("storage/faiss")
    engine._store = FakeStore()
    engine._bm25 = BM25Retriever([document])
    engine._store_lock = threading.RLock()
    engine._reranker = SimpleNamespace(predict=lambda *_args, **_kwargs: [0.1])
    engine._reranker_kind = "embedding_fallback"
    engine._reranker_error = "cross encoder unavailable"
    engine._reranker_lock = threading.RLock()

    result = engine.retrieve_advanced(
        "相关问题", top_k=1, mode="dense", rerank=True, min_rerank_score=0.5
    )
    assert [source.file for source in result.sources] == ["a.md"]


def test_custom_index_directory_comes_from_environment(monkeypatch):
    from app import index_dir_from_env, min_rerank_score_from_env

    monkeypatch.setenv("RAG_INDEX_DIR", "storage/custom-index")
    assert index_dir_from_env() == Path("storage/custom-index").resolve()
    monkeypatch.delenv("RAG_MIN_RERANK_SCORE", raising=False)
    assert min_rerank_score_from_env() is None
    monkeypatch.setenv("RAG_MIN_RERANK_SCORE", "0.28")
    assert min_rerank_score_from_env() == 0.28


def test_serve_command_accepts_path_model(monkeypatch):
    calls = {}
    monkeypatch.setitem(
        sys.modules,
        "uvicorn",
        SimpleNamespace(run=lambda app, **kwargs: calls.update(app=app, **kwargs)),
    )
    serve_command(
        SimpleNamespace(
            model=Path("external-model"),
            provider="local",
            api_base_url="",
            api_model="",
            embedding_model="embedding-model",
            reranker_model="reranker-model",
            context_size=4096,
            max_tokens=256,
            min_rerank_score=None,
            index=Path("storage/faiss"),
            host="127.0.0.1",
            port=8000,
        )
    )

    assert os.environ["RAG_MODEL_PATH"] == "external-model"
    assert calls == {"app": "app:app", "host": "127.0.0.1", "port": 8000, "reload": False}


def test_serve_command_configures_api_provider(monkeypatch):
    monkeypatch.setenv("RAG_API_KEY", "secret")
    monkeypatch.setitem(sys.modules, "uvicorn", SimpleNamespace(run=lambda *_args, **_kwargs: None))
    serve_command(
        SimpleNamespace(
            model=None,
            provider="api",
            api_base_url="https://example.com/v1",
            api_model="test-model",
            embedding_model="embedding-model",
            reranker_model="reranker-model",
            context_size=4096,
            max_tokens=256,
            min_rerank_score=None,
            index=Path("storage/faiss"),
            host="127.0.0.1",
            port=8000,
        )
    )

    assert os.environ["RAG_LLM_PROVIDER"] == "api"
    assert os.environ["RAG_API_BASE_URL"] == "https://example.com/v1"
    assert os.environ["RAG_API_MODEL"] == "test-model"


def test_openai_compatible_llm_parses_stream(monkeypatch):
    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def __iter__(self):
            return iter(
                [
                    b'data: {"choices":[]}\n',
                    b'data: {"choices":[{"delta":{"role":"assistant"}}]}\n',
                    b'data: {"choices":[{"delta":{"content":"hello"}}]}\n',
                    b'data: {"choices":[{"delta":{"content":" world"}}]}\n',
                    b'data: [DONE]\n',
                ]
            )

    monkeypatch.setattr("rag_core.urlopen", lambda *_args, **_kwargs: FakeResponse())
    llm = OpenAICompatibleLLM("https://example.com/v1", "secret", "test-model")

    assert llm.complete([{"role": "user", "content": "hi"}]) == "hello world"


def test_query_rewrite_failure_returns_original_question():
    class BrokenLLM:
        @staticmethod
        def complete(*_args, **_kwargs):
            raise RuntimeError("generation failed")

    engine = object.__new__(RAGEngine)
    engine._llm = BrokenLLM()

    assert engine.rewrite_question("原始问题") == ("原始问题", False)


def test_public_benchmark_converter_preserves_positive_labels(tmp_path, monkeypatch):
    from scripts import prepare_public_benchmark as benchmark

    rows = [
        {"query": "问题一", "positive": ["正确文档一"], "negative": ["错误文档一"]},
        {"query": "问题二", "positive": ["正确文档二"], "negative": ["错误文档二"]},
    ]

    class FakeRows(list):
        def select(self, indices):
            return [self[index] for index in indices]

    fake_datasets = types.ModuleType("datasets")
    fake_datasets.load_dataset = lambda *_args, **_kwargs: FakeRows(rows)
    monkeypatch.setitem(sys.modules, "datasets", fake_datasets)
    monkeypatch.setattr(benchmark, "enable_system_trust_store", lambda: None)

    manifest = benchmark.convert(tmp_path, queries_limit=2, corpus_limit=4, seed=7)
    cases = [
        __import__("json").loads(line)
        for line in (tmp_path / "evaluation" / "questions.jsonl").read_text(encoding="utf-8").splitlines()
    ]

    assert manifest["dataset"] == "C-MTEB/T2Reranking"
    assert manifest["queries"] == 2
    assert manifest["documents"] == 4
    assert all(len(case["expected_sources"]) == 1 for case in cases)
