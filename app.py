from __future__ import annotations

import json
import logging
import os
import uuid
from pathlib import Path
from threading import Lock
from typing import Literal

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from rag_core import (
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_RERANKER_MODEL,
    RAGEngine,
    build_index,
    index_is_ready,
    validate_document_name,
)


BASE_DIR = Path(__file__).resolve().parent
_knowledge_dir_env = os.getenv("RAG_KNOWLEDGE_DIR", "").strip()
_configured_knowledge_dir = Path(
    _knowledge_dir_env or str(BASE_DIR / "knowledge_base" / "legal_docs")
)
_example_knowledge_dir = BASE_DIR / "knowledge_base" / "project_docs"
if _knowledge_dir_env and not _configured_knowledge_dir.is_dir():
    raise RuntimeError(f"RAG_KNOWLEDGE_DIR 指向的目录不存在：{_configured_knowledge_dir}")
if not _knowledge_dir_env and not _configured_knowledge_dir.exists() and _example_knowledge_dir.is_dir():
    _configured_knowledge_dir = _example_knowledge_dir
KNOWLEDGE_DIR = _configured_knowledge_dir.resolve()
UPLOAD_DIR = KNOWLEDGE_DIR / "uploads"
DEFAULT_INDEX_DIR = BASE_DIR / "storage" / "faiss"
WEB_DIR = BASE_DIR / "web"
MAX_UPLOAD_BYTES = 20 * 1024 * 1024

app = FastAPI(title="法律混合 RAG 知识库问答系统", version="2.0.0")
app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")

_engine: RAGEngine | None = None
_engine_lock = Lock()
_index_build_lock = Lock()
logger = logging.getLogger(__name__)


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=500)
    use_rag: bool = True
    top_k: int = Field(default=4, ge=1, le=8)
    max_distance: float = Field(default=1.2, gt=0, le=4)
    retrieval_mode: Literal["dense", "bm25", "hybrid"] = "hybrid"
    rerank: bool = True
    rewrite_query: bool = True
    min_rerank_score: float | None = None
    document_type: Literal["all", "law", "case"] = "all"
    validity: Literal["all", "current", "historical", "unknown"] = "all"


def embedding_model_from_env() -> str:
    return os.getenv("RAG_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL).strip()


def index_dir_from_env() -> Path:
    value = os.getenv("RAG_INDEX_DIR", "").strip()
    return Path(value).resolve() if value else DEFAULT_INDEX_DIR


def min_rerank_score_from_env() -> float | None:
    value = os.getenv("RAG_MIN_RERANK_SCORE", "").strip()
    return float(value) if value else None


def document_limit_from_env() -> int:
    value = os.getenv("RAG_DOCUMENT_LIMIT", "200").strip()
    try:
        limit = int(value)
    except ValueError as exc:
        raise RuntimeError("RAG_DOCUMENT_LIMIT 必须是大于等于 0 的整数。") from exc
    if limit < 0:
        raise RuntimeError("RAG_DOCUMENT_LIMIT 不能小于 0。")
    return limit


def get_engine() -> RAGEngine:
    global _engine
    if _engine is None:
        with _engine_lock:
            if _engine is None:
                _engine = RAGEngine(
                    index_dir=index_dir_from_env(),
                    embedding_model=embedding_model_from_env(),
                    max_tokens=int(os.getenv("RAG_MAX_TOKENS", "512")),
                    reranker_model=os.getenv("RAG_RERANKER_MODEL", DEFAULT_RERANKER_MODEL),
                    api_base_url=os.getenv("RAG_API_BASE_URL", ""),
                    api_key=os.getenv("RAG_API_KEY", ""),
                    api_model=os.getenv("RAG_API_MODEL", ""),
                    domain_profile=os.getenv("RAG_DOMAIN_PROFILE", "legal_assistant"),
                    intent_routing=os.getenv("RAG_INTENT_ROUTING", "rule"),
                )
    return _engine


def ndjson(event: str, **data) -> bytes:
    return (json.dumps({"event": event, **data}, ensure_ascii=False) + "\n").encode("utf-8")


@app.get("/")
def home() -> FileResponse:
    return FileResponse(WEB_DIR / "index.html")


@app.get("/api/health")
def health() -> dict:
    configured = all(
        os.getenv(name, "").strip() for name in ("RAG_API_BASE_URL", "RAG_API_KEY", "RAG_API_MODEL")
    )
    return {
        "status": "ok",
        "llm_provider": "api",
        "model_name": os.getenv("RAG_API_MODEL", ""),
        "model_configured": configured,
        "model_loaded": _engine is not None and _engine.model_loaded,
        "reranker_status": _engine.reranker_status if _engine is not None else "not_loaded",
        "domain_profile": os.getenv("RAG_DOMAIN_PROFILE", "legal_assistant"),
        "intent_routing": os.getenv("RAG_INTENT_ROUTING", "rule"),
        "knowledge_dir_configured": KNOWLEDGE_DIR.is_dir(),
        "document_limit": document_limit_from_env(),
        "index_ready": index_is_ready(index_dir_from_env()),
    }


@app.post("/api/documents/upload")
async def upload_document(file: UploadFile = File(...)) -> dict:
    try:
        safe_name = validate_document_name(file.filename or "")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    destination = UPLOAD_DIR / safe_name
    temporary = UPLOAD_DIR / f".upload-{uuid.uuid4().hex}.part"

    size = 0
    try:
        with temporary.open("xb") as target:
            while chunk := await file.read(1024 * 1024):
                size += len(chunk)
                if size > MAX_UPLOAD_BYTES:
                    raise HTTPException(status_code=413, detail="单个文档不能超过 20 MB。")
                target.write(chunk)
        with _index_build_lock:
            if destination.exists():
                raise HTTPException(status_code=409, detail="同名文档已经存在，请先修改文件名。")
            temporary.replace(destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    finally:
        await file.close()

    return {"filename": safe_name, "size": size}


@app.post("/api/index/build")
def rebuild_index() -> dict:
    with _index_build_lock:
        try:
            result = build_index(
                KNOWLEDGE_DIR,
                index_dir_from_env(),
                embedding_model_from_env(),
                include_runtime_uploads=True,
                document_limit=document_limit_from_env(),
            )
            if _engine is not None:
                _engine.reload_index()
            return result
        except (FileNotFoundError, ValueError) as exc:
            logger.warning("知识库建库失败：%s", exc)
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            logger.exception("知识库建库异常")
            raise HTTPException(status_code=500, detail="知识库建库失败，请查看服务终端日志。") from exc


@app.post("/api/chat/stream")
def chat_stream(request: ChatRequest) -> StreamingResponse:
    try:
        engine = get_engine()
        retrieval, tokens = engine.stream_answer(
            request.question,
            use_rag=request.use_rag,
            top_k=request.top_k,
            max_distance=request.max_distance,
            retrieval_mode=request.retrieval_mode,
            rerank=request.rerank,
            rewrite_query=request.rewrite_query,
            min_rerank_score=(
                request.min_rerank_score
                if request.min_rerank_score is not None
                else min_rerank_score_from_env()
            ),
            document_type=request.document_type,
            validity=request.validity,
        )
    except (FileNotFoundError, ValueError) as exc:
        logger.warning("问答请求无法处理：%s", exc)
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("问答请求异常")
        raise HTTPException(status_code=500, detail="问答失败，请查看服务终端日志。") from exc

    def generate():
        yield ndjson(
            "sources",
            query=retrieval.query,
            mode=retrieval.mode,
            rewrite_applied=retrieval.rewrite_applied,
            reranked=retrieval.reranked,
            reranker_backend=retrieval.reranker_backend,
            original_query=retrieval.original_query or retrieval.query,
            rewritten_query=retrieval.rewritten_query or retrieval.query,
            intent=retrieval.intent,
            intent_confidence=retrieval.intent_confidence,
            route_source=retrieval.route_source,
            generation_chain=retrieval.generation_chain,
            sources=[source.to_dict() for source in retrieval.sources],
        )
        try:
            for token in tokens:
                yield ndjson("token", text=token)
            yield ndjson("done")
        except Exception as exc:
            logger.exception("流式生成异常")
            yield ndjson("error", message="模型生成失败，请查看服务终端日志。")
            yield ndjson("done", failed=True)

    return StreamingResponse(generate(), media_type="application/x-ndjson")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
