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
KNOWLEDGE_DIR = BASE_DIR / "knowledge_base" / "project_docs"
UPLOAD_DIR = KNOWLEDGE_DIR / "uploads"
DEFAULT_INDEX_DIR = BASE_DIR / "storage" / "faiss"
WEB_DIR = BASE_DIR / "web"
MAX_UPLOAD_BYTES = 20 * 1024 * 1024

app = FastAPI(title="本地进阶 RAG 知识库问答系统", version="2.0.0")
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


def llm_provider_from_env() -> str:
    provider = os.getenv("RAG_LLM_PROVIDER", "local").strip().lower()
    if provider not in {"local", "api"}:
        raise RuntimeError("RAG_LLM_PROVIDER 只能设置为 local 或 api。")
    return provider


def model_path_from_env() -> str:
    model_path = os.getenv("RAG_MODEL_PATH", "").strip()
    if not model_path:
        raise RuntimeError("本地模式尚未设置 RAG_MODEL_PATH，无法加载 GGUF 模型。")
    return model_path


def embedding_model_from_env() -> str:
    return os.getenv("RAG_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL).strip()


def index_dir_from_env() -> Path:
    value = os.getenv("RAG_INDEX_DIR", "").strip()
    return Path(value).resolve() if value else DEFAULT_INDEX_DIR


def min_rerank_score_from_env() -> float | None:
    value = os.getenv("RAG_MIN_RERANK_SCORE", "").strip()
    return float(value) if value else None


def get_engine() -> RAGEngine:
    global _engine
    if _engine is None:
        with _engine_lock:
            if _engine is None:
                provider = llm_provider_from_env()
                _engine = RAGEngine(
                    model_path=model_path_from_env() if provider == "local" else None,
                    index_dir=index_dir_from_env(),
                    embedding_model=embedding_model_from_env(),
                    context_size=int(os.getenv("RAG_CONTEXT_SIZE", "4096")),
                    max_tokens=int(os.getenv("RAG_MAX_TOKENS", "512")),
                    reranker_model=os.getenv("RAG_RERANKER_MODEL", DEFAULT_RERANKER_MODEL),
                    llm_provider=provider,
                    api_base_url=os.getenv("RAG_API_BASE_URL", ""),
                    api_key=os.getenv("RAG_API_KEY", ""),
                    api_model=os.getenv("RAG_API_MODEL", ""),
                )
    return _engine


def ndjson(event: str, **data) -> bytes:
    return (json.dumps({"event": event, **data}, ensure_ascii=False) + "\n").encode("utf-8")


@app.get("/")
def home() -> FileResponse:
    return FileResponse(WEB_DIR / "index.html")


@app.get("/api/health")
def health() -> dict:
    provider = llm_provider_from_env()
    configured = (
        bool(os.getenv("RAG_MODEL_PATH", "").strip())
        if provider == "local"
        else all(os.getenv(name, "").strip() for name in ("RAG_API_BASE_URL", "RAG_API_KEY", "RAG_API_MODEL"))
    )
    return {
        "status": "ok",
        "llm_provider": provider,
        "model_name": os.getenv("RAG_API_MODEL", "") if provider == "api" else "本地 GGUF",
        "model_configured": configured,
        "model_loaded": _engine is not None and _engine.model_loaded,
        "reranker_status": _engine.reranker_status if _engine is not None else "not_loaded",
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
