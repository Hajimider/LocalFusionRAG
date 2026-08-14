"""个人项目启动入口：修改下方主要配置，然后在 IDE 中直接运行本文件。"""

import os
import webbrowser
from argparse import Namespace
from pathlib import Path
from threading import Timer

from rag_core import DEFAULT_EMBEDDING_MODEL, DEFAULT_RERANKER_MODEL
from run_project import DEFAULT_INDEX, serve_command


# ==================== 主要配置：日常只修改这里 ====================

# 1. 回答模型："local" 使用本地 GGUF；"api" 使用 OpenAI 兼容 API。
LLM_PROVIDER = "api"

# 2. 本地模式：填写 GGUF 文件或模型目录；API 模式不用填写。
# 示例：LOCAL_MODEL_PATH = r"path/to/Qwen2.5-7B-Instruct-GGUF"
LOCAL_MODEL_PATH = r""

# 3. API 模式：填写兼容接口地址、密钥和模型名称；本地模式不用填写。
# 切勿把真实 API Key 提交到 GitHub。
API_BASE_URL = ""
API_KEY = ""  # 仅在本地运行时填写，禁止提交真实密钥。
API_MODEL = ""

# 4. 检索模型：可填写本地目录；留空时使用项目默认 BGE 模型名称。
EMBEDDING_MODEL = r""

# 5. 重排序模型：可填写本地目录；留空时使用默认名称或轻量回退。
# 示例：RERANKER_MODEL = r"path/to/bge-reranker-base"
RERANKER_MODEL = r""

# 6. 模型已下载时保持 True，禁止 BGE/Reranker 连接模型仓库。
# 该开关不影响大模型 API 联网。
MODEL_REPOSITORY_OFFLINE = True

# 7. 回答长度和网页端口；回答越长，耗时或 API 费用通常越高。
MAX_TOKENS = 512
PORT = 8000

# 8. 领域画像：本项目固定使用中国大陆中文法律资料。
DOMAIN_PROFILE = "legal_assistant"  # 法律辅助分析领域；不要改成其他领域除非复用旧编程语料。

# 9. 知识库目录：放入已授权法条/判例资料，可改为外部目录。
KNOWLEDGE_DIR = r"knowledge_base/legal_docs"  # 放入已授权的法条/判例 DOCX、PDF、Markdown 或 TXT。

# 10. 意图路由：rule 只用关键词，hybrid 对模糊问题再调用模型，llm 全部调用模型分类。
INTENT_ROUTING = "hybrid"

# ==================== 配置结束：以下代码通常不用修改 ====================


def refresh_user_hf_cache() -> None:
    """让未重启的 Windows IDE 也能读取最新的用户级缓存配置。"""
    if os.name != "nt":
        return
    try:
        import winreg

        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as key:
            for name in ("HF_HOME", "HF_HUB_CACHE", "HF_DATASETS_CACHE"):
                try:
                    os.environ[name] = winreg.QueryValueEx(key, name)[0]
                except FileNotFoundError:
                    pass
    except OSError:
        pass


def resolve_embedding_model(model: str) -> str:
    """离线时把模型名称解析成实际快照目录，避免库继续查旧缓存。"""
    if Path(model).exists() or not MODEL_REPOSITORY_OFFLINE:
        return model
    try:
        from huggingface_hub import snapshot_download

        return snapshot_download(model, local_files_only=True)
    except Exception as exc:
        raise FileNotFoundError(
            f"离线缓存中没有检索模型 {model}。请检查 E 盘缓存，"
            "或在主要配置区填写 EMBEDDING_MODEL 的本地目录。"
        ) from exc


def main() -> None:
    refresh_user_hf_cache()
    offline = "1" if MODEL_REPOSITORY_OFFLINE else "0"
    os.environ["HF_HUB_OFFLINE"] = offline
    os.environ["TRANSFORMERS_OFFLINE"] = offline
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    os.environ["RAG_DOMAIN_PROFILE"] = DOMAIN_PROFILE.strip() or "legal_assistant"
    os.environ["RAG_INTENT_ROUTING"] = INTENT_ROUTING.strip() or "hybrid"
    os.environ["RAG_KNOWLEDGE_DIR"] = str(Path(KNOWLEDGE_DIR).resolve())
    provider = LLM_PROVIDER.strip().lower()
    if provider not in {"local", "api"}:
        raise ValueError('LLM_PROVIDER 只能填写 "local" 或 "api"。')
    local_model = Path(LOCAL_MODEL_PATH.strip()) if LOCAL_MODEL_PATH.strip() else None
    if provider == "local" and (local_model is None or not local_model.exists()):
        raise FileNotFoundError("本地模式需要填写正确的 LOCAL_MODEL_PATH。")
    if provider == "api" and not all(value.strip() for value in (API_BASE_URL, API_KEY, API_MODEL)):
        raise ValueError("API 模式需要填写 API_BASE_URL、API_KEY 和 API_MODEL。")
    embedding_model = EMBEDDING_MODEL.strip() or DEFAULT_EMBEDDING_MODEL
    reranker_model = RERANKER_MODEL.strip() or DEFAULT_RERANKER_MODEL
    for label, value in (("EMBEDDING_MODEL", EMBEDDING_MODEL), ("RERANKER_MODEL", RERANKER_MODEL)):
        if value.strip() and not Path(value).exists():
            raise FileNotFoundError(f"{label} 指向的本地目录不存在：{value}")
    embedding_model = resolve_embedding_model(embedding_model)
    if RERANKER_MODEL.strip():
        try:
            import google.protobuf  # noqa: F401
            import sentencepiece  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "完整 Reranker 需要 sentencepiece 和 protobuf。"
                "请先在当前 Python 环境安装缺失依赖，或将 RERANKER_MODEL 留空以使用轻量重排序。"
            ) from exc

    os.environ["RAG_API_KEY"] = API_KEY if provider == "api" else ""
    Timer(1.0, lambda: webbrowser.open(f"http://127.0.0.1:{PORT}")).start()
    serve_command(
        Namespace(
            model=local_model if provider == "local" else None,
            provider=provider,
            api_base_url=API_BASE_URL,
            api_model=API_MODEL,
            embedding_model=embedding_model,
            reranker_model=reranker_model,
            context_size=4096,
            max_tokens=MAX_TOKENS,
            min_rerank_score=0.280595,
            host="127.0.0.1",
            port=PORT,
            index=DEFAULT_INDEX,
        )
    )


if __name__ == "__main__":
    main()
