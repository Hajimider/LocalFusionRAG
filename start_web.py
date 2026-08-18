"""个人项目启动入口：修改下方主要配置，然后在 IDE 中直接运行本文件。"""

import os
import webbrowser
from argparse import Namespace
from pathlib import Path
from threading import Timer

from dotenv import load_dotenv

from rag_core import DEFAULT_EMBEDDING_MODEL, DEFAULT_RERANKER_MODEL
from run_project import DEFAULT_INDEX, serve_command


# ==================== 主要配置：日常只修改这里 ====================

# 1. 回答模型固定使用 OpenAI 兼容 API；在项目根目录的 .env 中填写 BASE_URL、API_KEY、API_MODEL。
# .env 只需首次配置，真实密钥不会被提交到 GitHub。

# 2. 检索模型：可填写本地目录；留空时使用项目默认 BGE 模型名称。
EMBEDDING_MODEL = r""

# 3. 重排序模型：可填写本地目录；留空时使用默认名称或轻量回退。
# 示例：RERANKER_MODEL = r"path/to/bge-reranker-base"
RERANKER_MODEL = r""

# 4. 检索模型已下载时保持 True，禁止 BGE/Reranker 连接模型仓库。
# 该开关不影响回答 API 联网。
MODEL_REPOSITORY_OFFLINE = True

# 5. 回答长度和网页端口；回答越长，API 耗时或费用通常越高。
MAX_TOKENS = 512
PORT = 8000

# 6. 领域画像：本项目固定使用中国大陆中文法律资料。
DOMAIN_PROFILE = "legal_assistant"  # 法律辅助分析领域；不要改成其他领域除非复用旧编程语料。

# 7. 知识库目录：放入已授权法条/判例资料，可改为外部目录。
KNOWLEDGE_DIR = r"knowledge_base/legal_docs"  # 放入已授权的法条/判例 DOCX、PDF、Markdown 或 TXT。

# 8. Demo 默认均衡选取 200 份代表性资料建库；填写 0 时处理目录中的全部资料。
# 网页上传的文件始终会加入索引，不受这个上限影响。
DEMO_DOCUMENT_LIMIT = 200

# 9. 意图路由：rule 只用关键词，hybrid 对模糊问题再调用 API，llm 全部调用 API 分类。
# 默认使用 rule，避免 API 误把普通法条问题判成 current_law 后筛掉未知效力状态的资料。
INTENT_ROUTING = "rule"

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
    load_dotenv(Path(__file__).with_name(".env"))
    refresh_user_hf_cache()
    offline = "1" if MODEL_REPOSITORY_OFFLINE else "0"
    os.environ["HF_HUB_OFFLINE"] = offline
    os.environ["TRANSFORMERS_OFFLINE"] = offline
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    os.environ["RAG_DOMAIN_PROFILE"] = DOMAIN_PROFILE.strip() or "legal_assistant"
    os.environ["RAG_INTENT_ROUTING"] = INTENT_ROUTING.strip() or "rule"
    os.environ["RAG_KNOWLEDGE_DIR"] = str(Path(KNOWLEDGE_DIR).resolve())
    if DEMO_DOCUMENT_LIMIT < 0:
        raise ValueError("DEMO_DOCUMENT_LIMIT 不能小于 0。")
    os.environ["RAG_DOCUMENT_LIMIT"] = str(DEMO_DOCUMENT_LIMIT)
    provider = "api"
    api_base_url = (os.getenv("BASE_URL", "").strip() or os.getenv("API_BASE_URL", "").strip())
    api_key = os.getenv("API_KEY", "").strip()
    api_model = os.getenv("API_MODEL", "").strip()
    if provider == "api" and not all((api_base_url, api_key, api_model)):
        raise ValueError("API 模式需要填写 BASE_URL、API_KEY 和 API_MODEL。")
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

    os.environ["RAG_API_KEY"] = api_key if provider == "api" else ""
    Timer(1.0, lambda: webbrowser.open(f"http://127.0.0.1:{PORT}")).start()
    serve_command(
        Namespace(
            api_base_url=api_base_url,
            api_model=api_model,
            embedding_model=embedding_model,
            reranker_model=reranker_model,
            max_tokens=MAX_TOKENS,
            min_rerank_score=0.280595,
            host="127.0.0.1",
            port=PORT,
            index=DEFAULT_INDEX,
        )
    )


if __name__ == "__main__":
    main()
