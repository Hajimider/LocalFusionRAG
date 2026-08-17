from __future__ import annotations

import hashlib
import html
import json
import logging
import math
import os
import re
import shutil
import tempfile
import threading
import time
import uuid
import zipfile
from xml.etree import ElementTree
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from collections import Counter
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Iterator, Sequence

from rag_orchestration import IntentDecision, IntentRouter, PromptOrchestrator


DEFAULT_EMBEDDING_MODEL = "BAAI/bge-small-zh-v1.5"
DEFAULT_RERANKER_MODEL = "BAAI/bge-reranker-base"
ALLOWED_DOCUMENT_TYPES = {".pdf", ".docx", ".md", ".txt"}
LEGAL_DOCUMENT_CATEGORIES = {"宪法", "法律", "行政法规", "监察法规", "司法解释", "地方法规"}
DEMO_PRIORITY_TITLES = (
    "中华人民共和国宪法（2018年修正文本）",
    "中华人民共和国民法典",
    "中华人民共和国劳动合同法",
    "中华人民共和国消费者权益保护法",
    "中华人民共和国公司法",
    "中华人民共和国刑法",
    "中华人民共和国行政处罚法",
    "中华人民共和国行政诉讼法",
    "中华人民共和国民事诉讼法",
    "中华人民共和国劳动法",
    "中华人民共和国监察法实施条例",
    "保障农民工工资支付条例",
    "工伤保险条例",
)
INDEX_MANIFEST = "manifest.json"
INDEX_POINTER = "CURRENT"
INDEX_LOCK = threading.RLock()
RETRIEVAL_MODES = {"dense", "bm25", "hybrid"}
LLM_PROVIDERS = {"local", "api"}
FILESYSTEM_LOCK_TIMEOUT = 300.0
FILESYSTEM_LOCK_POLL_INTERVAL = 0.1
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Source:
    file: str
    page: int | None
    chunk: int
    distance: float | None
    excerpt: str
    retrieval_score: float = 0.0
    rerank_score: float | None = None
    methods: tuple[str, ...] = ()
    doc_type: str = "unknown"
    validity: str = "unknown"
    jurisdiction: str = ""
    effective_date: str = ""
    expiry_date: str = ""
    court: str = ""
    case_number: str = ""
    judgment_date: str = ""
    title: str = ""
    source_url: str = ""
    legal_category: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class RetrievalResult:
    query: str
    mode: str
    reranked: bool
    rewrite_applied: bool
    sources: list[Source]
    reranker_backend: str = "none"
    rewritten_query: str = ""
    intent: str = "qa"
    intent_confidence: float = 0.0
    route_source: str = "rule"
    generation_chain: str = "qa"
    original_query: str = ""


def tokenize_for_bm25(text: str) -> list[str]:
    """保留英文/代码词，并用中文双字词提供无需分词库的稳定基线。"""
    tokens: list[str] = []
    for segment in re.findall(r"[\u4e00-\u9fff]+|[A-Za-z0-9_+#.\-/]+", text.lower()):
        if re.fullmatch(r"[\u4e00-\u9fff]+", segment):
            tokens.extend(segment)
            tokens.extend(segment[index : index + 2] for index in range(max(1, len(segment) - 1)))
        else:
            tokens.append(segment)
    return tokens


class BM25Retriever:
    def __init__(self, documents: Sequence, k1: float = 1.5, b: float = 0.75) -> None:
        self.documents = list(documents)
        self.k1 = k1
        self.b = b
        self.corpus = [tokenize_for_bm25(document.page_content) for document in self.documents]
        self.term_frequencies = [Counter(tokens) for tokens in self.corpus]
        self.average_length = sum(map(len, self.corpus)) / max(1, len(self.corpus))
        document_frequency = Counter()
        for tokens in self.corpus:
            document_frequency.update(set(tokens))
        total = len(self.corpus)
        self.idf = {
            term: math.log(1 + (total - frequency + 0.5) / (frequency + 0.5))
            for term, frequency in document_frequency.items()
        }

    def search(self, query: str, k: int, min_coverage: float = 0.25) -> list[tuple[object, float]]:
        query_tokens = tokenize_for_bm25(query)
        unique_query = set(query_tokens)
        if not unique_query:
            return []

        ranked: list[tuple[object, float]] = []
        for document, tokens, frequencies in zip(self.documents, self.corpus, self.term_frequencies):
            coverage = len(unique_query.intersection(frequencies)) / len(unique_query)
            if coverage < min_coverage:
                continue
            length_ratio = len(tokens) / max(1.0, self.average_length)
            score = 0.0
            for term in query_tokens:
                frequency = frequencies.get(term, 0)
                if not frequency:
                    continue
                denominator = frequency + self.k1 * (1 - self.b + self.b * length_ratio)
                score += self.idf.get(term, 0.0) * frequency * (self.k1 + 1) / denominator
            if score > 0:
                ranked.append((document, score))
        ranked.sort(key=lambda item: item[1], reverse=True)
        return ranked[:k]


class EmbeddingReranker:
    """CrossEncoder 不可用时复用 BGE 向量做轻量二次排序。"""

    def __init__(self, embedding_function) -> None:
        self.embedding_function = embedding_function

    def predict(self, pairs: list[tuple[str, str]], **_) -> list[float]:
        if not pairs:
            return []
        query = pairs[0][0]
        query_vector = self.embedding_function.embed_query(query)
        document_vectors = self.embedding_function.embed_documents([document for _, document in pairs])
        return [
            float(sum(left * right for left, right in zip(query_vector, document_vector)))
            for document_vector in document_vectors
        ]


def reciprocal_rank_fusion(
    rankings: dict[str, list[tuple]],
    weights: dict[str, float] | None = None,
    rank_constant: int = 60,
) -> dict[tuple, float]:
    weights = weights or {}
    scores: dict[tuple, float] = {}
    for method, keys in rankings.items():
        weight = weights.get(method, 1.0)
        for rank, key in enumerate(keys, start=1):
            scores[key] = scores.get(key, 0.0) + weight / (rank_constant + rank)
    return scores


def resolve_model_path(model_path: str | Path) -> Path:
    path = Path(model_path).expanduser()
    if path.is_file() and path.suffix.lower() == ".gguf":
        return path.resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"找不到 GGUF 模型：{path}")

    split_files = sorted(path.glob("*00001-of-*.gguf"))
    gguf_files = split_files or sorted(path.glob("*.gguf"))
    if not gguf_files:
        raise FileNotFoundError(f"目录中没有 GGUF 文件：{path}")
    return gguf_files[0].resolve()


def validate_document_name(filename: str) -> str:
    safe_name = Path(filename or "").name
    if not safe_name or Path(safe_name).suffix.lower() not in ALLOWED_DOCUMENT_TYPES:
        allowed = "、".join(sorted(ALLOWED_DOCUMENT_TYPES))
        raise ValueError(f"只支持以下文档格式：{allowed}")
    return safe_name


def _read_text(path: Path) -> str:
    for encoding in ("utf-8-sig", "gb18030"):
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
    raise ValueError(f"无法识别文本编码：{path.name}")


def _read_docx(path: Path) -> str:
    """使用 OOXML 主文档提取段落和表格文本，不依赖 Office。"""
    try:
        with zipfile.ZipFile(path) as archive:
            document_xml = archive.read("word/document.xml")
        root = ElementTree.fromstring(document_xml)
    except (zipfile.BadZipFile, KeyError, ElementTree.ParseError) as exc:
        raise ValueError(f"DOCX 文件损坏或格式无效：{path.name}") from exc

    word_namespace = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
    paragraphs: list[str] = []
    for paragraph in root.iter(f"{word_namespace}p"):
        parts: list[str] = []
        for node in paragraph.iter():
            if node.tag == f"{word_namespace}t":
                parts.append(node.text or "")
            elif node.tag == f"{word_namespace}tab":
                parts.append("\t")
            elif node.tag in {f"{word_namespace}br", f"{word_namespace}cr"}:
                parts.append("\n")
        text = "".join(parts).strip()
        if text:
            paragraphs.append(text)
    return "\n\n".join(paragraphs)


def _legal_metadata_from_path(path: Path, root: Path) -> dict[str, str]:
    relative = path.relative_to(root)
    category = next((part for part in relative.parts[:-1] if part in LEGAL_DOCUMENT_CATEGORIES), "")
    if not category:
        return {}
    title = re.sub(r"_\d{8}$", "", path.stem)
    return {
        "doc_type": "law",
        "validity": "unknown",
        "title": title,
        "legal_category": category,
        "metadata_review": "required",
    }


def _parse_front_matter(text: str) -> tuple[dict[str, str], str]:
    """读取 Markdown/TXT 顶部的简单 key: value 元数据。"""
    lines = text.splitlines()
    if len(lines) < 3 or lines[0].strip() != "---":
        return {}, text
    try:
        end = next(index for index, line in enumerate(lines[1:], start=1) if line.strip() == "---")
    except StopIteration:
        return {}, text
    metadata: dict[str, str] = {}
    for line in lines[1:end]:
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip().strip("\"'")
        if key:
            metadata[key] = value
    return metadata, "\n".join(lines[end + 1:]).lstrip()


def _demo_document_sort_key(path: Path) -> tuple:
    stem = path.stem
    for priority, title in enumerate(DEMO_PRIORITY_TITLES):
        if stem == title or stem.startswith(f"{title}_"):
            date_match = re.search(r"_(\d{8})$", stem)
            date = int(date_match.group(1)) if date_match else 0
            return 0, priority, -date, path.name
    return 1, path.name


def _balanced_document_sample(paths: Sequence[Path], root: Path, limit: int | None) -> list[Path]:
    if limit is None or limit == 0 or len(paths) <= limit:
        return sorted(paths)
    if limit < 0:
        raise ValueError("文档上限不能小于 0。")

    groups: dict[str, list[Path]] = {}
    for path in paths:
        relative = path.relative_to(root)
        category = relative.parts[0] if len(relative.parts) > 1 else "其他"
        groups.setdefault(category, []).append(path)
    for group in groups.values():
        group.sort(key=_demo_document_sort_key)

    selected: list[Path] = []
    categories = sorted(groups)
    while len(selected) < limit and categories:
        remaining = []
        for category in categories:
            group = groups[category]
            if group:
                selected.append(group.pop(0))
                if len(selected) == limit:
                    break
            if group:
                remaining.append(category)
        categories = remaining
    return selected


def load_documents(
    source_dir: str | Path,
    include_runtime_uploads: bool = False,
    document_limit: int | None = None,
):
    root = Path(source_dir).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"知识库目录不存在：{root}")

    documents = []
    ignored_directories = {".cache", "__pycache__"}
    source_paths: list[Path] = []
    runtime_uploads: list[Path] = []
    legacy_doc_count = 0
    for path in sorted(root.rglob("*")):
        relative_parts = path.relative_to(root).parts[:-1]
        if any(part in ignored_directories for part in relative_parts):
            continue
        if not path.is_file():
            continue
        is_runtime_upload = "uploads" in relative_parts
        if is_runtime_upload and not include_runtime_uploads:
            continue
        suffix = path.suffix.lower()
        if suffix == ".doc":
            legacy_doc_count += 1
            continue
        if suffix not in ALLOWED_DOCUMENT_TYPES:
            continue
        if path.name.lower() in {"readme.md", "readme.txt"} or path.name.startswith("_"):
            continue
        (runtime_uploads if is_runtime_upload else source_paths).append(path)

    selected_paths = _balanced_document_sample(source_paths, root, document_limit)
    for path in [*selected_paths, *sorted(runtime_uploads)]:
        suffix = path.suffix.lower()
        source = path.relative_to(root).as_posix()
        if suffix == ".pdf":
            from langchain_community.document_loaders import PyPDFLoader

            loaded = PyPDFLoader(str(path)).load()
            sidecar_metadata: dict[str, str] = {}
            for sidecar in (path.with_suffix(path.suffix + ".json"), path.with_suffix(".json")):
                if sidecar.is_file():
                    try:
                        value = json.loads(_read_text(sidecar))
                        if isinstance(value, dict):
                            sidecar_metadata = {str(key): str(item) for key, item in value.items()}
                    except (json.JSONDecodeError, ValueError):
                        logger.warning("PDF 元数据旁车文件无法解析：%s", sidecar)
                    break
            for page_number, document in enumerate(loaded, start=1):
                text = document.page_content.strip()
                if text:
                    document.metadata.update({"source": source, "page": page_number, **sidecar_metadata})
                    documents.append(document)
        else:
            if suffix == ".docx":
                body = _read_docx(path)
                metadata = _legal_metadata_from_path(path, root)
            else:
                raw_text = _read_text(path)
                metadata, body = _parse_front_matter(raw_text)
            if body.strip():
                from langchain_core.documents import Document

                documents.append(Document(page_content=body, metadata={"source": source, **metadata}))

    if legacy_doc_count:
        logger.warning("跳过 %d 个旧版 DOC 文件；请先转换为 DOCX、PDF 或 TXT。", legacy_doc_count)
    if not documents:
        raise ValueError("知识库中没有可读取的 PDF、DOCX、Markdown 或 TXT 文档。")
    return documents


def split_documents(documents, chunk_size: int = 500, chunk_overlap: int = 80):
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", "。", "！", "？", ";", "；", "，", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    for index, chunk in enumerate(chunks):
        chunk.metadata["chunk"] = index
    return chunks


def create_embeddings(model_name_or_path: str = DEFAULT_EMBEDDING_MODEL):
    # Conda 的 CA 证书可能不包含 Windows 已信任的代理/校园网证书。
    # 优先使用独立 truststore；未安装时复用 pip 自带的同一组件。
    try:
        import truststore
    except ImportError:
        try:
            from pip._vendor import truststore
        except ImportError:
            truststore = None
    if truststore is not None:
        truststore.inject_into_ssl()

    from langchain_huggingface import HuggingFaceEmbeddings

    offline = os.getenv("HF_HUB_OFFLINE", "0").strip().lower() in {"1", "true", "yes"}
    return HuggingFaceEmbeddings(
        model_name=model_name_or_path,
        model_kwargs={"device": "cpu", "local_files_only": offline},
        encode_kwargs={"normalize_embeddings": True},
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@contextmanager
def filesystem_lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as file:
        file.seek(0, os.SEEK_END)
        if file.tell() == 0:
            file.write(b"0")
            file.flush()
        file.seek(0)
        if os.name == "nt":
            import msvcrt

            deadline = time.monotonic() + FILESYSTEM_LOCK_TIMEOUT
            while True:
                try:
                    msvcrt.locking(file.fileno(), msvcrt.LK_NBLCK, 1)
                    break
                except OSError as exc:
                    if time.monotonic() >= deadline:
                        raise TimeoutError(f"获取索引文件锁超时：{path}") from exc
                    time.sleep(FILESYSTEM_LOCK_POLL_INTERVAL)
        else:
            import fcntl

            fcntl.flock(file.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            file.seek(0)
            if os.name == "nt":
                msvcrt.locking(file.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(file.fileno(), fcntl.LOCK_UN)


def active_index_dir(index_dir: str | Path) -> Path:
    root = Path(index_dir).resolve()
    pointer = root / INDEX_POINTER
    if not pointer.is_file():
        return root
    version = pointer.read_text(encoding="utf-8").strip()
    if not re.fullmatch(r"[0-9a-f]{32}", version):
        raise ValueError("知识库索引版本指针损坏，请重新执行 build 命令。")
    active = root / "versions" / version
    if not active.is_dir():
        raise ValueError("知识库索引版本不存在，请重新执行 build 命令。")
    return active


def index_is_ready(index_dir: str | Path) -> bool:
    try:
        path = active_index_dir(index_dir)
    except (OSError, ValueError):
        return False
    return all((path / filename).is_file() for filename in ("index.faiss", "index.pkl", INDEX_MANIFEST))


def build_index(
    source_dir: str | Path,
    index_dir: str | Path,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    chunk_size: int = 500,
    chunk_overlap: int = 80,
    include_runtime_uploads: bool = False,
    document_limit: int | None = None,
) -> dict:
    from langchain_community.vectorstores import FAISS

    documents = load_documents(
        source_dir,
        include_runtime_uploads=include_runtime_uploads,
        document_limit=document_limit,
    )
    chunks = split_documents(documents, chunk_size, chunk_overlap)
    store = FAISS.from_documents(chunks, create_embeddings(embedding_model))
    destination = Path(index_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    # Windows 版 FAISS 的 C++ 文件接口无法稳定处理中文路径，先写入系统临时目录。
    temporary = Path(tempfile.mkdtemp(prefix="local-rag-faiss-"))
    try:
        store.save_local(str(temporary))
        with INDEX_LOCK, filesystem_lock(destination / ".index.lock"):
            version = uuid.uuid4().hex
            versions_dir = destination / "versions"
            version_dir = versions_dir / version
            version_dir.mkdir(parents=True, exist_ok=False)
            for filename in ("index.faiss", "index.pkl"):
                shutil.copy2(temporary / filename, version_dir / filename)
            (version_dir / INDEX_MANIFEST).write_text(
                json.dumps(
                    {
                        "format": 2,
                        "created_by": "local_rag_qa",
                        "embedding_model": embedding_model,
                        "checksums": {
                            filename: _sha256(version_dir / filename)
                            for filename in ("index.faiss", "index.pkl")
                        },
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            pointer = destination / f".{INDEX_POINTER}.{version}.new"
            pointer.write_text(version, encoding="utf-8")
            pointer.replace(destination / INDEX_POINTER)
            versions = sorted(
                (path for path in versions_dir.iterdir() if path.is_dir()),
                key=lambda path: path.stat().st_mtime_ns,
                reverse=True,
            )
            for old_version in versions[2:]:
                shutil.rmtree(old_version, ignore_errors=True)
    finally:
        shutil.rmtree(temporary, ignore_errors=True)
    return {
        "source_files": len({document.metadata.get("source", "") for document in documents}),
        "document_sections": len(documents),
        "chunks": len(chunks),
        "index_dir": str(destination),
    }


def load_index(index_dir: str | Path, embedding_model: str = DEFAULT_EMBEDDING_MODEL):
    from langchain_community.vectorstores import FAISS

    root = Path(index_dir).resolve()
    if not root.is_dir():
        raise FileNotFoundError("尚未建立完整知识库索引，请先执行 build 命令。")
    with INDEX_LOCK, filesystem_lock(root / ".index.lock"):
        path = active_index_dir(root)
        if not all((path / filename).is_file() for filename in ("index.faiss", "index.pkl", INDEX_MANIFEST)):
            raise FileNotFoundError("尚未建立完整知识库索引，请先执行 build 命令。")
        try:
            manifest = json.loads((path / INDEX_MANIFEST).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError("知识库索引清单损坏，请重新执行 build 命令。") from exc
        if manifest.get("created_by") != "local_rag_qa" or manifest.get("format") not in {1, 2}:
            raise ValueError("索引不是本项目生成的受信任索引，请重新执行 build 命令。")
        if manifest.get("embedding_model") != embedding_model:
            raise ValueError("索引使用了不同的向量模型，请使用相同模型或重新执行 build 命令。")
        if manifest.get("format") == 2:
            checksums = manifest.get("checksums", {})
            for filename in ("index.faiss", "index.pkl"):
                if checksums.get(filename) != _sha256(path / filename):
                    raise ValueError("知识库索引校验失败，请重新执行 build 命令。")
        # Windows 版 FAISS 读取中文路径也可能失败，复制到纯英文临时目录后加载。
        temporary = Path(tempfile.mkdtemp(prefix="local-rag-faiss-load-"))
        try:
            shutil.copy2(path / "index.faiss", temporary / "index.faiss")
            shutil.copy2(path / "index.pkl", temporary / "index.pkl")
            return FAISS.load_local(
                str(temporary),
                create_embeddings(embedding_model),
                allow_dangerous_deserialization=True,
            )
        finally:
            shutil.rmtree(temporary, ignore_errors=True)


def format_context(sources: Iterable[Source]) -> str:
    blocks = []
    for number, source in enumerate(sources, start=1):
        location = source.file
        if source.doc_type != "unknown" or source.validity != "unknown":
            location += f" | 类型:{source.doc_type} | 效力:{source.validity} | 地域:{source.jurisdiction or '未标注'}"
            if source.effective_date or source.expiry_date:
                location += f" | 有效期:{source.effective_date or '未知'}-{source.expiry_date or '未标注'}"
            if source.court or source.case_number or source.judgment_date:
                location += f" | 法院:{source.court} | 案号:{source.case_number} | 日期:{source.judgment_date}"
        if source.page is not None:
            location += f"，第 {source.page} 页"
        blocks.append(
            f"[资料 {number}｜{html.escape(location, quote=False)}]\n"
            f"{html.escape(source.excerpt, quote=False)}"
        )
    return "\n\n".join(blocks)


class LocalLLM:
    def __init__(
        self,
        model_path: str | Path | None,
        context_size: int = 4096,
        max_tokens: int = 512,
        threads: int | None = None,
    ) -> None:
        try:
            from llama_cpp import Llama
        except ImportError as exc:
            raise RuntimeError(
                "缺少 llama-cpp-python，无法加载 GGUF 模型。请按 README 的 CPU wheel 命令单独安装。"
            ) from exc

        resolved = resolve_model_path(model_path)
        cpu_count = os.cpu_count() or 4
        self.max_tokens = max_tokens
        self._lock = threading.Lock()
        self._model = Llama(
            model_path=str(resolved),
            n_ctx=context_size,
            n_batch=128,
            n_threads=threads or max(1, min(8, cpu_count - 2)),
            verbose=False,
        )

    def stream(self, messages: list[dict[str, str]], max_tokens: int | None = None) -> Iterator[str]:
        with self._lock:
            result = self._model.create_chat_completion(
                messages=messages,
                temperature=0.1,
                top_p=0.9,
                max_tokens=max_tokens or self.max_tokens,
                stream=True,
            )
            for event in result:
                token = event.get("choices", [{}])[0].get("delta", {}).get("content", "")
                if token:
                    yield token

    def complete(self, messages: list[dict[str, str]], max_tokens: int | None = None) -> str:
        return "".join(self.stream(messages, max_tokens=max_tokens)).strip()


class OpenAICompatibleLLM:
    """使用标准库调用 OpenAI 兼容的 Chat Completions 流式接口。"""

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        max_tokens: int = 512,
        timeout: float = 120.0,
    ) -> None:
        if not base_url.strip() or not api_key.strip() or not model.strip():
            raise ValueError("API 模式需要配置 base_url、api_key 和 model。")
        self.endpoint = base_url.rstrip("/") + "/chat/completions"
        self.api_key = api_key
        self.model = model
        self.max_tokens = max_tokens
        self.timeout = timeout

    def stream(self, messages: list[dict[str, str]], max_tokens: int | None = None) -> Iterator[str]:
        payload = json.dumps(
            {
                "model": self.model,
                "messages": messages,
                "temperature": 0.1,
                "top_p": 0.9,
                "max_tokens": max_tokens or self.max_tokens,
                "stream": True,
            }
        ).encode("utf-8")
        request = Request(
            self.endpoint,
            data=payload,
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.timeout) as response:
                for raw_line in response:
                    line = raw_line.decode("utf-8").strip()
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    event = json.loads(data)
                    choices = event.get("choices") or []
                    if not choices:
                        continue
                    token = choices[0].get("delta", {}).get("content", "")
                    if token:
                        yield token
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:500]
            raise RuntimeError(f"大模型 API 请求失败（HTTP {exc.code}）：{detail}") from exc
        except (URLError, TimeoutError) as exc:
            raise RuntimeError(f"无法连接大模型 API：{exc}") from exc
        except json.JSONDecodeError as exc:
            raise RuntimeError("大模型 API 返回了无法解析的流式数据。") from exc

    def complete(self, messages: list[dict[str, str]], max_tokens: int | None = None) -> str:
        return "".join(self.stream(messages, max_tokens=max_tokens)).strip()


class RAGEngine:
    def __init__(
        self,
        model_path: str | Path | None,
        index_dir: str | Path = "storage/faiss",
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
        context_size: int = 4096,
        max_tokens: int = 512,
        threads: int | None = None,
        reranker_model: str = DEFAULT_RERANKER_MODEL,
        llm_provider: str = "local",
        api_base_url: str = "",
        api_key: str = "",
        api_model: str = "",
        domain_profile: str = "legal_assistant",
        intent_routing: str = "rule",
    ) -> None:
        if llm_provider not in LLM_PROVIDERS:
            raise ValueError(f"不支持的模型提供方：{llm_provider}")
        self.index_dir = Path(index_dir)
        self.embedding_model = embedding_model
        self.model_path = model_path
        self.context_size = context_size
        self.max_tokens = max_tokens
        self.threads = threads
        self.reranker_model = reranker_model
        self.llm_provider = llm_provider
        self.api_base_url = api_base_url
        self.api_key = api_key
        self.api_model = api_model
        self.domain_profile = domain_profile
        self.intent_router = IntentRouter(intent_routing)
        self.prompt_orchestrator = PromptOrchestrator(domain_profile)
        self._llm: LocalLLM | OpenAICompatibleLLM | None = None
        self._llm_lock = threading.Lock()
        self._store = None
        self._store_lock = threading.RLock()
        self._bm25: BM25Retriever | None = None
        self._reranker = None
        self._reranker_kind = "not_loaded"
        self._reranker_lock = threading.RLock()
        self._reranker_error: str | None = None

    @property
    def llm(self) -> LocalLLM | OpenAICompatibleLLM:
        if self._llm is None:
            with self._llm_lock:
                if self._llm is None:
                    if self.llm_provider == "api":
                        self._llm = OpenAICompatibleLLM(
                            self.api_base_url, self.api_key, self.api_model, self.max_tokens
                        )
                    else:
                        if not self.model_path:
                            raise ValueError("本地模式需要提供 GGUF 模型路径。")
                        self._llm = LocalLLM(
                            self.model_path, self.context_size, self.max_tokens, self.threads
                        )
        return self._llm

    @property
    def model_loaded(self) -> bool:
        return self._llm is not None

    def reload_index(self) -> None:
        store = load_index(self.index_dir, self.embedding_model)
        bm25 = BM25Retriever(self._documents_from_store(store))
        with self._store_lock:
            self._store = store
            self._bm25 = bm25

    def _get_store(self):
        if self._store is None:
            with self._store_lock:
                if self._store is None:
                    self.reload_index()
        return self._store

    @staticmethod
    def _document_key(document) -> tuple:
        metadata = document.metadata
        return (
            str(metadata.get("source", "未知来源")),
            metadata.get("page"),
            int(metadata.get("chunk", 0)),
        )

    @staticmethod
    def _documents_from_store(store) -> list:
        documents = []
        for index in range(store.index.ntotal):
            document_id = store.index_to_docstore_id[index]
            document = store.docstore.search(document_id)
            if not isinstance(document, str):
                documents.append(document)
        return documents

    def _get_retrieval_snapshot(self) -> tuple[object, BM25Retriever]:
        with self._store_lock:
            if self._store is None:
                self.reload_index()
            if self._bm25 is None:
                self._bm25 = BM25Retriever(self._documents_from_store(self._store))
            return self._store, self._bm25

    def _activate_embedding_fallback(self, reason: Exception):
        try:
            fallback = EmbeddingReranker(self._get_store().embedding_function)
        except Exception as exc:
            self._reranker = None
            self._reranker_kind = "rrf_fallback"
            logger.warning("Reranker 轻量兜底失败，使用 RRF 排序：%s；原错误：%s", exc, reason)
            return None
        self._reranker = fallback
        self._reranker_kind = "embedding_fallback"
        logger.warning("CrossEncoder 不可用，使用 BGE 轻量重排序：%s", reason)
        return fallback

    def _get_reranker(self):
        if self._reranker is None and self._reranker_error is None:
            with self._reranker_lock:
                if self._reranker is None and self._reranker_error is None:
                    try:
                        cache_root = Path(
                            os.getenv("RAG_HF_HOME", str(self.index_dir.resolve().parent / "model_cache"))
                        )
                        cache_root.mkdir(parents=True, exist_ok=True)
                        os.environ["HF_HOME"] = str(cache_root)
                        os.environ["HF_HUB_CACHE"] = str(cache_root / "hub")
                        os.environ["TRANSFORMERS_CACHE"] = str(cache_root / "transformers")
                        os.environ["SENTENCE_TRANSFORMERS_HOME"] = str(cache_root / "sentence_transformers")
                        try:
                            import huggingface_hub.constants as hf_constants

                            hf_constants.HF_HUB_CACHE = str(cache_root / "hub")
                        except ImportError:
                            pass
                        from sentence_transformers import CrossEncoder

                        offline = os.getenv("HF_HUB_OFFLINE", "0").strip().lower() in {"1", "true", "yes"}
                        self._reranker = CrossEncoder(
                            self.reranker_model,
                            device="cpu",
                            cache_folder=str(cache_root / "sentence_transformers"),
                            local_files_only=offline,
                        )
                        self._reranker_kind = "cross_encoder"
                    except Exception as exc:
                        self._reranker_error = str(exc)
                        self._activate_embedding_fallback(exc)
        return self._reranker

    @property
    def reranker_status(self) -> str:
        return self._reranker_kind

    @staticmethod
    def _make_source(
        document,
        distance: float | None,
        retrieval_score: float,
        methods: tuple[str, ...],
        rerank_score: float | None = None,
    ) -> Source:
        metadata = document.metadata
        return Source(
            file=str(metadata.get("source", "未知来源")),
            page=metadata.get("page"),
            chunk=int(metadata.get("chunk", 0)),
            distance=None if distance is None else round(distance, 4),
            excerpt=document.page_content.strip(),
            retrieval_score=round(float(retrieval_score), 6),
            rerank_score=None if rerank_score is None else round(float(rerank_score), 6),
            methods=methods,
            doc_type=str(metadata.get("doc_type", "unknown")),
            validity=str(metadata.get("validity", "unknown")),
            jurisdiction=str(metadata.get("jurisdiction", "")),
            effective_date=str(metadata.get("effective_date", "")),
            expiry_date=str(metadata.get("expiry_date", "")),
            court=str(metadata.get("court", "")),
            case_number=str(metadata.get("case_number", "")),
            judgment_date=str(metadata.get("judgment_date", "")),
            title=str(metadata.get("title", "")),
            source_url=str(metadata.get("source_url", "")),
            legal_category=str(metadata.get("legal_category", "")),
        )

    @staticmethod
    def _matches_legal_filters(document, document_type: str, validity: str) -> bool:
        if document_type not in {"all", "law", "case"}:
            raise ValueError(f"未知文档类型：{document_type}")
        if validity not in {"all", "current", "historical", "unknown"}:
            raise ValueError(f"未知效力状态：{validity}")
        metadata = document.metadata
        return (
            (document_type == "all" or str(metadata.get("doc_type", "unknown")) == document_type)
            and (validity == "all" or str(metadata.get("validity", "unknown")) == validity)
        )

    def retrieve(self, question: str, top_k: int = 4, max_distance: float = 1.2) -> list[Source]:
        results = self._get_store().similarity_search_with_score(question, k=top_k)
        sources = []
        for document, distance in results:
            numeric_distance = float(distance)
            if numeric_distance > max_distance:
                continue
            sources.append(self._make_source(document, numeric_distance, 1 / (1 + numeric_distance), ("dense",)))
        return sources

    def _rerank_candidates(
        self,
        question: str,
        ordered_keys: list[tuple],
        documents: dict[tuple, object],
    ) -> tuple[list[tuple], dict[tuple, float], bool]:
        pairs = [(question, documents[key].page_content) for key in ordered_keys]
        with self._reranker_lock:
            model = self._get_reranker()
            if model is None:
                return ordered_keys, {}, False
            try:
                scores = model.predict(pairs, batch_size=8, show_progress_bar=False)
                rerank_scores = {key: float(score) for key, score in zip(ordered_keys, scores)}
                return sorted(ordered_keys, key=rerank_scores.get, reverse=True), rerank_scores, True
            except Exception as exc:
                self._reranker_error = str(exc)
                fallback = (
                    self._activate_embedding_fallback(exc)
                    if self._reranker_kind == "cross_encoder"
                    else None
                )
                if fallback is not None:
                    try:
                        scores = fallback.predict(pairs, batch_size=8, show_progress_bar=False)
                        rerank_scores = {key: float(score) for key, score in zip(ordered_keys, scores)}
                        return sorted(ordered_keys, key=rerank_scores.get, reverse=True), rerank_scores, True
                    except Exception as fallback_exc:
                        logger.warning("BGE 轻量重排序失败，使用 RRF 排序：%s", fallback_exc)
                self._reranker = None
                self._reranker_kind = "rrf_fallback"
                return ordered_keys, {}, False

    def retrieve_advanced(
        self,
        question: str,
        top_k: int = 4,
        max_distance: float = 1.2,
        mode: str = "hybrid",
        rerank: bool = True,
        min_rerank_score: float | None = None,
        document_type: str = "all",
        validity: str = "all",
    ) -> RetrievalResult:
        if mode not in RETRIEVAL_MODES:
            raise ValueError(f"未知检索方式：{mode}")
        store, bm25 = self._get_retrieval_snapshot()
        candidate_k = max(8, top_k * 3)
        if document_type != "all" or validity != "all":
            candidate_k = max(64, top_k * 10)
        documents: dict[tuple, object] = {}
        distances: dict[tuple, float] = {}
        rankings: dict[str, list[tuple]] = {}

        if mode in {"dense", "hybrid"}:
            dense_keys = []
            dense_k = candidate_k
            if document_type != "all" or validity != "all":
                dense_k = max(candidate_k, int(getattr(getattr(store, "index", None), "ntotal", candidate_k)))
            for document, distance in store.similarity_search_with_score(question, k=dense_k):
                if not self._matches_legal_filters(document, document_type, validity):
                    continue
                numeric_distance = float(distance)
                if numeric_distance > max_distance:
                    continue
                key = self._document_key(document)
                documents[key] = document
                distances[key] = numeric_distance
                dense_keys.append(key)
            rankings["dense"] = dense_keys

        bm25_scores: dict[tuple, float] = {}
        if mode in {"bm25", "hybrid"}:
            bm25_keys = []
            bm25_k = len(bm25.documents) if document_type != "all" or validity != "all" else candidate_k
            for document, score in bm25.search(question, bm25_k):
                if not self._matches_legal_filters(document, document_type, validity):
                    continue
                key = self._document_key(document)
                documents[key] = document
                bm25_scores[key] = float(score)
                bm25_keys.append(key)
            rankings["bm25"] = bm25_keys

        if mode == "dense":
            fused_scores = {key: 1 / (1 + distances[key]) for key in rankings["dense"]}
        elif mode == "bm25":
            fused_scores = bm25_scores
        else:
            fused_scores = reciprocal_rank_fusion(
                rankings,
                weights={"dense": 0.65, "bm25": 0.35},
            )

        ordered_keys = sorted(fused_scores, key=fused_scores.get, reverse=True)[:candidate_k]
        rerank_scores: dict[tuple, float] = {}
        reranked = False
        if rerank and ordered_keys:
            ordered_keys, rerank_scores, reranked = self._rerank_candidates(question, ordered_keys, documents)
        if reranked and self.reranker_status == "cross_encoder" and min_rerank_score is not None:
            ordered_keys = [
                key for key in ordered_keys if rerank_scores.get(key, float("-inf")) >= min_rerank_score
            ]

        sources = []
        for key in ordered_keys[:top_k]:
            methods = tuple(method for method, keys in rankings.items() if key in keys)
            sources.append(
                self._make_source(
                    documents[key],
                    distances.get(key),
                    fused_scores[key],
                    methods,
                    rerank_scores.get(key),
                )
            )
        backend = self.reranker_status if rerank and ordered_keys else "none"
        return RetrievalResult(
            question, mode, reranked, False, sources, backend,
            rewritten_query=question, original_query=question,
        )

    def rewrite_question(self, question: str) -> tuple[str, bool]:
        messages = [
            {
                "role": "system",
                "content": (
                    "你是检索查询改写器。提取问题中的核心实体、术语和限制条件，"
                    "输出一条适合知识库检索的简短查询。只输出查询，不回答问题，不解释。"
                ),
            },
            {"role": "user", "content": question},
        ]
        try:
            rewritten = self.llm.complete(messages, max_tokens=64).strip().strip('"“”')
        except Exception as exc:
            logger.warning("查询改写失败，使用原始问题：%s", exc)
            return question, False
        if not rewritten or len(rewritten) > 200 or "\n" in rewritten:
            return question, False
        return rewritten, rewritten != question

    def prepare(
        self,
        question: str,
        use_rag: bool = True,
        top_k: int = 4,
        max_distance: float = 1.2,
        retrieval_mode: str = "hybrid",
        rerank: bool = True,
        rewrite_query: bool = True,
        min_rerank_score: float | None = None,
        document_type: str = "all",
        validity: str = "all",
    ) -> tuple[list[dict[str, str]], RetrievalResult]:
        question = question.strip()
        if not question:
            raise ValueError("问题不能为空。")

        route_llm = None
        if use_rag and self.intent_router.mode != "rule":
            try:
                route_llm = self.llm
            except Exception as exc:
                logger.warning("意图路由模型不可用，使用规则回退：%s", exc)
        decision = self.intent_router.route(question, route_llm)

        if not use_rag:
            if self.domain_profile == "legal_assistant":
                system_prompt = (
                    "你是法律资料辅助助手。当前没有提供知识库证据，不得编造具体法条、判例或确定性法律结论；"
                    "请建议用户提供资料或咨询专业律师。回答仅供资料检索和案件辅助分析，不构成律师意见。"
                )
            else:
                system_prompt = "请根据你已有的知识简洁、准确地回答用户问题。"
            return [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question},
            ], RetrievalResult(
                question, "none", False, False, [], rewritten_query=question,
                intent=decision.intent, intent_confidence=decision.confidence,
                route_source=decision.route_source, generation_chain=decision.generation_chain,
                original_query=question,
            )

        retrieval_query, rewritten = self.rewrite_question(question) if rewrite_query else (question, False)
        effective_document_type = document_type
        effective_validity = validity
        if self.domain_profile == "legal_assistant":
            if decision.intent in {"current_law"} and effective_document_type == "all":
                effective_document_type, effective_validity = "law", "current"
            elif decision.intent == "historical_law" and effective_document_type == "all":
                effective_document_type, effective_validity = "law", "historical"
            elif decision.intent in {"case_search", "case_analysis"} and effective_document_type == "all":
                effective_document_type = "case"
        retrieval = self.retrieve_advanced(
            retrieval_query,
            top_k=top_k,
            max_distance=max_distance,
            mode=retrieval_mode,
            rerank=rerank,
            min_rerank_score=min_rerank_score,
            document_type=effective_document_type,
            validity=effective_validity,
        )
        retrieval = RetrievalResult(
            retrieval.query,
            retrieval.mode,
            retrieval.reranked,
            rewritten,
            retrieval.sources,
            retrieval.reranker_backend,
            rewritten_query=retrieval_query,
            intent=decision.intent,
            intent_confidence=decision.confidence,
            route_source=decision.route_source,
            generation_chain=decision.generation_chain,
            original_query=question,
        )
        if not retrieval.sources:
            return [], retrieval

        return self.messages_for_sources(question, retrieval.sources, decision), retrieval

    def messages_for_sources(
        self,
        question: str,
        sources: list[Source],
        decision: IntentDecision | None = None,
    ) -> list[dict[str, str]]:
        decision = decision or IntentDecision("qa", 1.0, "默认普通问答", "rule")
        return self.prompt_orchestrator.build_messages(question, format_context(sources), decision)

    def stream_answer(
        self,
        question: str,
        use_rag: bool = True,
        top_k: int = 4,
        max_distance: float = 1.2,
        retrieval_mode: str = "hybrid",
        rerank: bool = True,
        rewrite_query: bool = True,
        min_rerank_score: float | None = None,
        document_type: str = "all",
        validity: str = "all",
    ) -> tuple[RetrievalResult, Iterator[str]]:
        messages, retrieval = self.prepare(
            question,
            use_rag,
            top_k,
            max_distance,
            retrieval_mode,
            rerank,
            rewrite_query,
            min_rerank_score,
            document_type,
            validity,
        )
        if use_rag and not retrieval.sources:
            return retrieval, iter(["知识库中没有足够信息，无法回答这个问题。"])
        return retrieval, self.llm.stream(messages)
