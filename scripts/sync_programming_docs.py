from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_CACHE = BASE_DIR / "data" / "programming_docs_sources"
DEFAULT_OUTPUT = BASE_DIR / "knowledge_base" / "programming_docs"


@dataclass(frozen=True)
class SourceRepository:
    name: str
    url: str
    docs_path: str
    license_name: str


SOURCES = (
    # 语言、运行时与类型系统
    SourceRepository("nodejs", "https://github.com/nodejs/node.git", "doc/api", "Node.js license；以仓库条款为准"),
    SourceRepository("typescript", "https://github.com/microsoft/TypeScript-Website.git", "packages/documentation/copy/en", "CC BY 4.0；以仓库条款为准"),
    # Web 前端与后端框架
    SourceRepository("fastapi", "https://github.com/fastapi/fastapi.git", "docs", "MIT；以仓库条款为准"),
    SourceRepository("react", "https://github.com/reactjs/react.dev.git", "src/content", "CC BY 4.0；以仓库条款为准"),
    SourceRepository("vue", "https://github.com/vuejs/docs.git", "src", "CC BY 4.0；以仓库条款为准"),
    SourceRepository("nextjs", "https://github.com/vercel/next.js.git", "docs", "MIT；以仓库条款为准"),
    SourceRepository("pydantic", "https://github.com/pydantic/pydantic.git", "docs", "MIT；以仓库条款为准"),
    # 数据库、容器与云原生
    SourceRepository("redis", "https://github.com/redis/docs.git", "content", "CC BY 4.0；以仓库条款为准"),
    SourceRepository("docker", "https://github.com/docker/docs.git", "content", "Apache-2.0；以仓库条款为准"),
    SourceRepository("kubernetes", "https://github.com/kubernetes/website.git", "content/en/docs", "CC BY 4.0；以仓库条款为准"),
    # AI 应用与大模型工程
    SourceRepository("langchain", "https://github.com/langchain-ai/langchain.git", "docs", "MIT；以仓库条款为准"),
    SourceRepository("llamaindex", "https://github.com/run-llama/llama_index.git", "docs", "MIT；以仓库条款为准"),
    SourceRepository("openai_cookbook", "https://github.com/openai/openai-cookbook.git", "examples", "MIT；以仓库条款为准"),
    # 测试与工程工具
    SourceRepository("playwright", "https://github.com/microsoft/playwright.git", "docs", "Apache-2.0；以仓库条款为准"),
)


def run_git(*args: str, cwd: Path | None = None) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return result.stdout.strip()


def sync_repository(source: SourceRepository, cache_root: Path) -> tuple[Path, str]:
    repository = cache_root / source.name
    if (repository / ".git").is_dir():
        run_git("pull", "--ff-only", cwd=repository)
    else:
        run_git(
            "clone", "--depth", "1", "--filter=blob:none", "--sparse", source.url, str(repository)
        )
        run_git("sparse-checkout", "set", source.docs_path, "LICENSE", "LICENSE.md", cwd=repository)
    return repository, run_git("rev-parse", "HEAD", cwd=repository)


def normalized_path(source: SourceRepository, relative: Path) -> Path:
    suffix = ".md" if relative.suffix.lower() == ".mdx" else relative.suffix.lower()
    return Path(source.name) / relative.with_suffix(suffix)


def export_documents(
    source: SourceRepository,
    repository: Path,
    commit: str,
    output_root: Path,
    min_chars: int,
) -> int:
    docs_root = repository / source.docs_path
    count = 0
    for path in sorted(docs_root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in {".md", ".mdx"}:
            continue
        text = path.read_text(encoding="utf-8", errors="replace").strip()
        if len(text) < min_chars:
            continue
        relative = path.relative_to(docs_root)
        destination = output_root / normalized_path(source, relative)
        destination.parent.mkdir(parents=True, exist_ok=True)
        header = (
            "---\n"
            f"source_repository: {source.url}\n"
            f"source_commit: {commit}\n"
            f"source_path: {source.docs_path}/{relative.as_posix()}\n"
            f"license: {source.license_name}\n"
            "---\n\n"
        )
        destination.write_text(header + text + "\n", encoding="utf-8")
        count += 1
    return count


def sync(cache_root: Path, output_root: Path, min_docs: int, min_chars: int) -> dict:
    cache_root.mkdir(parents=True, exist_ok=True)
    staging = output_root.with_name(f".{output_root.name}.next")
    backup = output_root.with_name(f".{output_root.name}.backup")
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True)

    source_reports = []
    for source in SOURCES:
        print(f"同步 {source.name} ...", flush=True)
        repository, commit = sync_repository(source, cache_root)
        count = export_documents(source, repository, commit, staging, min_chars)
        source_reports.append(
            {
                "name": source.name,
                "repository": source.url,
                "commit": commit,
                "license": source.license_name,
                "documents": count,
            }
        )

    total = sum(item["documents"] for item in source_reports)
    if total < min_docs:
        shutil.rmtree(staging, ignore_errors=True)
        raise RuntimeError(f"有效文档只有 {total} 篇，低于要求的 {min_docs} 篇，旧知识库未修改。")

    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "document_count": total,
        "minimum_required": min_docs,
        "sources": source_reports,
    }
    (staging / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    shutil.rmtree(backup, ignore_errors=True)
    if output_root.exists():
        output_root.replace(backup)
    staging.replace(output_root)
    shutil.rmtree(backup, ignore_errors=True)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="同步官方编程文档并构建 Markdown 知识库")
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE, help="官方仓库缓存目录")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Markdown 知识库目录")
    parser.add_argument("--min-docs", type=int, default=1000, help="激活新知识库所需的最少文档数")
    parser.add_argument("--min-chars", type=int, default=200, help="过滤过短文档的字符数")
    args = parser.parse_args()
    result = sync(args.cache.resolve(), args.output.resolve(), args.min_docs, args.min_chars)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
