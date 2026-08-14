from __future__ import annotations

import argparse
import hashlib
import ipaddress
import json
import re
import socket
import time
from collections import deque
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from typing import Iterable
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qsl, urlencode, urldefrag, urljoin, urlsplit, urlunsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener
from urllib.robotparser import RobotFileParser


LEGAL_KEYWORDS = (
    "法律", "法规", "条例", "办法", "规定", "法条", "司法解释", "决定", "意见",
    "判决", "裁定", "案例", "判例", "案号", "法院", "裁判", "争议焦点", "指导性案例",
)
CASE_KEYWORDS = ("判决书", "裁定书", "指导性案例", "指导案例", "参考案例", "判例", "案号", "原告", "被告", "裁判要旨", "基本案情")
LAW_TYPE_KEYWORDS = ("中华人民共和国", "法律", "法规", "条例", "办法", "规定", "司法解释")
LINK_KEYWORDS = LEGAL_KEYWORDS + ("民法", "刑法", "诉讼", "行政", "劳动", "公司", "知识产权")
SKIP_SUFFIXES = {
    ".jpg", ".jpeg", ".png", ".gif", ".webp", ".svg", ".css", ".js", ".mp3",
    ".mp4", ".avi", ".zip", ".rar", ".7z", ".doc", ".docx", ".xls", ".xlsx",
}
BLOCK_TAGS = {"p", "div", "section", "article", "main", "li", "br", "h1", "h2", "h3", "h4", "tr"}
IGNORED_TAGS = {"script", "style", "noscript", "svg", "nav", "footer", "header", "form"}
TRACKING_QUERY_KEYS = {"spm", "from", "source", "ref"}


@dataclass(frozen=True)
class CrawlerConfig:
    seeds: tuple[str, ...]
    allowed_domains: frozenset[str]
    output_dir: Path
    max_pages: int = 100
    max_depth: int = 1
    delay_seconds: float = 2.0
    timeout_seconds: float = 20.0
    max_bytes: int = 15 * 1024 * 1024
    min_text_chars: int = 300
    obey_robots: bool = True
    allow_http: bool = False
    allow_private_hosts: bool = False
    user_agent: str = "LocalFusionRAG-LegalCrawler/1.0 (public legal research)"


@dataclass(frozen=True)
class CrawlEvent:
    url: str
    status: str
    detail: str = ""
    file: str = ""
    content_type: str = ""
    sha256: str = ""


@dataclass(frozen=True)
class FetchResult:
    url: str
    content_type: str
    charset: str
    body: bytes


class NoRedirectHandler(HTTPRedirectHandler):
    """禁止 urllib 自动跳转，由爬虫逐跳执行安全与 robots 检查。"""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


class LegalHTMLParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.title_parts: list[str] = []
        self.text_parts: list[str] = []
        self.links: list[tuple[str, str]] = []
        self._ignored_depth = 0
        self._in_title = False
        self._link_href = ""
        self._link_text: list[str] = []

    def handle_starttag(self, tag: str, attrs) -> None:
        tag = tag.lower()
        attributes = dict(attrs)
        if tag in IGNORED_TAGS:
            self._ignored_depth += 1
            return
        if self._ignored_depth:
            return
        if tag == "title":
            self._in_title = True
        if tag == "a":
            self._link_href = attributes.get("href", "").strip()
            self._link_text = []
        if tag in BLOCK_TAGS:
            self.text_parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in IGNORED_TAGS and self._ignored_depth:
            self._ignored_depth -= 1
            return
        if self._ignored_depth:
            return
        if tag == "title":
            self._in_title = False
        if tag == "a" and self._link_href:
            self.links.append((self._link_href, " ".join(self._link_text).strip()))
            self._link_href = ""
            self._link_text = []
        if tag in BLOCK_TAGS:
            self.text_parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._ignored_depth:
            return
        value = re.sub(r"\s+", " ", data).strip()
        if not value:
            return
        if self._in_title:
            self.title_parts.append(value)
        if self._link_href:
            self._link_text.append(value)
        self.text_parts.append(value)

    @property
    def title(self) -> str:
        return " ".join(self.title_parts).strip()

    @property
    def text(self) -> str:
        lines = []
        for line in "".join(self.text_parts).splitlines():
            clean = re.sub(r"[ \t]+", " ", line).strip()
            if clean and (not lines or clean != lines[-1]):
                lines.append(clean)
        return "\n\n".join(lines)


def normalize_url(url: str, base_url: str = "") -> str:
    absolute = urljoin(base_url, url.strip())
    absolute, _ = urldefrag(absolute)
    parts = urlsplit(absolute)
    if parts.scheme.lower() not in {"http", "https"} or not parts.hostname:
        return ""
    query = [
        (key, value)
        for key, value in parse_qsl(parts.query, keep_blank_values=True)
        if not key.lower().startswith("utm_") and key.lower() not in TRACKING_QUERY_KEYS
    ]
    netloc = parts.hostname.lower()
    if parts.port and not ((parts.scheme == "http" and parts.port == 80) or (parts.scheme == "https" and parts.port == 443)):
        netloc = f"{netloc}:{parts.port}"
    path = re.sub(r"/{2,}", "/", parts.path or "/")
    return urlunsplit((parts.scheme.lower(), netloc, path, urlencode(sorted(query)), ""))


def host_is_allowed(host: str, allowed_domains: Iterable[str]) -> bool:
    host = host.lower().rstrip(".")
    return any(host == domain.lower().rstrip(".") or host.endswith("." + domain.lower().rstrip(".")) for domain in allowed_domains)


def host_is_public(host: str) -> bool:
    try:
        addresses = {item[4][0] for item in socket.getaddrinfo(host, None)}
    except socket.gaierror:
        return False
    for address in addresses:
        try:
            ip = ipaddress.ip_address(address.split("%", 1)[0])
        except ValueError:
            return False
        if not ip.is_global:
            return False
    return bool(addresses)


def robots_interval(parser: RobotFileParser | None, user_agent: str, configured_delay: float) -> float:
    if parser is None:
        return configured_delay
    delay = parser.crawl_delay(user_agent)
    if delay is None:
        delay = parser.crawl_delay("*")
    rate = parser.request_rate(user_agent) or parser.request_rate("*")
    rate_delay = rate.seconds / rate.requests if rate and rate.requests > 0 else 0.0
    return max(configured_delay, float(delay or 0.0), rate_delay)


def decode_html(body: bytes, charset: str) -> str:
    encodings = [charset, "utf-8", "gb18030"]
    for encoding in dict.fromkeys(value for value in encodings if value):
        try:
            return body.decode(encoding)
        except (LookupError, UnicodeDecodeError):
            continue
    return body.decode("utf-8", errors="replace")


def legal_metadata(title: str, text: str, url: str) -> dict[str, str]:
    sample = f"{title}\n{text[:8000]}"
    host = (urlsplit(url).hostname or "").lower()
    if any(keyword in sample for keyword in CASE_KEYWORDS) or any(value in host for value in ("rmfyalk", "wenshu")):
        doc_type = "case"
    elif any(keyword in sample for keyword in LAW_TYPE_KEYWORDS):
        doc_type = "law"
    else:
        doc_type = "unknown"
    if any(keyword in sample for keyword in ("已废止", "已失效", "废止决定", "历史版本")):
        validity = "historical"
    elif "现行有效" in sample:
        validity = "current"
    else:
        validity = "unknown"
    case_number = ""
    match = re.search(r"[（(]\d{4}[）)][^\s，。；;]{1,40}号", sample)
    if match:
        case_number = match.group(0)
    court = ""
    match = re.search(r"([\u4e00-\u9fff]{2,30}人民法院)", sample)
    if match:
        court = match.group(1)
    return {
        "doc_type": doc_type,
        "validity": validity,
        "jurisdiction": "",
        "effective_date": "",
        "expiry_date": "",
        "court": court,
        "case_number": case_number,
        "judgment_date": "",
        "title": title or "未命名法律资料",
        "source_url": url,
        "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "metadata_review": "required",
    }


def safe_filename(title: str, url: str, suffix: str) -> str:
    clean = re.sub(r"[<>:\"/\\|?*\x00-\x1f]", "_", title).strip(" ._")
    clean = re.sub(r"\s+", "_", clean)[:72] or "legal_document"
    identity = hashlib.sha256(url.encode("utf-8")).hexdigest()[:12]
    return f"{clean}_{identity}{suffix}"


def markdown_document(metadata: dict[str, str], text: str) -> str:
    front_matter = ["---"]
    for key, value in metadata.items():
        front_matter.append(f"{key}: {json.dumps(value, ensure_ascii=False)}")
    front_matter.extend(["---", "", f"# {metadata['title']}", "", text.strip(), ""])
    return "\n".join(front_matter)


def atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.part")
    temporary.write_bytes(data)
    temporary.replace(path)


class LegalCrawler:
    def __init__(self, config: CrawlerConfig) -> None:
        if not config.seeds:
            raise ValueError("至少需要一个种子 URL。")
        if not config.allowed_domains:
            raise ValueError("官方域名白名单不能为空。")
        if config.max_pages <= 0 or config.max_depth < 0 or config.delay_seconds < 0:
            raise ValueError("页数、深度和请求间隔配置无效。")
        self.config = config
        self.robots: dict[str, RobotFileParser | None] = {}
        self.last_request: dict[str, float] = {}
        self.hash_files: dict[str, str] = self._load_hash_files()
        self.seen_hashes: set[str] = set(self.hash_files)
        self.events: list[CrawlEvent] = []
        self.opener = build_opener(NoRedirectHandler())

    def _load_hash_files(self) -> dict[str, str]:
        path = self.config.output_dir / "crawl_hashes.json"
        if not path.is_file():
            return {}
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        return {
            digest: filename
            for digest, filename in value.items()
            if (
                isinstance(digest, str)
                and isinstance(filename, str)
                and filename == Path(filename).name
                and (self.config.output_dir / filename).is_file()
            )
        } if isinstance(value, dict) else {}

    def _validate_url(self, url: str) -> tuple[bool, str]:
        parts = urlsplit(url)
        if parts.scheme == "http" and not self.config.allow_http:
            return False, "只允许 HTTPS"
        if parts.scheme not in {"http", "https"} or not parts.hostname:
            return False, "URL 格式无效"
        if not host_is_allowed(parts.hostname, self.config.allowed_domains):
            return False, "域名不在白名单"
        if not self.config.allow_private_hosts and not host_is_public(parts.hostname):
            return False, "目标域名未解析到公开网络地址"
        return True, ""

    def _wait(self, host: str, minimum_delay: float | None = None) -> None:
        elapsed = time.monotonic() - self.last_request.get(host, 0.0)
        delay = self.config.delay_seconds if minimum_delay is None else minimum_delay
        if elapsed < delay:
            time.sleep(delay - elapsed)
        self.last_request[host] = time.monotonic()

    def _robots_for(self, url: str) -> RobotFileParser | None:
        parts = urlsplit(url)
        host_key = f"{parts.scheme}://{parts.netloc}"
        if host_key in self.robots:
            return self.robots[host_key]
        robots_url = f"{host_key}/robots.txt"
        try:
            result = self._fetch(robots_url, check_robots=False)
            parser = RobotFileParser(robots_url)
            parser.parse(decode_html(result.body, result.charset).splitlines())
            self.robots[host_key] = parser
        except HTTPError as exc:
            if exc.code == 404:
                self.robots[host_key] = None
            else:
                parser = RobotFileParser(robots_url)
                parser.parse(["User-agent: *", "Disallow: /"])
                self.robots[host_key] = parser
        except (URLError, TimeoutError, ValueError, OSError):
            parser = RobotFileParser(robots_url)
            parser.parse(["User-agent: *", "Disallow: /"])
            self.robots[host_key] = parser
        return self.robots[host_key]

    def _fetch(self, url: str, check_robots: bool = True) -> FetchResult:
        current_url = url
        for _ in range(6):
            valid, reason = self._validate_url(current_url)
            if not valid:
                raise ValueError(f"重定向被拒绝：{reason}" if current_url != url else reason)
            minimum_delay = self.config.delay_seconds
            if check_robots and self.config.obey_robots:
                parser = self._robots_for(current_url)
                if parser is not None and not parser.can_fetch(self.config.user_agent, current_url):
                    raise PermissionError("robots.txt 禁止访问")
                minimum_delay = robots_interval(parser, self.config.user_agent, self.config.delay_seconds)
            host = urlsplit(current_url).hostname or ""
            self._wait(host, minimum_delay)
            request = Request(current_url, headers={
                "User-Agent": self.config.user_agent,
                "Accept": "text/html,application/pdf;q=0.9,*/*;q=0.1",
            })
            try:
                response = self.opener.open(request, timeout=self.config.timeout_seconds)
            except HTTPError as exc:
                if exc.code not in {301, 302, 303, 307, 308} or not exc.headers.get("Location"):
                    raise
                target = normalize_url(exc.headers["Location"], current_url)
                exc.close()
                if not target:
                    raise ValueError("重定向目标 URL 无效")
                current_url = target
                continue
            with response:
                declared_length = response.headers.get("Content-Length")
                if declared_length and int(declared_length) > self.config.max_bytes:
                    raise ValueError("响应体超过大小限制")
                body = response.read(self.config.max_bytes + 1)
                if len(body) > self.config.max_bytes:
                    raise ValueError("响应体超过大小限制")
                content_type = response.headers.get_content_type().lower()
                charset = response.headers.get_content_charset() or ""
                return FetchResult(current_url, content_type, charset, body)
        raise ValueError("重定向次数超过 5 次")

    def _save_pdf(self, result: FetchResult) -> CrawlEvent:
        if not result.body.startswith(b"%PDF"):
            return CrawlEvent(result.url, "skipped", "内容类型为 PDF，但文件头无效", content_type=result.content_type)
        digest = hashlib.sha256(result.body).hexdigest()
        if digest in self.seen_hashes:
            return CrawlEvent(
                result.url,
                "duplicate",
                "正文哈希重复",
                self.hash_files.get(digest, ""),
                result.content_type,
                digest,
            )
        title = Path(urlsplit(result.url).path).stem or "legal_document"
        metadata = legal_metadata(title, "", result.url)
        filename = safe_filename(title, result.url, ".pdf")
        path = self.config.output_dir / filename
        atomic_write(path, result.body)
        atomic_write(path.with_suffix(path.suffix + ".json"), json.dumps(metadata, ensure_ascii=False, indent=2).encode("utf-8"))
        self.seen_hashes.add(digest)
        self.hash_files[digest] = filename
        return CrawlEvent(result.url, "saved", "PDF 及旁车元数据已保存", filename, result.content_type, digest)

    def _save_html(self, result: FetchResult) -> tuple[CrawlEvent, list[tuple[str, str]]]:
        parser = LegalHTMLParser()
        parser.feed(decode_html(result.body, result.charset))
        text = parser.text
        if len(text) < self.config.min_text_chars:
            return CrawlEvent(result.url, "skipped", "正文过短", content_type=result.content_type), parser.links
        sample = f"{parser.title}\n{text[:12000]}"
        if not any(keyword in sample for keyword in LEGAL_KEYWORDS):
            return CrawlEvent(result.url, "skipped", "未检测到法律相关内容", content_type=result.content_type), parser.links
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if digest in self.seen_hashes:
            return CrawlEvent(
                result.url,
                "duplicate",
                "正文哈希重复",
                self.hash_files.get(digest, ""),
                result.content_type,
                digest,
            ), parser.links
        metadata = legal_metadata(parser.title, text, result.url)
        filename = safe_filename(metadata["title"], result.url, ".md")
        atomic_write(self.config.output_dir / filename, markdown_document(metadata, text).encode("utf-8"))
        self.seen_hashes.add(digest)
        self.hash_files[digest] = filename
        return CrawlEvent(result.url, "saved", "HTML 已转换为 Markdown", filename, result.content_type, digest), parser.links

    def crawl(self) -> dict[str, object]:
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        queue = deque((normalize_url(seed), 0) for seed in self.config.seeds)
        visited: set[str] = set()
        while queue and len(visited) < self.config.max_pages:
            url, depth = queue.popleft()
            if not url or url in visited:
                continue
            visited.add(url)
            valid, reason = self._validate_url(url)
            if not valid:
                self.events.append(CrawlEvent(url, "blocked", reason))
                continue
            try:
                result = self._fetch(url)
                is_pdf = result.content_type == "application/pdf" or urlsplit(result.url).path.lower().endswith(".pdf")
                if is_pdf:
                    event = self._save_pdf(result)
                    links: list[tuple[str, str]] = []
                elif result.content_type in {"text/html", "application/xhtml+xml"}:
                    event, links = self._save_html(result)
                else:
                    event, links = CrawlEvent(result.url, "skipped", "不支持的内容类型", content_type=result.content_type), []
                self.events.append(event)
                if depth >= self.config.max_depth:
                    continue
                for href, anchor in links:
                    candidate = normalize_url(href, result.url)
                    if not candidate or candidate in visited or Path(urlsplit(candidate).path).suffix.lower() in SKIP_SUFFIXES:
                        continue
                    candidate_host = urlsplit(candidate).hostname or ""
                    if not host_is_allowed(candidate_host, self.config.allowed_domains):
                        continue
                    hint = f"{anchor} {urlsplit(candidate).path}"
                    if any(keyword in hint for keyword in LINK_KEYWORDS) or candidate.lower().endswith(".pdf"):
                        queue.append((candidate, depth + 1))
            except PermissionError as exc:
                self.events.append(CrawlEvent(url, "blocked", str(exc)))
            except (HTTPError, URLError, TimeoutError, ValueError, OSError) as exc:
                self.events.append(CrawlEvent(url, "failed", str(exc)[:300]))

        atomic_write(
            self.config.output_dir / "crawl_hashes.json",
            json.dumps(self.hash_files, ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8"),
        )
        manifest = self.config.output_dir / "crawl_manifest.jsonl"
        atomic_write(manifest, "".join(json.dumps(asdict(event), ensure_ascii=False) + "\n" for event in self.events).encode("utf-8"))
        counts: dict[str, int] = {}
        for event in self.events:
            counts[event.status] = counts.get(event.status, 0) + 1
        summary = {
            "finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "visited": len(visited),
            "counts": counts,
            "output_dir": self.config.output_dir.as_posix(),
        }
        atomic_write(self.config.output_dir / "crawl_summary.json", json.dumps(summary, ensure_ascii=False, indent=2).encode("utf-8"))
        return summary


def load_seed_file(path: Path) -> tuple[str, ...]:
    if not path.is_file():
        raise FileNotFoundError(f"种子文件不存在：{path}")
    return tuple(line.strip() for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip() and not line.lstrip().startswith("#"))


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="抓取公开官方法律页面并转换为 RAG 知识库资料")
    parser.add_argument("--seed-file", type=Path, default=Path("data/legal_seed_urls.txt"))
    parser.add_argument("--output-dir", type=Path, default=Path("knowledge_base/legal_docs"))
    parser.add_argument("--allowed-domain", action="append", required=True)
    parser.add_argument("--max-pages", type=int, default=100)
    parser.add_argument("--max-depth", type=int, default=1)
    parser.add_argument("--delay", type=float, default=2.0)
    parser.add_argument("--timeout", type=float, default=20.0)
    return parser


def main() -> None:
    args = create_parser().parse_args()
    config = CrawlerConfig(
        seeds=load_seed_file(args.seed_file),
        allowed_domains=frozenset(args.allowed_domain),
        output_dir=args.output_dir,
        max_pages=args.max_pages,
        max_depth=args.max_depth,
        delay_seconds=args.delay,
        timeout_seconds=args.timeout,
    )
    print(json.dumps(LegalCrawler(config).crawl(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
