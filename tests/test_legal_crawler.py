import json
import threading
from functools import partial
from http.server import BaseHTTPRequestHandler, SimpleHTTPRequestHandler, ThreadingHTTPServer
from urllib.robotparser import RobotFileParser

from scripts.sync_legal_docs import CrawlerConfig, LegalCrawler, host_is_allowed, normalize_url, robots_interval


class QuietHandler(SimpleHTTPRequestHandler):
    def log_message(self, *_args):
        pass


def route_handler(routes, hits):
    class RouteHandler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def do_GET(self):
            path = self.path.split("?", 1)[0]
            hits[path] = hits.get(path, 0) + 1
            status, headers, body = routes.get(path, (404, {"Content-Type": "text/plain"}, b"not found"))
            self.send_response(status)
            for key, value in headers.items():
                self.send_header(key, value)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    return RouteHandler


def start_server(handler):
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread


def stop_server(server, thread):
    server.shutdown()
    server.server_close()
    thread.join(timeout=2)


def local_config(seed, output, **overrides):
    values = {
        "seeds": (seed,),
        "allowed_domains": frozenset({"127.0.0.1"}),
        "output_dir": output,
        "max_pages": 10,
        "max_depth": 1,
        "delay_seconds": 0,
        "timeout_seconds": 2,
        "min_text_chars": 20,
        "allow_http": True,
        "allow_private_hosts": True,
    }
    values.update(overrides)
    return CrawlerConfig(**values)


def test_url_normalization_and_domain_allowlist():
    assert normalize_url("/law?a=1&utm_source=test#part", "https://www.gov.cn/root/") == "https://www.gov.cn/law?a=1"
    assert host_is_allowed("sub.court.gov.cn", {"court.gov.cn"})
    assert not host_is_allowed("court.gov.cn.example.com", {"court.gov.cn"})


def test_robots_interval_uses_strictest_rate():
    parser = RobotFileParser()
    parser.parse(["User-agent: *", "Crawl-delay: 3", "Request-rate: 2/10"])
    assert robots_interval(parser, "LocalFusionRAG-LegalCrawler/1.0", 1.0) == 5.0


def test_crawler_converts_html_pdf_respects_robots_and_deduplicates(tmp_path):
    web_root = tmp_path / "site"
    output = tmp_path / "legal_docs"
    web_root.mkdir()
    repeated = "中华人民共和国法律资料。人民法院依法审理案件并作出判决。" * 12
    (web_root / "robots.txt").write_text("User-agent: *\nDisallow: /blocked.html\n", encoding="utf-8")
    (web_root / "index.html").write_text(
        "<html><head><title>法律资料首页</title></head><body>"
        + "现行法律法规目录。" * 30
        + '<a href="/case.html">人民法院案例</a>'
        + '<a href="/duplicate.html">重复判例</a>'
        + '<a href="/doc.pdf">法律 PDF</a>'
        + '<a href="/blocked.html">禁止访问的法律页</a>'
        + "</body></html>",
        encoding="utf-8",
    )
    case_html = f"<html><head><title>人民法院指导案例</title></head><body>{repeated}</body></html>"
    (web_root / "case.html").write_text(case_html, encoding="utf-8")
    (web_root / "duplicate.html").write_text(case_html, encoding="utf-8")
    (web_root / "blocked.html").write_text("<html><body>法律秘密页面</body></html>", encoding="utf-8")
    (web_root / "doc.pdf").write_bytes(b"%PDF-1.4\n% local crawler test\n")

    server = ThreadingHTTPServer(("127.0.0.1", 0), partial(QuietHandler, directory=str(web_root)))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        base = f"http://127.0.0.1:{server.server_port}"
        config = CrawlerConfig(
            seeds=(f"{base}/index.html",),
            allowed_domains=frozenset({"127.0.0.1"}),
            output_dir=output,
            max_pages=5,
            max_depth=1,
            delay_seconds=0,
            timeout_seconds=2,
            min_text_chars=20,
            allow_http=True,
            allow_private_hosts=True,
        )
        summary = LegalCrawler(config).crawl()
        second_summary = LegalCrawler(config).crawl()
        case_file = next(
            path for path in output.glob("*.md")
            if 'doc_type: "case"' in path.read_text(encoding="utf-8")
        )
        case_file.unlink()
        third_summary = LegalCrawler(config).crawl()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)

    assert summary["counts"] == {"saved": 3, "duplicate": 1, "blocked": 1}
    assert summary["visited"] == 5
    assert second_summary["counts"] == {"duplicate": 4, "blocked": 1}
    assert third_summary["counts"] == {"duplicate": 3, "saved": 1, "blocked": 1}
    assert case_file.is_file()
    assert len(list(output.glob("*.md"))) == 2
    assert any('doc_type: "case"' in path.read_text(encoding="utf-8") for path in output.glob("*.md"))
    pdf_files = list(output.glob("*.pdf"))
    assert len(pdf_files) == 1
    sidecar = json.loads(pdf_files[0].with_suffix(".pdf.json").read_text(encoding="utf-8"))
    assert sidecar["source_url"].endswith("/doc.pdf")
    assert sidecar["metadata_review"] == "required"
    hashes = json.loads((output / "crawl_hashes.json").read_text(encoding="utf-8"))
    assert len(hashes) == 3
    manifest = [json.loads(line) for line in (output / "crawl_manifest.jsonl").read_text(encoding="utf-8").splitlines()]
    assert any(item["status"] == "blocked" and item["url"].endswith("/blocked.html") for item in manifest)
    assert all(item["file"] for item in manifest if item["status"] == "duplicate")


def test_redirect_is_blocked_before_requesting_disallowed_host(tmp_path):
    target_hits = {}
    target_server, target_thread = start_server(route_handler({
        "/target": (200, {"Content-Type": "text/html; charset=utf-8"}, b"target")
    }, target_hits))
    source_hits = {}
    redirect_url = f"http://localhost:{target_server.server_port}/target"
    source_server, source_thread = start_server(route_handler({
        "/robots.txt": (404, {"Content-Type": "text/plain"}, b""),
        "/start": (302, {"Location": redirect_url}, b""),
    }, source_hits))
    try:
        seed = f"http://127.0.0.1:{source_server.server_port}/start"
        summary = LegalCrawler(local_config(seed, tmp_path / "out")).crawl()
    finally:
        stop_server(source_server, source_thread)
        stop_server(target_server, target_thread)

    assert summary["counts"] == {"failed": 1}
    assert source_hits["/start"] == 1
    assert target_hits.get("/target", 0) == 0


def test_redirect_target_rechecks_cached_robots_rules(tmp_path):
    hits = {}
    server, thread = start_server(route_handler({
        "/robots.txt": (200, {"Content-Type": "text/plain"}, b"User-agent: *\nDisallow: /blocked\n"),
        "/start": (302, {"Location": "/blocked"}, b""),
        "/blocked": (200, {"Content-Type": "text/html; charset=utf-8"}, b"blocked"),
    }, hits))
    try:
        seed = f"http://127.0.0.1:{server.server_port}/start"
        summary = LegalCrawler(local_config(seed, tmp_path / "out")).crawl()
    finally:
        stop_server(server, thread)

    assert summary["counts"] == {"blocked": 1}
    assert hits["/start"] == 1
    assert hits.get("/blocked", 0) == 0


def test_cross_host_redirect_checks_target_robots_and_rate(tmp_path):
    target_hits = {}
    target_server, target_thread = start_server(route_handler({
        "/robots.txt": (200, {"Content-Type": "text/plain"}, b"User-agent: *\nCrawl-delay: 7\n"),
        "/target": (200, {"Content-Type": "text/html; charset=utf-8"}, b"short"),
    }, target_hits))
    source_hits = {}
    target_url = f"http://localhost:{target_server.server_port}/target"
    source_server, source_thread = start_server(route_handler({
        "/robots.txt": (404, {"Content-Type": "text/plain"}, b""),
        "/start": (302, {"Location": target_url}, b""),
    }, source_hits))
    waits = []
    try:
        seed = f"http://127.0.0.1:{source_server.server_port}/start"
        crawler = LegalCrawler(local_config(
            seed,
            tmp_path / "out",
            allowed_domains=frozenset({"127.0.0.1", "localhost"}),
        ))
        crawler._wait = lambda host, minimum_delay=None: waits.append((host, minimum_delay))
        summary = crawler.crawl()
    finally:
        stop_server(source_server, source_thread)
        stop_server(target_server, target_thread)

    assert summary["counts"] == {"skipped": 1}
    assert target_hits["/robots.txt"] == 1
    assert target_hits["/target"] == 1
    assert ("localhost", 7.0) in waits


def test_robots_error_blocks_content_request(tmp_path):
    hits = {}
    server, thread = start_server(route_handler({
        "/robots.txt": (500, {"Content-Type": "text/plain"}, b"error"),
        "/index": (200, {"Content-Type": "text/html; charset=utf-8"}, b"legal content"),
    }, hits))
    try:
        seed = f"http://127.0.0.1:{server.server_port}/index"
        summary = LegalCrawler(local_config(seed, tmp_path / "out")).crawl()
    finally:
        stop_server(server, thread)

    assert summary["counts"] == {"blocked": 1}
    assert hits["/robots.txt"] == 1
    assert hits.get("/index", 0) == 0


def test_response_size_and_depth_limits(tmp_path):
    hits = {}
    index = (
        "<html><head><title>Legal index</title></head><body>"
        + "law regulation article " * 30
        + '<a href="/child">law child</a></body></html>'
    ).encode()
    server, thread = start_server(route_handler({
        "/index": (200, {"Content-Type": "text/html; charset=utf-8"}, index),
        "/child": (200, {"Content-Type": "text/html; charset=utf-8"}, index),
        "/large": (200, {"Content-Type": "text/html; charset=utf-8"}, b"x" * 100),
    }, hits))
    try:
        base = f"http://127.0.0.1:{server.server_port}"
        depth_summary = LegalCrawler(local_config(
            f"{base}/index", tmp_path / "depth", max_depth=0, obey_robots=False
        )).crawl()
        size_summary = LegalCrawler(local_config(
            f"{base}/large", tmp_path / "size", max_bytes=20, obey_robots=False
        )).crawl()
    finally:
        stop_server(server, thread)

    assert depth_summary["visited"] == 1
    assert hits.get("/child", 0) == 0
    assert size_summary["counts"] == {"failed": 1}
    assert not list((tmp_path / "size").glob("*.md"))
