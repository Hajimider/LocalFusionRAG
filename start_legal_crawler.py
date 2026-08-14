"""法律公开资料爬虫入口：修改主要配置后，在 IDE 中直接运行本文件。"""

import json
from pathlib import Path

from scripts.sync_legal_docs import CrawlerConfig, LegalCrawler, load_seed_file


# ==================== 主要配置：日常只修改这里 ====================

# 1. 种子文件：每行一个公开官方页面直达 URL，以 # 开头的行会被忽略。
SEED_FILE = Path("data/legal_seed_urls.txt")

# 2. 允许访问的官方域名。爬虫拒绝跳转到白名单之外的网站。
ALLOWED_DOMAINS = {
    "flk.npc.gov.cn",
    "www.gov.cn",
    "www.court.gov.cn",
    "rmfyalk.court.gov.cn",
    "www.chinacourt.org",
}

# 3. 输出目录：抓取结果直接作为法律 RAG 知识库资料。
OUTPUT_DIR = Path("knowledge_base/legal_docs")

# 4. 抓取规模：建议先用 30 页、1 层测试，确认结果后再逐步增大。
MAX_PAGES = 30
MAX_DEPTH = 1

# 5. 请求间隔与超时：官方站点请保持低频访问，不建议小于 2 秒。
REQUEST_INTERVAL_SECONDS = 2.0
TIMEOUT_SECONDS = 20.0

# ==================== 配置结束：以下代码通常不用修改 ====================


def main() -> None:
    config = CrawlerConfig(
        seeds=load_seed_file(SEED_FILE),
        allowed_domains=frozenset(ALLOWED_DOMAINS),
        output_dir=OUTPUT_DIR,
        max_pages=MAX_PAGES,
        max_depth=MAX_DEPTH,
        delay_seconds=REQUEST_INTERVAL_SECONDS,
        timeout_seconds=TIMEOUT_SECONDS,
    )
    summary = LegalCrawler(config).crawl()
    print("法律资料抓取完成：")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
