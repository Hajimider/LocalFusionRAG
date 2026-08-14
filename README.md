# LocalFusionRAG

面向中国大陆中文法律资料的本地混合检索与案件辅助分析系统。

## 1. 实际业务问题

法律工作需要在现行法条、历史版本和公开判例之间快速定位证据，但资料格式分散、版本状态容易混淆，直接让大模型回答又可能脱离依据或把个案判决误当成普遍规则。本项目把已下载或获授权的法律资料统一导入本地知识库，提供可筛选、可引用、低置信度可拒答的检索问答链路。

## 2. 核心能力

- 支持 PDF、Markdown、TXT 导入、切分、哈希去重和 FAISS 版本化索引。
- 使用 BGE 语义检索与中文 BM25 关键词检索，通过 RRF 融合，并接入 CrossEncoder 或 BGE 向量回退重排序。
- 支持现行法条、历史法条、判例检索、案件分析、法律对比和时间线意图路由。
- Markdown YAML front matter 支持 `doc_type`、`validity`、`jurisdiction`、日期、法院、案号和来源链接过滤。
- 支持本地 GGUF 模型或 OpenAI 兼容 API，输出来源引用并在资料不足时拒答。
- FastAPI 网页支持上传、重建索引、流式问答、来源展开和法律资料筛选。

## 3. 目录结构

```text
LocalFusionRAG/
├── app.py                 # FastAPI 接口
├── start_web.py           # IDE 一键启动与主要配置
├── start_legal_crawler.py # IDE 一键采集公开法律资料
├── run_project.py         # 通用启动命令
├── rag_core.py            # 解析、索引、召回、重排序和生成
├── rag_orchestration.py   # 法律意图路由与 Prompt 编排
├── knowledge_base/
│   └── legal_docs/        # 放入已授权法条/判例资料
├── storage/faiss/         # 索引版本目录
├── web/                   # 网页界面
├── tests/                 # 自动化测试
└── requirements.txt
```

## 4. 数据准备

请从官方或获授权渠道下载资料并放入 `knowledge_base/legal_docs/`。推荐入口：

- 国家法律法规数据库：https://flk.npc.gov.cn/
- 人民法院案例库：https://rmfyalk.court.gov.cn/
- 中国法院网：https://www.chinacourt.org/
- 中国裁判文书网：https://wenshu.court.gov.cn/

不要将未获授权的批量案例、个人敏感信息或真实 API Key 提交到 GitHub。

法律 Markdown 可使用以下头信息：

```yaml
---
doc_type: law       # law 或 case
validity: current   # current、historical 或 unknown
jurisdiction: 全国
effective_date: 2024-01-01
expiry_date:
title: 文件标题
source_url: https://官方来源地址
---
```

系统支持 PDF、DOCX、Markdown 和 TXT。PDF 不能写 front matter 时，可在同目录放同名 `.pdf.json` 或 `.json` 旁车文件，内容使用同样的字段；系统会把旁车元数据复制到每一页。DOCX 会直接提取段落和表格文字，并根据“宪法、法律、行政法规、监察法规、司法解释、地方法规”目录生成待核验分类。目录说明文件和以下划线开头的模板文件不会进入索引。

旧版二进制 `.doc` 不会直接进入索引，需要先用 Word 或 LibreOffice 转换为 `.docx`、PDF 或 TXT。转换时保留原文件，避免覆盖法律资料原件。

判例请补充 `court`、`case_number`、`judgment_date`。字段不确定时留空，系统不会自动推断效力。

## 5. 采集公开资料

项目提供标准库爬虫，不需要额外安装爬虫依赖。先在 `data/legal_seed_urls.txt` 中填写公开官方页面的直达 URL，再打开 `start_legal_crawler.py`：

- `ALLOWED_DOMAINS`：允许访问的官方域名白名单。
- `MAX_PAGES`、`MAX_DEPTH`：最大页数和链接跟进深度，首次建议保持 30 页、1 层。
- `REQUEST_INTERVAL_SECONDS`：同一站点的访问间隔，建议不少于 2 秒。
- `OUTPUT_DIR`：抓取结果目录，默认直接写入法律知识库。

在 IDE 中直接运行 `start_legal_crawler.py`。HTML 会转成带 front matter 的 Markdown，PDF 会生成同名 `.pdf.json`；`crawl_manifest.jsonl` 记录每个 URL 的成功、重复、跳过或失败状态，`crawl_summary.json` 提供汇总，`crawl_hashes.json` 保存正文哈希与文件映射，用于后续运行继续去重。

爬虫只访问白名单内公开页面并遵守 `robots.txt`，站点声明的 `Crawl-delay` 或 `Request-rate` 比本地配置更严格时以站点规则为准；除明确返回 404 外，无法读取 robots 规则时会拒绝抓取正文。爬虫不会登录、处理验证码或绕过访问控制，依赖 JavaScript 渲染且没有静态正文的页面会记录为跳过，需要改用具体静态页面或手动下载。

自动生成的资料类型、效力、法院和案号属于启发式元数据，并带有 `metadata_review: required` 标记。建立正式知识库前应回到官方来源人工核验，不要直接把自动分类结果当作法律事实。

抓取完成后运行 `start_web.py`，点击“重建知识库索引”。下载语料默认被 `.gitignore` 排除，不会随代码上传 GitHub。

## 6. 安装

```bash
python -m venv .venv
python -m pip install -r requirements.txt
```

本地 GGUF 需要额外安装与你的 CPU 匹配的 `llama-cpp-python`；也可以使用 OpenAI 兼容 API，避免本地加载大模型。

## 7. 启动

打开 `start_web.py`，只修改文件顶部的主要配置：

- `LLM_PROVIDER`：`local` 或 `api`。
- `LOCAL_MODEL_PATH`：本地 GGUF 文件或目录，API 模式留空。
- `API_BASE_URL`、`API_KEY`、`API_MODEL`：API 模式填写，密钥不要提交。
- `EMBEDDING_MODEL`、`RERANKER_MODEL`：填写本地模型目录；留空使用默认名称或回退。
- `KNOWLEDGE_DIR`：法律资料目录，可改为外部目录。
- `MODEL_REPOSITORY_OFFLINE`：模型已经下载到本地时设为 `True`。

在 IDE 中直接运行 `start_web.py`，浏览器会打开 `http://127.0.0.1:8000`。首次使用先上传资料或复制资料，再点击“重建知识库索引”。

### 首次建库耗时

当前法律语料包含 3,183 份可读取 DOCX，切分后约为 40,781 个文本块。在普通 16 GB 内存的笔记本 CPU 环境中，首次完成 BGE 向量计算和 FAISS 保存通常需要 45～120 分钟，具体取决于可用内存、CPU 功耗模式和后台程序。文档读取与切分只占少量时间，主要耗时在向量计算。

建库期间请保持电脑接通电源，不要关闭服务、重复点击“重建知识库索引”或同时加载本地 7B 大模型。程序会在全部向量计算完成后统一写入新索引，因此处理中索引目录长时间没有新文件属于正常现象；任务管理器中 Python 进程持续占用 CPU，说明仍在正常工作。只有 Python 连续约 10 分钟几乎不占用 CPU、终端没有新错误且页面一直无响应时，才需要按故障处理。

当前版本每次重建都会重新处理全部资料，不是增量更新。首次索引成功后可直接问答，只有新增、删除或修改知识库文档时才需要再次重建。

## 8. 问答示例

- “现行有效的相关法条有哪些？”并选择“仅法条 / 现行有效”。
- “某案号的裁判理由是什么？”并选择“仅判例”。
- “这组事实涉及哪些争议焦点？”保留“法条与判例”。

回答中的 `[资料N]` 对应下方可展开的来源。系统只做资料检索和辅助分析，不替代律师意见或正式法律意见。

## 9. 测试

```bash
python -m pytest -q
```

测试覆盖前置元数据解析、法律意图路由、资料类型/效力过滤、混合召回、重排序回退、拒答和 API 接口，并验证爬虫重定向、robots、站点限速、响应大小、抓取深度与跨运行去重边界。

## 10. 许可与合规

代码按仓库许可使用；法律资料的版权、访问许可和个人信息处理由资料提供者负责。项目不保证资料实时更新，使用前应回到官方来源核对现行状态。
