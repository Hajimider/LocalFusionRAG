# Local-Hybrid-RAG-QA

## 项目说明

Local-Hybrid-RAG-QA 是一个面向中文技术文档的本地知识库问答与检索评测项目。项目支持 PDF、Markdown 和 TXT 文档，完整流程包括 LangChain 文档解析、重叠切分、BGE 向量化、FAISS 索引、BM25 关键词召回、RRF 融合、CrossEncoder 重排序、低相关拒答、本地或 API 大模型生成，以及 FastAPI 页面和流式接口。

项目使用固定随机种子从公开 `C-MTEB/T2Reranking` 构造 99 个查询、1,591 篇候选文档和 806 篇相关文档的受控子集。在 `chunk_size=300`、`chunk_overlap=50`、`Top-K=5` 下，Hybrid + Reranker 取得 `Hit@5 = 0.9798`、`Recall@5 = 0.4193`、`MRR = 0.7983`、`NDCG@5 = 0.6374`。这些结果来自固定公开子集实验，不是 C-MTEB 官方全量排行榜成绩。

回答阶段可通过 llama.cpp 调用本地 GGUF 模型，也可调用 OpenAI 兼容 API。系统只把召回片段交给大模型，回答中保留资料编号和来源；CrossEncoder 加载失败时退回 BGE 向量重排序，检索分数低于校准阈值时直接拒答，避免强行生成没有资料支持的结论。

## 实际问题与解决思路

企业内部的研发规范、运维手册和项目说明通常分散在多份文档中。员工需要反复打开文件、搜索关键词并人工拼接信息；当提问方式与原文不同，或答案分布在多份资料中时，查找成本会进一步增加。项目面向这一场景，把分散文档整理成可检索、可追溯的本地知识库。

| 问题 | 项目处理方式 | 可检查的输出 |
| --- | --- | --- |
| 技术资料分散在 PDF、Markdown 和 TXT 中，人工查找耗时 | 统一解析多格式文档，切分后建立本地知识库索引 | 文档区段数、检索片段数、知识库状态 |
| 员工的自然语言问法与文档术语不一致，关键词搜索容易漏掉答案 | 结合 BGE 语义检索与 BM25 关键词检索，并用 RRF 融合结果 | 检索方式、命中片段、四组消融指标 |
| 一个问题的答案可能分布在多份制度或操作手册中 | 召回多个候选片段，再用 CrossEncoder 按问题相关性重排 | 来源文件、PDF 页码、候选顺序和分数 |
| 知识库没有相关资料时，大模型仍可能根据自身知识猜测 | 使用校准后的相关性阈值拒答，只让模型根据已检索资料回答 | 拒答结果、阈值报告、独立测试指标 |
| 回答没有出处，员工难以确认内容是否来自现行资料 | 为上下文编号并随回答返回来源文件、页码和原文片段 | 回答引用、来源片段、检索查询 |
| 内部资料不适合上传到第三方服务 | 支持本地 BGE、FAISS、CrossEncoder 和 GGUF 模型，API 作为可选后端 | 当前模型后端、离线状态、健康检查 |

系统以知识库文档作为回答依据，不把大模型参数中的记忆当作内部事实。资料不足时直接说明无法回答，员工可以根据返回的文件和页码复核原文。

## 数据与问题

公开检索实验使用 `C-MTEB/T2Reranking` 的查询、正样本文档和困难负样本文档。准备脚本固定随机种子 `20260813`，尝试抽取 100 条查询并跳过没有有效正样本的记录，最终得到 99 个有效查询：

- 公开子集包含 1,591 篇去重候选文档，其中 806 篇被至少一个查询标注为相关文档。
- 官方 `positive` 作为相关文档，`negative` 作为困难负样本，不修改原始正负标签。
- 公开子集用于比较检索和排序算法，不用于训练或微调大模型。
- 原始转换数据不提交到代码仓库，数据许可和使用条件以数据集主页为准。
- 项目自带 5 份技术文档，只用于网页演示、端到端开发和拒答阈值验证。

本地挑战集共有 36 题，其中 26 题可回答、10 题不可回答，并包含 3 题跨文档问题。题型覆盖关键词、语义改写、比较、多事实、跨文档和拒答；它是开发验证集，不是公开基准。

## 数据处理与划分

公开子集按固定种子转换为独立的知识库和 JSONL 评测题，不与网页默认知识库混用。36 条本地挑战题再按可回答类别分层、按编号交替分为 18 条校准题和 18 条独立测试题：

| 数据集 | 样本数量 | 用途 |
| --- | ---: | --- |
| 公开检索子集 | 99 个查询 | 比较 Dense、BM25、Hybrid 和 Hybrid + Reranker |
| 本地校准集 | 18 题 | 选择 CrossEncoder 低相关拒答阈值 |
| 本地独立测试集 | 18 题 | 固定阈值后评估可回答召回和拒答召回 |

公开子集不参与拒答阈值选择。本地独立测试题不参与阈值搜索，避免用同一批题同时选阈值和报告效果。

| 工程问题 | 处理方式 | 输出 |
| --- | --- | --- |
| PDF、Markdown、TXT 结构不同 | PDF 按页提取，文本文件按编码读取并保留来源元数据 | LangChain `Document` 列表 |
| 长文档无法直接放入上下文 | 使用中文分隔符执行重叠窗口切分 | 文档区段数和 Chunk 数 |
| 公开数据与演示文档相互污染 | 分离 `data/public_t2reranking/` 与 `knowledge_base/project_docs/` | 两套独立语料和索引目录 |
| 重建索引时并发读写 | 新版本写完并校验后原子更新活动指针 | 最近两个可用索引版本 |
| 阈值在测试集上调优 | 校准集选择阈值，独立测试集只报告一次 | `rejection_calibration.json` |

## 数据链路

```text
PDF、Markdown、TXT 技术文档
  -> LangChain 加载并保留文件名与 PDF 页码
  -> 中文重叠窗口切分
  -> BGE 向量化并建立 FAISS 版本化索引
  -> 用户问题查询改写
  -> BGE 语义召回 + BM25 关键词召回
  -> RRF 融合和候选去重
  -> CrossEncoder 重排序
  -> 校准阈值判断是否拒答
  -> 本地 GGUF 或 OpenAI 兼容 API 基于证据生成
  -> 引用来源与回答通过 NDJSON 流式返回
  -> FastAPI 知识库问答页面
```

公开评测数据只在运行准备脚本时下载和转换。网页默认读取 `knowledge_base/project_docs/`，不会把 1,591 篇公开评测文档混入日常知识库。

## 检索与生成

Dense 检索使用 `BAAI/bge-small-zh-v1.5` 生成归一化向量，并通过 FAISS 搜索语义相近片段。BM25 对中英文混合文本分词后计算关键词相关度。Hybrid 将两路排名交给 RRF，以排名而不是原始分数完成融合，避免不同检索器的分数量纲直接相加。

CrossEncoder 使用 `BAAI/bge-reranker-base` 对“问题—候选片段”联合打分。模型不可用时，系统先退回 BGE 向量重排序；向量回退也失败时保留 RRF 顺序。拒答阈值 `0.280595` 只在 18 条校准题上选择，并固定用于另一组 18 条测试题。

生成端有两种后端：

| 后端 | 模型连接方式 | 主要用途 |
| --- | --- | --- |
| 本地模型 | llama-cpp-python 加载 GGUF | 数据不离开本机，可离线运行 |
| API 模型 | 标准库调用 OpenAI 兼容 Chat Completions | 响应更快，可按服务商切换模型 |

两种后端共用同一套检索、重排序、拒答和引用上下文。项目没有训练或微调 BGE、Reranker 和回答模型，实验重点是 RAG 检索链路、阈值校准和工程实现。

## 运行结果

以下结果来自固定的 `C-MTEB/T2Reranking` 公开子集。在 `chunk_size=300`、`chunk_overlap=50`、`Top-K=5` 下，对四种检索配置运行同一套 99 个查询：

| 配置 | Hit@5 | Recall@5 | MRR | NDCG@5 |
| --- | ---: | ---: | ---: | ---: |
| Dense | 0.9394 | 0.3766 | 0.7387 | 0.5845 |
| BM25 | 0.9596 | 0.3762 | 0.7332 | 0.5656 |
| Hybrid | 0.9596 | 0.3613 | 0.7453 | 0.5689 |
| Hybrid + Reranker | **0.9798** | **0.4193** | **0.7983** | **0.6374** |

相对 Dense，Hybrid + Reranker 的 Hit@5、Recall@5、MRR 和 NDCG@5 分别提高 4.04、4.27、5.96 和 5.29 个百分点。Hybrid 单独融合后 Recall@5 略低于 Dense，说明混合召回本身不是稳定增益来源，当前改进主要来自第二阶段重排序。

本地 36 题按类别分层后拆成校准集和独立测试集。固定校准阈值后，测试结果如下：

| 指标 | 结果 |
| --- | ---: |
| 独立测试题数 | 18 |
| 总体准确率 | 0.9444 |
| 平衡准确率 | 0.9615 |
| 可回答问题召回率 | 0.9231 |
| 拒答召回率 | 1.0000 |

| 模块 | 结果 |
| --- | --- |
| 文档解析 | 支持 PDF、Markdown、TXT；保留文件来源和 PDF 页码 |
| 索引一致性 | SHA-256 校验、版本化目录、原子切换、跨进程文件锁 |
| 检索消融 | Dense、BM25、Hybrid、Hybrid + Reranker 四组对照 |
| 模型后端 | 本地 llama.cpp/GGUF 与 OpenAI 兼容 API 共用 RAG 链路 |
| 流式问答接口 | `POST /api/chat/stream` 返回来源、回答片段和完成状态 |
| 自动化测试 | 24 项测试覆盖检索、拒答、索引、API 和引用校验 |

公开子集结果见 `reports/public_t2reranking_summary.json`，拒答校准与独立测试结果见 `reports/rejection_calibration.json`。报告只保存可复核的精简指标，详细逐题结果由评测脚本写入 `outputs/`。

## 复现

### 1. 下载数据集

公开检索实验使用 [C-MTEB/T2Reranking](https://huggingface.co/datasets/C-MTEB/T2Reranking)。安装依赖后执行：

```powershell
python scripts/prepare_public_benchmark.py
```

脚本会按固定种子下载并转换数据：

```text
data/public_t2reranking/
├── knowledge_base/                 # 转换后的候选文档
├── evaluation/
│   └── questions.jsonl             # 查询与相关文档标签
└── manifest.json                   # 数据源、种子和样本数量
```

公开数据、FAISS 索引、本地模型和运行报告不提交到代码仓库。只使用网页自带示例知识库时，可以跳过公开数据下载。

### 2. 安装依赖

建议使用 Python 3.10：

```powershell
python -m pip install -r requirements.txt
```

只有本地 GGUF 模式需要单独安装 `llama-cpp-python`。CPU 环境可执行：

```powershell
python -m pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu
```

API 模式不需要安装 `llama-cpp-python`。完整 CrossEncoder 依赖 `sentencepiece` 和 `protobuf`，已经写入 `requirements.txt`。

### 3. IDE 运行

PyCharm 或 VS Code 打开项目根目录后，选择已安装项目依赖的 Python 环境，直接运行 `start_web.py`。该文件顶部集中配置回答模型、API 地址、检索模型、离线模式、回答长度和端口，注释中写明了每项参数的用途。

也可以执行：

```powershell
python start_web.py
```

服务启动后会打开 `http://127.0.0.1:8000`。如果修改了 `PORT`，请以终端打印的地址为准。首次问答前需要在页面点击“构建索引”；已有索引时可以直接复用。

主要配置如下：

| 参数 | 默认值 | 修改后的影响 |
| --- | --- | --- |
| `LLM_PROVIDER` | `api` | 可设为 `local` 或 `api`，切换本地 GGUF 与 OpenAI 兼容接口 |
| `LOCAL_MODEL_PATH` | 空 | 本地模式填写 GGUF 文件或模型目录，可引用项目外部路径 |
| `API_BASE_URL` | OpenAI 兼容地址 | API 模式的服务地址，按实际服务商替换 |
| `API_KEY` | 空 | API 密钥，只在本机填写，禁止提交到代码仓库 |
| `API_MODEL` | 示例模型名 | API 模型名称，按服务商支持列表替换 |
| `EMBEDDING_MODEL` | 空 | 留空使用默认 BGE；也可填写已下载的本地模型目录 |
| `RERANKER_MODEL` | 空 | 留空使用默认模型或轻量回退；本地目录可启用完整 CrossEncoder |
| `MODEL_REPOSITORY_OFFLINE` | `True` | 禁止 BGE 和 Reranker 连接模型仓库，不影响大模型 API |
| `MAX_TOKENS` | `256` | 修改最大回答长度；增大后本地耗时或 API 费用通常更高 |
| `PORT` | `8000` | 修改本地页面和 API 端口 |

本地模式示例配置位于 `start_web.py`：

```python
LLM_PROVIDER = "local"
LOCAL_MODEL_PATH = r"path/to/Qwen2.5-7B-Instruct-GGUF"
EMBEDDING_MODEL = r"path/to/bge-small-zh-v1.5"
RERANKER_MODEL = r"path/to/bge-reranker-base"
MODEL_REPOSITORY_OFFLINE = True
```

API 模式需要填写 `API_BASE_URL`、`API_KEY` 和 `API_MODEL`。密钥属于本地私密配置，不要将填写密钥后的启动文件提交到 GitHub。两种模式都可以引用项目外部的模型目录，不需要复制大型模型文件。

页面提供：

- 上传 PDF、Markdown 或 TXT 文档并重新构建索引。
- 在 Dense、BM25 和 Hybrid 三种检索方式间切换。
- 开关查询改写、候选片段重排序和知识库检索。
- 调整 Top-K，查看流式回答、来源文件、页码和重排分数。

问答页面使用 `POST /api/chat/stream` 的 NDJSON 流式接口。服务先返回检索来源，再逐段返回回答；健康状态、模型后端、索引状态和 Reranker 状态可以通过 `GET /api/health` 检查。

### 4. 启动与页面排查

- 终端出现 `Uvicorn running on ...` 后，页面后端才算启动完成。回答模型、Embedding 和 Reranker 都是按需加载，首次请求通常比后续请求慢。
- 页面建库提示模型仓库连接失败时，检查 `EMBEDDING_MODEL` 是否指向完整本地目录；离线模式下不会自动下载缺失权重。
- CrossEncoder 提示不可用时，确认 Reranker 目录包含模型权重和分词器文件，并已安装 `sentencepiece`、`protobuf`；系统会自动退回 BGE 轻量重排序。
- 健康检查正常但问答失败时，先确认已经建立索引，再检查本地 GGUF 路径或 API 三项配置。
- 修改 Python、`start_web.py` 或后端路由后必须重启服务。修改前端后建议按 `Ctrl + F5` 强制刷新，避免浏览器继续使用旧脚本。
- API 模式需要网络访问模型服务；`MODEL_REPOSITORY_OFFLINE=True` 只禁止检索模型访问 Hugging Face。

### 5. 构建知识库索引

将文档放入 `knowledge_base/project_docs/`，然后在页面点击“构建索引”。也可以执行：

```powershell
python run_project.py build --embedding-model path/to/bge-small-zh-v1.5
```

默认使用 `chunk_size=500`、`chunk_overlap=80`。索引写入 `storage/faiss/versions/`，构建完成后原子更新 `CURRENT` 指针，并只保留最近两个版本。

### 6. 运行公开检索实验

先准备公开数据，再运行默认单组实验：

```powershell
python scripts/prepare_public_benchmark.py
python scripts/run_retrieval_experiments.py `
  --embedding-model path/to/bge-small-zh-v1.5 `
  --reranker-model path/to/bge-reranker-base
```

默认运行 `chunk_size=300`、`chunk_overlap=50`、`Top-K=5` 的完整检索消融。需要比较全部四组切分和 Top-K 参数时追加 `--grid`；CPU 上运行 CrossEncoder 耗时较长。

### 7. 校准拒答阈值

先对 36 条本地挑战题生成检索报告，再从报告中拆分校准集和独立测试集：

```powershell
python run_project.py evaluate --retrieval-only `
  --cases evaluation/challenge_questions.jsonl `
  --embedding-model path/to/bge-small-zh-v1.5 `
  --reranker-model path/to/bge-reranker-base `
  --report outputs/challenge_retrieval.json

python scripts/calibrate_rejection.py `
  --report outputs/challenge_retrieval.json `
  --output outputs/rejection_calibration.json
```

校准脚本只在 18 条校准题上选择平衡准确率最高的阈值，再用固定阈值计算另外 18 条题的结果。不要根据独立测试结果反复修改阈值。

### 8. 评估和启动问答服务

```powershell
# 运行测试
python -m pytest -q

# 启动 FastAPI 页面
python start_web.py
```

完整生成评测还需要提供本地 GGUF 模型，并逐题运行基础模型与 RAG 回答。公开检索对比和拒答校准不需要加载回答模型；日常使用建议直接在 `start_web.py` 中修改主要配置并从 IDE 启动。

## 项目结构

```text
local-hybrid-rag-qa-algorithm-intern/
├── start_web.py                       # 可编辑主要配置的 IDE 一键入口
├── run_project.py                     # build、ask、serve、evaluate 命令
├── rag_core.py                        # 文档、索引、检索、重排与模型后端
├── app.py                             # FastAPI 页面、上传、建库和流式接口
├── scripts/
│   ├── prepare_public_benchmark.py    # 公开数据下载与固定子集转换
│   ├── run_retrieval_experiments.py   # 检索消融和参数实验
│   └── calibrate_rejection.py         # 拒答阈值校准与独立测试
├── web/
│   ├── index.html                     # 知识库问答页面
│   ├── styles.css                     # 响应式样式
│   └── app.js                         # 上传、建库和 NDJSON 渲染
├── evaluation/
│   ├── questions.json                 # 旧版 5 题兼容集
│   └── challenge_questions.jsonl      # 36 题本地挑战集
├── knowledge_base/
│   └── project_docs/                  # 网页和端到端示例文档
├── reports/                           # 可提交的精简实验结果
├── data/public_t2reranking/           # 公开数据转换结果，不提交
├── storage/faiss/                     # 网页知识库索引，不提交
├── outputs/                           # 逐题评测和实验索引，不提交
├── tests/
│   └── test_core.py                   # 检索、索引、API 和安全测试
├── requirements.txt
└── README.md
```

公开数据、FAISS 索引、逐题报告、上传文档、Python 缓存和本地 IDE 状态由 `.gitignore` 排除。目录树用于说明职责，不表示这些本地产物需要提交。

## 输出文件

```text
storage/faiss/versions/<version>/      # 网页知识库的 FAISS 索引、元数据和校验清单
outputs/public_benchmark/*.json        # 公开子集检索消融与参数实验明细
outputs/rejection_calibration.json     # 拒答阈值、校准集和独立测试结果
reports/*.json                         # 可提交、可核对的精简实验结果
```

页面通过 `/api/health` 返回模型后端、模型加载、Reranker 和索引状态；`/api/documents/upload` 接收单个不超过 20 MB 的文档；`/api/index/build` 重建版本化索引；`/api/chat/stream` 通过 NDJSON 返回来源、回答片段和完成状态。

## 测试

```powershell
python -m pytest -q
```

当前共有 24 项测试，覆盖中文 BM25 词项、RRF 融合、弱相关拒答、CrossEncoder 回退、多相关文档指标、JSONL 题集、引用合法性、索引版本切换、文件名校验、上传限制、健康检查和流式 API。测试使用临时文档、固定向量和模拟模型，不下载公开数据，不加载 GGUF，也不会覆盖正式 FAISS 索引。

## 局限

- 公开结果来自 99 个查询的固定受控子集，不是 C-MTEB 官方全量评测或排行榜成绩。
- 本地挑战集只有 36 题和 5 份示例文档，只适合开发检查，不能代表大规模企业知识库。
- Hybrid 在当前公开子集上的 Recall@5 低于 Dense，主要增益来自 CrossEncoder 重排序，换语料后需要重新验证。
- 拒答阈值依赖当前 Reranker 和示例知识库；更换模型、切分参数或文档后应重新校准。
- PDF 仅支持可提取文本的文件，不包含 OCR、复杂表格恢复和版面分析。
- CrossEncoder 和 BGE 在 CPU 上首次加载较慢，公开子集的完整重排序实验耗时较长。
- 本地 GGUF 的速度和内存占用取决于模型规模、量化格式、上下文长度和 CPU 线程。
- API 回答需要把检索片段发送给外部服务，不适合直接处理未经授权的敏感资料。
- 项目没有微调大模型，也没有知识图谱、多模态检索和生产级权限管理。

## English Summary

Local-Hybrid-RAG-QA is a Chinese document retrieval and question-answering project for CPU-oriented local deployment. It parses PDF, Markdown, and TXT files, combines BGE/FAISS dense retrieval with BM25 through reciprocal rank fusion, reranks candidates with a CrossEncoder, and rejects low-confidence questions using a calibrated threshold. On a fixed 99-query subset derived from C-MTEB/T2Reranking, Hybrid + Reranker reaches 0.9798 Hit@5, 0.4193 Recall@5, 0.7983 MRR, and 0.6374 NDCG@5. The system supports both local GGUF inference through llama.cpp and OpenAI-compatible APIs, exposes document upload, versioned indexing, health checks, and NDJSON streaming through FastAPI, and includes 24 automated tests.

## 参考资料

- [C-MTEB](https://github.com/FlagOpen/FlagEmbedding/tree/master/C_MTEB)
- [T2Reranking](https://huggingface.co/datasets/C-MTEB/T2Reranking)
- [BGE Embedding](https://huggingface.co/BAAI/bge-small-zh-v1.5)
- [BGE Reranker](https://huggingface.co/BAAI/bge-reranker-base)
- [FAISS](https://faiss.ai/)
- [LangChain](https://python.langchain.com/)
- [FastAPI](https://fastapi.tiangolo.com/)
- [llama.cpp](https://github.com/ggml-org/llama.cpp)
- [Qwen2.5 Models](https://huggingface.co/Qwen)
