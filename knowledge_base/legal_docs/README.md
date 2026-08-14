# 法律资料导入说明

请将已经从官方渠道下载或获得授权的中国大陆中文法律资料放入本目录，再点击网页中的“重建知识库索引”。项目不会自动批量抓取受限裁判文书。

索引器支持 DOCX、PDF、Markdown 和 TXT。DOCX 会提取段落与表格文字，并根据所在的法律分类目录生成待核验元数据；旧版 `.doc` 会被跳过，请先用 Word 或 LibreOffice 转换为 DOCX、PDF 或 TXT，且不要覆盖原件。

也可以在 `data/legal_seed_urls.txt` 填写公开官方直达链接，再从 IDE 运行 `start_legal_crawler.py`。爬虫生成的 Markdown、PDF、旁车 JSON 和抓取清单只保存在本地，不会提交到 GitHub；`crawl_hashes.json` 会保留正文哈希，供再次运行时继续去重。

Markdown 文件可以在开头添加以下元数据，字段值只用于检索过滤和来源展示：

```yaml
---
doc_type: law
validity: current
jurisdiction: 全国
effective_date: 2024-01-01
expiry_date:
title: 法律文件标题
source_url: https://官方来源地址
---
```

判例资料使用 `doc_type: case`，并补充 `court`、`case_number`、`judgment_date`。爬虫自动生成的元数据带有 `metadata_review: required`，仅作为整理提示；无法核实的字段填 `unknown` 或留空，建立索引前应回到官方来源人工核验。

建议官方入口：

- 国家法律法规数据库：https://flk.npc.gov.cn/
- 人民法院案例库：https://rmfyalk.court.gov.cn/
- 中国法院网：https://www.chinacourt.org/
- 中国裁判文书网：https://wenshu.court.gov.cn/

法律回答仅用于资料检索和案件辅助分析，不构成律师意见或正式法律意见。
