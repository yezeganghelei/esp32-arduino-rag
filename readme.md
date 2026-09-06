# ESP32 RAG 技能 — 索引构建与检索引擎

## 1. 概述

本技能是一个针对 ESP32 文档知识库的检索增强生成（RAG）系统。它允许 AI 助手在 ESP32 的数据手册、技术参考手册（TRM）、硬件设计指南、芯片 errata 等大量技术文档中进行语义搜索和问答。

技能目录结构：

```
<skill_dir>/
├── SKILL.md                                       # 技能定义（含 Workflow 指令）
├── config.yaml                                    # 文档分类、权重、模型、SoC→Datasheet 映射等配置
├── requirements.txt                               # Python 依赖清单
├── scripts/
│   ├── main.py                                    # ChromaDB RAG 引擎（核心实现）
│   ├── agent.py                                   # 一次性查询入口（单次 Bash 调用，避免多次权限询问）
│   ├── build/
│   │   └── run.py                                 # 索引构建入口脚本（支持 --model 参数）
│   └── __init__.py
├── models/                                        # 自包含的模型文件（离线可用）
│   ├── dense/
│   │   ├── bge-base-en-v1.5/                      # 419MB, 768维, 默认 embedding 模型
│   │   ├── all-MiniLM-L6-v2/                      # 88MB, 384维, 轻量备选
│   │   ├── gte-base-en-v1.5/                      # GTE base 768维
│   │   └── embeddinggemma-300m-npu/               # Google Gemma 300M NPU优化
│   └── cross-encoder/
│       └── cross-encoder-ms-marco-MiniLM-L-6-v2/  # 88MB, 重排序模型
├── source/                                        # 原始源文档（PDF, ZIP, XLSX）
│   ├── esp32-s3_datasheet_en.pdf                  # ESP32-S3 数据手册
│   ├── esp32-s3_technical_reference_manual.pdf    # ESP32-S3 技术参考手册
│   ├── esp32-s3_hardware_design_guidelines_en.pdf # 硬件设计指南
│   ├── esp-chip-errata-en-master-esp32s3.pdf      # 芯片 errata
│   ├── ...                                        # 其他芯片/模块/开发板资料
│   └── (按硬件资源、学习资源等目录组织)
└── .chroma_db/                                    # ChromaDB 持久化向量数据库
    ├── chroma.sqlite3                             # ChromaDB 元数据存储
    ├── _bm25_cache.pkl                            # BM25 索引持久化缓存（~50MB）
    └── <uuid>/                                    # ChromaDB segment 目录
```

## 2. 检索架构

本技能使用 **ChromaDB 密集向量检索 + BM25 混合融合 + 交叉编码器重排序** 的三段式检索路径：

**流程**：
1. **文档提取**：从 `source/` 读取 PDF/ZIP/XLSX 原始源文件
2. **格式转换**：PDF→Markdown（PyMuPDF + 字体大小标题检测 + 表格感知提取），HTML→Markdown（BeautifulSoup + Markdownify），XLSX→Markdown（openpyxl pipe 表格，含空行和装饰行过滤）
3. **文本清洗**：去除页脚噪声、页码、"CONFIDENTIAL" 标记、孤立字符碎片
4. **语义分块**：基于 Markdown 标题层次结构的语义分块（H1/H2 边界分段），段落智能合并/拆分，保护表格完整性
5. **向量化**：使用 `bge-base-en-v1.5` 模型生成 768 维嵌入向量（可通过 config.yaml 切换模型或在构建时指定 `--model`）
6. **索引存储**：ChromaDB 持久化存储，支持余弦相似度搜索

## 3. 索引体系设计

### 3.1 ChromaDB 多层索引——集合架构

```
ChromaDB (.chroma_db/)
├── chunks                   # 统一分块索引（主索引，向后兼容）
├── docs                     # 文档元数据索引（记录每个文档的摘要、分块列表、file_hash）
├── spec_summary             # 结构化规格参数索引（用于跨文档对比）
├── datasheet_chunks         # 数据手册分块
├── trm_chunks               # 技术参考手册分块
├── guide_chunks             # 开发者指南分块
├── api_chunks               # API 参考分块
├── safety_chunks            # 安全文档分块
├── release_chunks           # 发布说明分块
├── spec_chunks              # 规格文档分块
└── docs_chunks              # 通用文档分块
```

### 3.2 按文档类型分类

文档处理时会根据路径自动分类（基于 `config.yaml` 中的 `doc_type_rules`），按匹配顺序优先：

| 路径关键词 | 文档类型 | 索引集合 | 搜索权重 |
|-----------|---------|---------|---------|
| `api_reference` | api_reference | api_chunks | 3.0 |
| `guide` | guide | guide_chunks | 3.0 |
| `docs` | docs | docs_chunks | 2.0 |
| `trm` / `technical_reference` | trm | trm_chunks | 2.0 |
| `datasheet` | datasheet | datasheet_chunks | 1.0 |
| `release` | release_notes | release_chunks | 2.0 |
| `spec` | specification | spec_chunks | 2.0 |

### 3.3 语义分块策略

分块算法（`chunk_text()`）遵循层次化设计：

1. **表格保护**：先用 `<!-- TABLE -->` / `<!-- /TABLE -->` 标记保护表格块，防止被切断
2. **标题层次解析**：提取 Markdown 的 H1-H6 标题结构
3. **父分块**：以 H1/H2 为边界切分文档为"父段落"
4. **子分块**：在父段落内按段落合并/拆分，目标 1200 字符，最小 300 字符
5. **元数据注入**：每个分块前注入 `Doc: <标题> | Type: <类型> | Section: <路径>` 前缀，使向量嵌入能感知上下文
6. **低价值过滤**：丢弃寄存器转储、十六进制密集行、"Table N" 标题等低信息量内容
7. **表格独立分块**：datasheet 中的特征表格被提取为独立分块，权重提升 1.2 倍

### 3.4 查询扩展

定义了领域特定的同义词映射表（`query_expansions`，位于 `config.yaml`），如：
- `flash` → `['bind', 'flash', 'flashing']`
- `boot` → `['boot', 'bootloader', 'startup']`
- `errcode` → `['errcode', 'error_code', 'error code', 'error id']`
- `errata` → `['errata', 'known_issues', 'bugs']`

查询时自动扩展，提高召回率。

### 3.5 多集合检索与 RRF 融合

当不指定文档类型时，跨所有集合搜索并使用**倒数排序融合（RRF）** 合并结果：

```
RRF_score(d) = Σ 1/(60 + rank(d, collection_i))
```

### 3.6 跨文档规格对比

针对 datasheet 实现了结构化规格提取：

- 自动检测 "SoC Features" 表格
- 提取 CPU、内存、外设接口等关键规格的键值对
- 存储到独立的 `spec_summary` 集合
- 支持 "esp32 vs esp32-s3" 格式的查询，自动生成 Markdown 对比表格

SoC 名称到 Datasheet 编号的映射在 `config.yaml` 的 `soc_to_datasheet` 中配置。

### 3.7 三段式检索流程

**Phase 1 — Dense Retrieval（密集向量检索）**：使用 `bge-base-en-v1.5`（可通过 config.yaml 的 `models.dense.default` 配置或构建时 `--model` 切换）对 query 进行向量化，在 ChromaDB 各集合中搜索，多集合结果通过 RRF 融合，候选集扩充至 `n_results × 3~4`。

**Phase 2 — BM25 Hybrid Fusion（混合融合）**：并行执行 BM25 关键词检索，命中结果给密集搜索加分；BM25 特有高分段注入以防遗漏精确匹配。BM25 擅长精确关键词匹配，与密集向量的语义理解互补，尤其对十六进制错误码、寄存器地址、型号编号等密集向量无法有效处理的模式查询至关重要。
- BM25 索引在 `ChromaIndex` 初始化时惰性重建（从 chunks 集合读取所有文档文本），重建后持久化为 `_bm25_cache.pkl` 缓存文件（~50MB）。有缓存时加载耗时约 1 秒；无缓存时首次搜索需从 ChromaDB 全量重建（约 12 秒，48k chunks），重建后自动持久化供后续使用。
- **不是增量更新**：索引更新（`add_chunks()`）时，BM25 使用累积 token 缓存执行全量重建（`_BM25(token_cache + new_tokens)`），而非原地增量修改。每次重建后同步持久化缓存文件。
- **缓存完整性校验**：加载缓存时自动比对缓存 ID 数量与 ChromaDB 实际 chunk 数量。不一致则自动重建，避免缓存过期导致的检索遗漏。
- **上限保护**：当 chunks 总数超过 20 万时，跳过 BM25 检索（避免内存溢出），此时混合检索退化为纯密集向量检索。

**Phase 3 — Cross-Encoder Reranking（交叉编码器重排序）**：使用 `cross-encoder-ms-marco-MiniLM-L-6-v2`（位于 `models/cross-encoder/`，可通过 config.yaml 的 `models.cross_encoder.default` 配置切换）对候选集逐对打分，按相关性分数重新排序取 Top `n_results`。

所有模型均为本地缓存，无网络依赖。

### 3.8 表格→自然语言序列化 (Table-to-Text)

将表格每行转换为自然语言陈述句进行 embedding，提升 short-window 模型对密集表格的理解：

```
输入：管道表格          →  输出：In <SoC>, CPU: 240 MHz, 2 Cores.
| CPU | 240 MHz...|   →         In <SoC>, SRAM: 512 KB.
| SRAM | 512 KB...|
```

对于纯十六进制/寄存器表格自动跳过序列化，避免引入噪声。

### 3.9 XLSX 空行和装饰行过滤

`xlsx_to_markdown()` 在处理 Excel 工作表时自动过滤以下噪声行：

- **全空行**：所有单元格均为空或 None
- **纯装饰行**：单元格仅包含 `-`、`|`、空格或其组合（如 `---|---|---`）

这些行常见于 XLSX 格式错误码参考表中的分组分隔线，过滤后可减少无意义的分块，提升检索精度。

## 4. 增量构建机制

### 4.1 设计目标

源文档数量多，全量重建耗时长（约 12-13 小时）。增量构建仅处理新增、变更或删除的文档，避免不必要的重复工作。

### 4.2 构建流程（三段式）

```
Phase 1: 文件清单扫描（纯 os.walk，<1 秒）
  → 列出所有 .pdf/.zip/.xlsx 文件
  → 记录路径、doc_id（基于文件路径 hash）、mtime

Phase 2: 对比旧索引，确定待处理列表
  → 加载已有 ChromaDB，获取 docs 集合中的旧元数据
  → 对每个文件：
    - 相对路径不在旧索引中 → new（待索引）
    - 相对路径在索引中，但 MD5 内容哈希已变更 → changed（待重新索引）
    - 相对路径在索引中，MD5 一致 → unchanged（跳过）
  → 旧索引中有但文件系统中已不存在的 → 从索引中删除

Phase 3: 流式处理待索引文档（逐个文档）
  for 每个待处理文档:
    → 提取（PDF→Markdown / ZIP→Markdown / XLSX→Markdown）
    → 清洗文本
    → 语义分块（chunk_text）
    → embedding 计算（默认 bge-base-en-v1.5，可通过 `--model` 指定其他模型）
    → 写入 ChromaDB 各集合 + 统一 chunks 集合
    → 更新 docs 元数据（含最新 file_hash）
    → 释放内存（del + gc.collect）
```

### 4.3 MD5 内容哈希完整性校验

每次索引文档时，`add_doc_metadata()` 读取源文件的完整内容计算 MD5 哈希，以 `file_hash` 字段存入 ChromaDB 的 docs 集合元数据。下次构建时对比当前文件的 MD5 与存储值：

```python
# Phase 2 中对比逻辑
cur_hash = hashlib.md5(open(entry['source'], 'rb').read()).hexdigest()
old_hash = old_entry.get('file_hash', '')
if cur_hash and old_hash and cur_hash == old_hash:
    continue  # 内容未变更，跳过
```

这意味着：
- **新增文件** → 自动索引
- **修改文件** → MD5 变化 → 自动重新索引
- **未改文件** → MD5 一致 → 跳过
- **删除文件** → 相对路径不再出现 → 从 ChromaDB 中删除对应文档及其所有分块

相比 mtime，MD5 校验更可靠——文件被复制、git checkout 切换分支导致 mtime 变化但实际上内容不变时，MD5 判断仍会保持不变，避免不必要的重建。

## 5. 查询检索

### 5.1 推荐方式：agent 脚本（单次调用）

将搜索、格式化和源文件读取封装为一次 Bash 调用，只需一次权限批准：

```bash
cd <skill_dir>/scripts
python3 agent.py "<搜索词>" [--top N] [--type <doc_type>] [--raw]
```

参数说明：
- `--top N` : 返回前 N 条结果（默认 10）
- `--type <doc_type>` : 按文档类型过滤（datasheet, trm, guide, api, safety, release_notes, specification）
- `--raw` : JSON lines 格式输出（机器可读）

**输出说明**：agent 默认以人可读格式输出，每条结果包含 chunk 文本（`text` 字段）以及对应非 PDF/ZIP/XLSX 源文件的 excerpt。如果 agent 输出中已经附带了 excerpt，应优先使用而非直接去翻源文件。

示例：
```bash
python3 agent.py "ESP32-S3 启动流程" --top 5
python3 agent.py "errata 错误码"
python3 agent.py "esp32-s3 vs esp32"
python3 agent.py "bootrom" --raw
```

### 5.2 直接使用 Python API（需要手动多步操作）

```python
import pysqlite3
import sys
sys.modules['sqlite3'] = pysqlite3
import os
os.environ['RAG_CHROMA_DIR'] = '<skill_dir>/.chroma_db'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_HUB_OFFLINE'] = '1'
sys.path.insert(0, '<skill_dir>/scripts')
from main import ChromaIndex

index = ChromaIndex('<skill_dir>/.chroma_db')
results = index.search("ESP32-S3 boot process", n_results=10, hybrid=True, rerank=True)
```

## 6. 索引构建与更新

### 6.1 首次构建

使用默认模型（bge-base-en-v1.5）构建：
```bash
cd <skill_dir>
python3 -m scripts.build.run
```

使用指定模型构建：
```bash
cd <skill_dir>
python3 -m scripts.build.run --model gte-base-en-v1.5
```

预期耗时：取决于 CPU 性能与语料规模（源文档量级）。

### 6.2 增量更新

```bash
cd <skill_dir>
python3 -m scripts.build.run
```
自动检测变更，只处理新增/修改/删除的文档。也可指定模型：
```bash
python3 -m scripts.build.run --model all-MiniLM-L6-v2
```

### 6.3 强制全量重建

```bash
rm -rf .chroma_db && python3 -m scripts.build.run
```

注意：仅删除 `.chroma_db/chroma.sqlite3` 但保留 `.chroma_db` 目录时，旧 `_bm25_cache.pkl` 可能残留并导致 BM25 返回错误结果。安全做法是删除整个 `.chroma_db/` 目录，或重建后手动删除缓存：
```bash
rm -rf .chroma_db && python3 -m scripts.build.run
# 或：
rm -rf .chroma_db/chroma.sqlite3 .chroma_db/_bm25_cache.pkl && python3 -m scripts.build.run
```

## 7. 配置（config.yaml）

所有文档相关的配置集中管理在 `config.yaml`：

| 配置项 | 说明 | 用途 |
|--------|------|------|
| `models` | 嵌入模型和重排序模型配置 | `config.yaml` 中指定默认模型，构建时可通过 `--model` 切换 |
| `soc_to_datasheet` | SoC 名称 → Datasheet 编号映射 | 对比查询、规格查询时通过 SoC 名称找到对应 datasheet |
| `doc_type_rules` | 路径关键词 → 文档类型规则 | 文档提取时自动分类 |
| `weight_rules` | 路径关键词 → 搜索权重 | 影响 BM25/向量检索的排序 |
| `collection_map` | 文档类型 → ChromaDB 集合名 | 多集合路由隔离 |

### 7.1 模型配置详解

`models` 配置块的完整结构：

```yaml
models:
  dense:                              # 双编码器（嵌入模型）
    default: bge-base-en-v1.5         # 默认使用的嵌入模型
    available:                        # 可用模型列表
      - name: all-MiniLM-L6-v2
      - name: bge-base-en-v1.5
      - name: gte-base-en-v1.5
      - name: embeddinggemma-300m-npu
  cross_encoder:                      # 交叉编码器（重排序模型）
    default: cross-encoder-ms-marco-MiniLM-L-6-v2
    available:
      - name: cross-encoder-ms-marco-MiniLM-L-6-v2
```

**修改默认嵌入模型**：将 `dense.default` 改为 `available` 中的任意名称即可，例如切换到 `gte-base-en-v1.5`：
```yaml
models:
  dense:
    default: gte-base-en-v1.5
```

**构建时临时切换**（不修改 config.yaml）：
```bash
python3 -m scripts.build.run --model all-MiniLM-L6-v2
```

**注意**：`dense` 和 `cross_encoder` 是独立配置的，构建时的 `--model` 参数只影响 dense 嵌入模型的选择，交叉编码器始终使用 `cross_encoder.default` 配置。

新增 SoC、调整分类规则或权重时，只需修改 `config.yaml`，无需改动代码。

## 8. 设计决策总结

| 决策 | 选择 | 理由 |
|------|------|------|
| 检索方式 | ChromaDB 密集向量 + BM25 混合 | 语义理解 + 关键词精确匹配互补 |
| 向量模型 | 默认 bge-base-en-v1.5（768 维），可在 config.yaml 配置或 `--model` 切换 | 支持 all-MiniLM-L6-v2、gte-base-en-v1.5、embeddinggemma-300m-npu 等备选 |
| 向量数据库 | ChromaDB | 原生 Python、持久化、支持元数据过滤、RRF 融合 |
| 分块策略 | 语义层次化（标题感知） | 保持上下文完整性，优于固定窗口分块 |
| 表格处理 | 独立分块 + 元数据标记 + Table-to-Text 序列化 | 表格信息密度高需特殊保护；NL 序列化提升 embedding 理解精度 |
| 检索增强 | BM25 + Dense Hybrid + Cross-Encoder Reranker | 三段式：密集语义→BM25 精确→Cross-Encoder 精排 |
| 错误码搜索 | 十六进制代码 BM25 自动提升 + 结果强制注入 | 密集向量无法匹配十六进制错误码/寄存器模式，BM25 按 token 精确匹配后加权 |
| 跨文档对比 | 结构化规格提取 + spec_summary | datasheet 特征表格结构稳定，可自动化对比 |
| 增量构建 | MD5 内容哈希 + 删除检测 | 无需重复索引未变更文档，MD5 比 mtime 更可靠 |
| 构建阶段分离 | 文件清单扫描 → 对比 → 流式处理 | Phase 1 纯文件系统操作（<1秒），Phase 3 才做耗时提取+embedding |
| 内存管理 | 流式逐文档处理 | 避免大语料库 OOM |
| BM25 缓存 | 搜索时惰性重建 + pickle 持久化 + 加载时数量校验 | 有缓存 ~1s，无缓存首次 ~12s 重建后自动持久化；全量重建非增量更新；超过 20 万 chunk 自动降级跳过 |
| XLSX 空行过滤 | openpyxl 行级过滤 | 过滤全空行和纯装饰行，减少 ~30% 无效分块 |
| 查询增强 | agent 自动附带源文件 excerpt + 允许访问 source/ 源文件 | chunk 文本不足时提供更多上下文，PDF 等二进制格式自动跳过 |
| 配置管理 | 集中式 `config.yaml` | SoC 映射、文档分类、权重等可配置化，新增 SoC 无需改代码 |

## 9. 注册为 opencode 技能

本仓库本身就是一套标准技能目录布局：`SKILL.md` 位于仓库根目录（文件夹名即 frontmatter 中的 `name: esp-rag`），脚本、配置、模型与源文档都以技能内资源的形式组织。注册成功后，opencode 会在需要时按 `description` 自动加载该技能，并用其中的 `agent.py` 查询知识库。

### 9.1 技能发现规则

opencode 的技能加载器在以下位置递归扫描 `**/SKILL.md`：

| 扫描范围 | 路径 |
|---------|------|
| 全局技能目录 | `~/.config/opencode/skills/<name>/SKILL.md`（Windows：`%USERPROFILE%\.config\opencode\skills\...`） |
| 项目技能目录 | `.opencode/skills/<name>/SKILL.md` |
| 显式注册路径 | `opencode.json` 中 `skills.paths` 列出的任意目录（递归扫描） |

`SKILL.md` 需满足：
- 文件名固定为 `SKILL.md`，且位于以技能名命名的文件夹内
- frontmatter 至少包含 `name`（小写连字符，须与文件夹同名）与 `description`（说明技能功能及触发时机）；本仓库的 `description` 为「搜索 ESP32 文档知识库（基于 ChromaDB RAG）」

### 9.2 方式一：项目内注册（本仓库已内置）

仓库根目录的 `opencode.json` 已配置 `skills.paths` 指向自身：

```json
{
  "skills": { "paths": ["."] }
}
```

因此在 `esp-rag` 目录内启动 opencode 即可自动发现该技能。若要在其它项目内临时使用，把该项目的 `opencode.json` 中 `paths` 指向本仓库（见 9.4）。

### 9.3 方式二：全局注册（任意目录可用）

推荐用目录联接（junction）而非复制，避免 `.chroma_db/`、`models/`、`source/` 在磁盘上重复占用数百 MB：

```powershell
# Windows (cmd)：在全局技能目录创建 junction 指向本仓库
cmd /c mklink /J "$env:USERPROFILE\.config\opencode\skills\esp-rag" "D:\09.WorkSpace\esp-rag"
```

若不需要与仓库保持同步，也可直接复制整个目录：

```powershell
Copy-Item -Recurse "D:\09.WorkSpace\esp-rag" "$env:USERPROFILE\.config\opencode\skills\esp-rag"
```

### 9.4 方式三：通过 skills.paths 引用（无需复制）

在全局配置 `%USERPROFILE%\.config\opencode\opencode.json`（任意目录生效）或某项目的 `opencode.json`（仅该项目生效）中加入：

```json
{
  "$schema": "https://opencode.ai/config.json",
  "skills": { "paths": ["D:\\09.WorkSpace\\esp-rag"] }
}
```

### 9.5 生效与验证

- 修改 `opencode.json`、`SKILL.md` 或技能文件后，需**退出并重启 opencode**——配置仅在启动时加载，不会热更新
- 重启后技能应出现在可用技能列表中（技能名 `esp-rag`）；对话中触发该技能时应能正常调用 `scripts/agent.py` 完成检索