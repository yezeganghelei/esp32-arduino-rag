---
name: esp-rag
description: 搜索 ESP32 文档知识库（基于 ChromaDB RAG）
user_prompt: |
  搜索 ESP32 文档知识库（基于 ChromaDB RAG）。

  用法：
    esp-rag <搜索查询>           # 通用语义搜索
    esp-rag --spec <SoC名称>     # 查询芯片规格
    esp-rag --compare <A,B>      # 对比两款芯片

  示例：
    esp-rag esp32-s3 启动流程
    esp-rag --spec esp32-s3
    esp-rag --compare esp32-s3,esp32
---

使用 ChromaDB 驱动的 RAG 索引搜索 NVIDIA DriveOS 文档知识库。

## 工作流

1. **搜索** — agent 脚本只做纯搜索，不做意图判断。意图理解由你（LLM）负责，根据理解选用合适的参数：
   ```bash
   # 通用搜索
   python3 <skill_dir>/scripts/agent.py "<查询词>" [--top N] [--type <doc_type>]

   # 结构化规格查询（如 "Orin-N 的 GPU 参数"）
   python3 <skill_dir>/scripts/agent.py --spec "<SoC名称>"

   # 芯片对比（如 "Orin-N 和 Orin-X 的区别"）
   python3 <skill_dir>/scripts/agent.py --compare "<SoC_A>,<SoC_B>"
   ```

   参数说明：
   - `--top N` : 返回前 N 条结果（默认 10）
   - `--type <doc_type>` : 按文档类型过滤（datasheet, trm, guide, api, safety, release_notes, specification, docs）
     注意：Error ID Reference 文件的类型是 `docs`，而非 `trm`。不过错误码查询通常无需指定类型，RAG 引擎的 BM25 十六进制提升机制会自动优先匹配 Error ID Reference XLSX。
   - `--raw` : JSON lines 格式输出（机器可读）
   - `--spec <name>` : 查询 SoC 的结构化规格数据（如 "Orin-N"）
   - `--compare <A,B>` : 对比两个 SoC 的规格（如 "Orin-N,Orin-X"）

   **示例：**
   ```bash
   # 基本搜索
   python3 <skill_dir>/scripts/agent.py "DriveOS 启动流程" --top 5

   # 错误码查询 — 不指定 --type，默认搜索即可命中 Error ID Reference XLSX
   python3 <skill_dir>/scripts/agent.py "0x8213"

   # 查询规格
   python3 <skill_dir>/scripts/agent.py --spec "Orin-N"

   # 芯片对比
   python3 <skill_dir>/scripts/agent.py --compare "Orin-N,Orin-X"

   # JSON 格式输出
   python3 <skill_dir>/scripts/agent.py "bootrom" --raw
   ```

2. **回答**：语言要求中文。严格基于检索到的内容，不编造文档中未包含的信息。根据信息来源按以下原则和优先级作答：

   ### 核心原则

   1. **忠于上下文**：严格基于检索结果回答，**严禁**使用自身预训练知识补充任何事实性信息。
   2. **坦诚不知**：如果检索内容不足，按下方优先级逐步尝试；若全部不足，回复"根据现有资料，我无法回答这个问题"，**切勿**编造答案。
   3. **注明来源**：回答中每个关键事实或数据，都必须注明来源。使用 `[来源: <文档名称>-<文档标题>]` 格式标注。
   4. **处理冲突**：如果检索到的不同片段存在矛盾，指出冲突并分别说明不同来源的说法。

   ### 信息优先级

   ```
   ① agent 输出的 chunk text（`text` 字段）充分 → 直接作答，标注 chunk 来源标题
   ② chunk text 不足 → 用 agent 附带的 source excerpt（`> _Excerpt from source file:_`）
   ③ excerpt 仍不够 → 主动去 `source/` 目录下读取对应源文件，**使用相对路径**：`Read(~/.daedalus/skills/nvidia-rag/source/<相对路径>)`，不要使用 `/home/liqing/...` 形式
   ④ 源文件也找不到 → 明确告知用户无法从文档中获取答案
   ```

   **提示**：如果 chunk excerpt 信噪比低，可重新搜索（换关键词/增 `--top`）提高召回率，不要第一时间去翻源文件。

## 配置（config.yaml）

所有文档相关的配置集中在 `<skill_dir>/config.yaml`：

- **`models`** — 嵌入模型和重排序模型的默认选择及可用列表。
  - `models.dense.default`: 默认嵌入模型（`bge-base-en-v1.5`），可用构建 `--model` 参数临时切换
  - `models.cross_encoder.default`: 默认重排序模型（`cross-encoder-ms-marco-MiniLM-L-6-v2`），仅通过 config.yaml 修改
- **`soc_to_datasheet`** — SoC 名称到 Datasheet 编号的映射，用于对比查询和规格查询。
- **`doc_type_rules`** — 文档类型分类规则（通过子路径关键词匹配）。
- **`weight_rules`** — 文档权重规则，影响召回排序。
- **`collection_map`** — 文档类型到 ChromaDB 集合名称的映射。
- **`query_expansions`** — 查询同义词扩展映射表。
- **`spec_feature_keywords`** — SoC 规格表特征关键词。

新增 SoC 或调整分类规则只需修改 `config.yaml`，无需改动代码。

### 修改默认模型

编辑 `config.yaml`，更改 `models.dense.default`：
```yaml
models:
  dense:
    default: gte-base-en-v1.5    # 改为你需要的模型名
```

### 构建时临时切换（不修改 config.yaml）

```bash
python3 -m scripts.build.run --model all-MiniLM-L6-v2
```

## 搜索优化

- **混合检索**：`hybrid=True` 时，BM25 关键词检索与密集向量检索互补。对于包含十六进制错误码（如 `0x8213`）的查询，系统自动检测并为匹配的 BM25 结果加权提升，确保错误码查询优先命中 Error ID Reference XLSX。
- **查询扩展**：领域同义词自动扩展（如 `errcode` → `errcode, error_code, error id`；`mgbe` → `mgbe, multi_gigabit_ethernet, mgbe1, mgbe3` 等）。
- **过滤低价值内容**：寄存器转储、十六进制密集行、"Table N" 纯标题等低信息量内容自动过滤。
- **BM25 持久化缓存**：BM25 索引在首次搜索时从 ChromaDB 重建并缓存为 `.chroma_db/_bm25_cache.pkl`（~50MB），后续搜索直接加载缓存（~1s）。索引更新时自动增量刷新缓存。缓存加载时自动校验文档数量一致性，不一致则重建（解决缓存过期问题）。
- **XLSX 空行过滤**：XLSX 表格处理时自动过滤全空行和纯装饰行（如 `---|---|---`），减少无效分块。
- **对比查询**：自动检测 "Orin-N vs Orin-X" 这类对比查询，直接从 spec_summary 集合提取结构化规格数据进行对比。

## 重要规则

- **不要在技能正常执行期间重建 RAG 索引**。索引必须已存在。
- 所有回答必须**只**基于检索到的文档内容。
- 如果无法从文档中找到答案，明确说明。
- 访问源文件时**必须使用 `~/.daedalus/skills/nvidia-rag/source/` 相对路径**，不要使用 `/home/liqing/...` 绝对路径。
- 对于跨文档对比查询（如 "Orin-X 和 Orin-N 的区别"），**必须**使用 `--compare` 标志：`python3 <skill_dir>/scripts/agent.py --compare "Orin-N,Orin-X"`。不要做普通搜索，否则返回的是零散段落而非完整对比。
- 错误码查询（如 "ErrCode-0x10 ReptrId-0x8213 ErrAttr-0x11c"）建议提取关键十六进制值（如 `0x8213`）与模块名称（如 `mgbe1`）组合搜索，效果最佳。注意：RAG 引擎的 BM25 十六进制提升机制会自动优先匹配 Error ID Reference XLSX（类型为 `docs`），无需指定 `--type`。
- **强制重建索引后必须删除 BM25 缓存**：如果重建索引时 `.chroma_db/` 目录未清空（例如仅删除了 `chroma.sqlite3` 而未删除整个目录），BM25 缓存文件 `_bm25_cache.pkl` 可能残留。该缓存基于 chunks 集合的文档 ID 构建，若 chunks 集合内容发生变化但缓存未更新，将导致 BM25 检索返回错误结果或空结果。解决方法：
  ```bash
  rm -f <skill_dir>/.chroma_db/_bm25_cache.pkl
  ```
  首次 `search()` 调用时将自动重建。

## 依赖

Python 包（通过 `pip install -r requirements.txt` 安装）：
```bash
cd <skill_dir>
pip install -r requirements.txt
```

关键依赖：sentence-transformers, chromadb, PyMuPDF, rank-bm25, BeautifulSoup, markdownify, openpyxl。

## 目录结构

- `<skill_dir>/source/` — 原始源文档（PDF, ZIP, XLSX），按 SoC 系列和文档类型组织
- `<skill_dir>/.chroma_db/` — ChromaDB 密集向量索引（持久化，含 file_hash（MD5）跟踪 + BM25 缓存 `_bm25_cache.pkl`）
- `<skill_dir>/config.yaml` — 文档分类、权重、SoC→Datasheet 映射等配置
- `<skill_dir>/scripts/main.py` — RAG 引擎实现（索引构建、搜索、对比、十六进制提升、BM25 缓存）
- `<skill_dir>/scripts/agent.py` — 一次性查询入口（单次 Bash 调用完成搜索+输出，避免多次权限询问）
- `<skill_dir>/models/` — 本地离线模型
  - `models/dense/` — 双编码器嵌入模型（bge-base-en-v1.5 为默认，可选 all-MiniLM-L6-v2、gte-base-en-v1.5、embeddinggemma-300m-npu）
  - `models/cross-encoder/` — 交叉编码器重排序模型（cross-encoder-ms-marco-MiniLM-L-6-v2）

## 构建/重建索引

如果 `source/` 中的源文档更新了，需要重建 ChromaDB 索引：

```bash
cd <skill_dir>
rm -f .chroma_db/_bm25_cache.pkl  # 清除可能失效的 BM25 缓存
python3 -m scripts.build.run                     # 使用默认模型（bge-base-en-v1.5）
python3 -m scripts.build.run --model gte-base-en-v1.5  # 使用指定模型构建
```

这将：
1. **增量** — 按文件内容 MD5 与现有索引对比，只重新索引新增/变更/删除的文档
2. **流式** — 逐文档处理（尤其 ZIP 文件：逐个提取内部子文件 → 分块 → 向量化 → 索引，完成后立即释放内存，不会将整个 ZIP 一次性加载到内存）
3. **监控** — 每 10 分钟自动打印一次索引状态（文档数、chunk 数）

构建指定文件或目录：
```bash
python3 -m scripts.build.run <path>
```

从零开始全量重建：
```bash
rm -rf <skill_dir>/.chroma_db && python3 -m scripts.build.run