---
name: esp-rag
description: Ask the ESP32 documentation & source-code knowledge base (ChromaDB RAG + LLM)
user_prompt: |
  Query the ESP32 documentation & source-code knowledge base.

  Usage:
    esp-rag "<question>"        # full RAG answer (docs + code, with citations)

  Examples:
    esp-rag "ESP32-S3 的 GPIO 中断怎么配置？"
---

Query the ESP32 knowledge base using a ChromaDB-powered RAG index. The agent runs
the **SAME RAG flow as the web chat**: LLM query rewrite → retrieval (docs + code)
→ answer generation. It returns a **complete answer with `[Source: ...]` citations
and a source list** — not raw chunks.

## Workflow

1. **Run the agent (single call)** — `scripts/agent.py` performs the whole RAG
   pipeline (rewrite → retrieve → generate) using the configured LLM. It always
   returns a ready-to-present answer.

   First activate the virtual environment, or call its interpreter directly:

   Linux / macOS:
   ```bash
   source <skill_dir>/.venv/bin/activate
   # or, without activating:
   <skill_dir>/.venv/bin/python
   ```

   Windows (PowerShell):
   ```powershell
   <skill_dir>\.venv\Scripts\Activate.ps1
   # or, without activating:
   <skill_dir>\.venv\Scripts\python.exe
   ```

   Then run:
   ```bash
   # Full RAG answer (searches BOTH the documentation DB and the code DB)
   python <skill_dir>/scripts/agent.py "<question>" [--top N] [--type <doc_type>]
   ```

   Parameters:
   - `--top N` : number of chunks to retrieve (default 10)
   - `--type <doc_type>` : filter by document type (datasheet, trm, guide, api, safety, release_notes, specification, docs)
     Note: error-code references (e.g. chip errata) have doc type `docs`, not `trm`. However, error-code queries usually do not need a type filter; the RAG engine's BM25 hex boosting automatically prioritizes reference tables that contain error codes.
   - `--raw` : JSON output `{"answer": ..., "sources": [...]}` (machine-readable)
   - `--code` : query **only** the source-code database (uses `code.docs_dir` / `code.chroma_dir` from config.yaml)
   - `--docs-only` : query **only** the documentation database

   **Default behavior** (no `--code` / `--docs-only`): the RAG flow searches **both**
   databases (docs + code), then the LLM generates one answer from the combined
   context. Use `--docs-only` for pure spec/register questions where code would be
   noise, and `--code` when only source code is wanted.

   **Examples:**
   ```bash
   # General question (docs + code)
   python <skill_dir>/scripts/agent.py "ESP32-S3 的 GPIO 中断怎么配置？" --top 5

   # Error-code query — no --type needed; BM25 hex boosting prioritizes reference tables
   python <skill_dir>/scripts/agent.py "0x3000"

   # Documentation only / code only
   python <skill_dir>/scripts/agent.py --docs-only "LEDC clock sources"
   python <skill_dir>/scripts/agent.py --code "gpio_set_level"

   # JSON output
   python <skill_dir>/scripts/agent.py "bootrom" --raw
   ```

   The run also prints stage timings to stderr / `logs/agent.log`
   (`rag_api imported`, `rewrite`, `retrieve`, `judge`, `multi_search`, `query done`).
   First call after a cold start is slower (model + BM25 load).

2. **Present the answer**: Answer in the same language the user used (English
   question → English, Chinese → Chinese, German → German). The agent's answer is
   already faithful and cited — present it as-is or lightly reformat, but **do not
   add facts that are not in it**.

   If the answer states that the documents are insufficient, **re-run** with
   different keywords/phrasings before concluding the fact is absent:
   - synonyms / the exact attribute name (e.g. "number of GPIO pins total", "how many GPIO")
   - the peripheral/module name, register or pin name
   - raise `--top`, try `--docs-only` or `--code`, or a different `--type`
   Only after several distinct runs still report insufficiency may you tell the
   user the answer is not available in the documents.

## Configuration (config.yaml)

All document-related configuration is centralized in `<skill_dir>/config.yaml`:

- **`models`** — default and available embedding and reranking models.
  - `models.dense.default`: default embedding model (`bge-base-en-v1.5`); can be temporarily switched with the build `--model` parameter
  - `models.cross_encoder.default`: default reranking model (`cross-encoder-ms-marco-MiniLM-L-6-v2`); change only via config.yaml
- **`doc_type_rules`** — document type classification rules (matched by subpath keywords).
- **`weight_rules`** — document weight rules, affecting recall ranking.
- **`collection_map`** — document type to ChromaDB collection name mapping.
- **`query_expansions`** — query synonym expansion table.
- **`code`** — source-code indexing block (`enabled`, `extensions`, `exclude_dirs`, `dedup`, `docs_dir`, `chroma_dir`, …).

To add a new SoC or adjust classification rules, just edit `config.yaml` — no code changes needed.

### Change the default model

Edit `config.yaml` and change `models.dense.default`:
```yaml
models:
  dense:
    default: gte-base-en-v1.5    # change to the model name you need
```

### Switch temporarily at build time (without editing config.yaml)

```bash
python -m scripts.build.run --model all-MiniLM-L6-v2
```

## Search optimizations

- **Hybrid retrieval**: when `hybrid=True`, BM25 keyword retrieval complements dense vector retrieval. For queries containing hexadecimal error codes, the system automatically detects them and boosts matching BM25 results, ensuring error-code queries prioritize reference tables containing error codes.
- **Query expansion**: domain synonyms are expanded automatically (e.g. `errcode` -> `errcode, error_code, error id`).
- **Low-value content filtering**: register dumps, hex-dense lines, bare "Table N" headings and other low-information content are filtered out automatically.
- **BM25 persistent cache**: the BM25 index is rebuilt from ChromaDB on first search and cached as `.chroma_esp32_all/_bm25_cache.pkl`; subsequent searches load the cache directly (~1s). On load, the cache validates document-count consistency and rebuilds if inconsistent.
- **XLSX blank-row filtering**: when processing XLSX tables, all-empty rows and pure decoration rows (e.g. `---|---|---`) are filtered out, reducing invalid chunks.

## Important rules

- **Do not rebuild the RAG index while the skill is running normally**. The index must already exist.
- All answers must be based **only** on the documents (the agent's RAG answer already enforces this).
- If the answer cannot be found in the documents, state so clearly.
- **Distinguish "not found by this run" from "not present in the documents"**: a fact missing from one run means retrieval missed it, not that the document lacks it — re-search (see Workflow step 2) before declaring it absent.
- When accessing source files, use the relative path under `<skill_dir>/source/`
  (`RAG_DOCS_DIR`) or `code.docs_dir` (e.g. `source_code/`); do not use absolute paths.
- For error-code queries, combine the key hexadecimal value with the module name.
  BM25 hex boosting prioritizes reference tables containing error codes (type `docs`).
- **After a forced index rebuild, delete the BM25 cache** if the `.chroma_esp32_all/`
  directory was not fully emptied (a stale `_bm25_cache.pkl` yields wrong/empty BM25 results):
  Linux / macOS:
  ```bash
  rm -f <skill_dir>/.chroma_esp32_all/_bm25_cache.pkl
  ```
  Windows (PowerShell):
  ```powershell
  Remove-Item -Force <skill_dir>\.chroma_esp32_all\_bm25_cache.pkl
  ```
  It is rebuilt automatically on the first `search()` call.

## Dependencies & LLM

Runtime dependencies are installed from `requirements.txt`.

- Core retrieval: sentence-transformers, chromadb, PyMuPDF, rank-bm25, BeautifulSoup, markdownify, openpyxl.
- **The agent now performs LLM answer generation, so an OpenAI-compatible LLM must be configured** (loaded from `<skill_dir>/.env`):
  - `LLM_API_KEY` — required
  - `LLM_BASE_URL` — e.g. `https://api.deepseek.com/v1`
  - `LLM_MODEL` — e.g. `deepseek-chat`
  - `LLM_TEMPERATURE`, `LLM_MAX_TOKENS` (optional)
  If the LLM is not configured, the answer step fails — check `.env`.
- Windows pin note: `grpcio==1.66.2` + `protobuf==6.33.6` are pinned (grpcio 1.83.1's unsigned
  `cygrpc.pyd` is blocked by Windows Smart App Control; protobuf 7.x breaks the grpcio ABI).

Create the virtual environment (once) and install:
```powershell
cd <skill_dir>
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Directory structure

- `<skill_dir>/source/` — raw source documents (PDF, ZIP, XLSX, MD), organized by SoC series and document type (corpus pointed to via `RAG_DOCS_DIR` / `code.docs_dir`)
- `<skill_dir>/source_code/` — source-code corpus (optional, for the code DB)
- `<skill_dir>/.chroma_esp32_all/` — ChromaDB vector index for **documentation** (persisted, file_hash (MD5) tracking + BM25 cache `_bm25_cache.pkl`)
- `<skill_dir>/.chroma_esp32_code/` — ChromaDB vector index for **source code** (opt-in, see the `code:` block in config.yaml)
- `<skill_dir>/config.yaml` — document classification, weights, models, code block, etc.
- `<skill_dir>/scripts/main.py` — RAG engine (index build, search, hex boosting, BM25 cache)
- `<skill_dir>/scripts/rag_api.py` — unified retrieval interface (`search` / `run_query` / `multi_search`)
- `<skill_dir>/scripts/agent.py` — one-shot CLI entry: **full RAG answer** (same flow as web & MCP)
- `<skill_dir>/scripts/api/chat.py` — shared `rag_answer()` (used by web / MCP / CLI) + stage logging
- `<skill_dir>/scripts/mcp/server.py` — MCP Server (tools also use `rag_answer`)
- `<skill_dir>/models/` — local offline models
  - `models/dense/` — bi-encoder embedding models (bge-base-en-v1.5 default; all-MiniLM-L6-v2, gte-base-en-v1.5, embeddinggemma-300m-npu)
  - `models/cross-encoder/` — cross-encoder reranking model (cross-encoder-ms-marco-MiniLM-L-6-v2)
- `<skill_dir>/logs/` — runtime logs (`agent.log`, `mcp.log`, …)

## Build/rebuild the index

If the source documents in `source/` (or `source_code/`) are updated, rebuild the
ChromaDB index. Activate the virtual environment first (see Workflow), then clear the
potentially stale BM25 cache and build:

Linux / macOS:
```bash
cd <skill_dir>
rm -f .chroma_esp32_all/_bm25_cache.pkl
python -m scripts.build.run                      # default model (bge-base-en-v1.5)
python -m scripts.build.run --model gte-base-en-v1.5   # specified model
```

Windows (PowerShell):
```powershell
cd <skill_dir>
Remove-Item -Force .chroma_esp32_all\_bm25_cache.pkl
python -m scripts.build.run                      # default model (bge-base-en-v1.5)
python -m scripts.build.run --model gte-base-en-v1.5   # specified model
```

This will:
1. **Incremental** — compare file content MD5 against the existing index and only re-index added/changed/deleted documents
2. **Streaming** — process one document at a time (ZIP: extract each inner sub-file → chunk → vectorize → index, freeing memory immediately)
3. **Monitoring** — print index status (document count, chunk count) periodically

Build a specific file or directory:
```bash
python -m scripts.build.run <path>
```

Build the **code** index (separate DB, from `code.docs_dir`):
```powershell
$env:RAG_INDEX_CODE="1"; python -m scripts.build.run
```

Full rebuild from scratch:
```powershell
Remove-Item -Recurse -Force <skill_dir>\.chroma_esp32_all
python -m scripts.build.run
```

> Incremental rebuild requires the **same embedding model** (`bge-base-en-v1.5`) as the
> existing index; switching models requires a full rebuild.
