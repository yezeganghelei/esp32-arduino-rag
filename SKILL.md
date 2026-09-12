---
name: esp-rag
description: Search the ESP32 documentation knowledge base (ChromaDB RAG)
user_prompt: |
  Search the ESP32 documentation knowledge base (ChromaDB RAG).

  Usage:
    esp-rag <search query>        # general semantic search
    esp-rag --spec <SoC name>     # look up chip specifications
    esp-rag --compare <A,B>       # compare two chips

  Examples:
    esp-rag esp32-s3 boot process
    esp-rag --spec esp32-s3
    esp-rag --compare esp32-s3,esp32
---

Search the ESP32 documentation knowledge base using a ChromaDB-powered RAG index.

## Workflow

1. **Search** — the agent script only performs a raw search; it does not interpret intent. Intent understanding is your (the LLM's) responsibility; choose the appropriate parameters based on your understanding.

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
   # General search
   python <skill_dir>/scripts/agent.py "<query>" [--top N] [--type <doc_type>]

   # Structured spec lookup (e.g. "esp32-s3 GPU specs")
   python <skill_dir>/scripts/agent.py --spec "<SoC name>"

   # Chip comparison (e.g. "difference between esp32-s3 and esp32")
   python <skill_dir>/scripts/agent.py --compare "<SoC_A>,<SoC_B>"
   ```

   Parameters:
   - `--top N` : return the top N results (default 10)
   - `--type <doc_type>` : filter by document type (datasheet, trm, guide, api, safety, release_notes, specification, docs)
     Note: error-code references (e.g. chip errata) have doc type `docs`, not `trm`. However, error-code queries usually do not need a type filter; the RAG engine's BM25 hex boosting automatically prioritizes reference tables that contain error codes.
   - `--raw` : JSON lines output (machine-readable)
   - `--spec <name>` : look up structured spec data for a SoC (e.g. "esp32-s3")
   - `--compare <A,B>` : compare the specs of two SoCs (e.g. "esp32-s3,esp32")

   **Examples:**
   ```bash
   # Basic search
   python <skill_dir>/scripts/agent.py "ESP32-S3 boot process" --top 5

   # Error-code query — no --type needed; a default search hits the reference tables containing error codes
   python <skill_dir>/scripts/agent.py "0x3000"

   # Spec lookup
   python <skill_dir>/scripts/agent.py --spec "esp32-s3"

   # Chip comparison
   python <skill_dir>/scripts/agent.py --compare "esp32-s3,esp32"

   # JSON output
   python <skill_dir>/scripts/agent.py "bootrom" --raw
   ```

2. **Answer**: Answer in the same language the user used in their question, whatever it is (e.g. English question -> English answer, Chinese -> Chinese, German -> German). Base your answer strictly on the retrieved content; do not fabricate information not contained in the documents. Prioritize sources according to the following principles and order:

   ### Core principles

   1. **Stay faithful to context**: answer strictly based on retrieval results; **never** supplement factual information from your own pretrained knowledge.
   2. **Admit ignorance**: if the retrieved content is insufficient, try the fallbacks below in order; if all are insufficient, reply "Based on the available materials, I cannot answer this question" — **never** make up an answer.
      **Before concluding a fact is absent, you MUST run several re-searches with different keywords** (synonyms, the exact attribute name, the module/peripheral name, phrases like "total / count / number of", register or pin names, etc.). A single low-recall query is NOT evidence that the documents lack the fact. Never say "not in retrieved specs / not in the documents" based on one search.
   3. **Cite sources**: every key fact or figure in the answer must be attributed. Use the format `[Source: <document name>-<document title>]`.
   4. **Handle conflicts**: if retrieved passages contradict each other, point out the conflict and explain each source's claim separately.

   ### Information priority

   ```
   ① The chunk text returned by the agent (the `text` field) is sufficient -> answer directly, citing the chunk source title
   ② Chunk text insufficient -> use the source excerpt included by the agent (`> _Excerpt from source file:_`)
   ③ Excerpt still insufficient -> read the corresponding source file under the `source/` directory, **using a relative path**: `Read(<skill_dir>/source/<relative path>)`; do not use absolute paths
   ④ Source file not found either -> clearly tell the user the answer cannot be obtained from the documents
   ```

   **Tip**: if the chunk excerpt has a low signal-to-noise ratio, re-search (change keywords / increase `--top`) to improve recall; do not go straight to the source file.

## Configuration (config.yaml)

All document-related configuration is centralized in `<skill_dir>/config.yaml`:

- **`models`** — default and available embedding and reranking models.
  - `models.dense.default`: default embedding model (`bge-base-en-v1.5`); can be temporarily switched with the build `--model` parameter
  - `models.cross_encoder.default`: default reranking model (`cross-encoder-ms-marco-MiniLM-L6-v2`); change only via config.yaml
- **`soc_to_datasheet`** — SoC name to datasheet number mapping, used for comparison and spec queries.
- **`doc_type_rules`** — document type classification rules (matched by subpath keywords).
- **`weight_rules`** — document weight rules, affecting recall ranking.
- **`collection_map`** — document type to ChromaDB collection name mapping.
- **`query_expansions`** — query synonym expansion table.
- **`spec_feature_keywords`** — SoC spec table feature keywords.

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
- **BM25 persistent cache**: the BM25 index is rebuilt from ChromaDB on first search and cached as `.chroma_esp32_all/_bm25_cache.pkl` (~50MB); subsequent searches load the cache directly (~1s). The cache is refreshed incrementally when the index is updated. On load, the cache validates document-count consistency and rebuilds if inconsistent (solving the stale-cache problem).
- **XLSX blank-row filtering**: when processing XLSX tables, all-empty rows and pure decoration rows (e.g. `---|---|---`) are filtered out, reducing invalid chunks.
- **Comparison queries**: queries like "esp32-s3 vs esp32" are detected automatically, and structured spec data is extracted directly from the `spec_summary` collection for comparison.

## Important rules

- **Do not rebuild the RAG index while the skill is running normally**. The index must already exist.
- All answers must be based **only** on the retrieved document content.
- If the answer cannot be found in the documents, state so clearly.
- **Exhaust recall before declaring absence**: when a fact is missing from the first results, re-search with different keywords/phrasings before giving up. Examples: for a pin count try `"<chip> GPIO number total"`, `"how many GPIO"`, or the peripheral name; raise `--top`; try another `--type`; or query the exact register/section name. Only after several distinct queries still return nothing may you state the answer is not available in the documents.
- **Distinguish "not found by this search" from "not present in the documents"**: a fact being absent from the top-N chunks means the search missed it, not that the document lacks it. When the retrieved chunk only shows a module name without the value (e.g. a block diagram listing `GPIO`), do NOT record it as "count not in retrieved specs" — issue a dedicated query for that value first.
- **Do not contradict a previously retrieved fact**: once a value has been retrieved from the documents, reuse it consistently in later answers/tables rather than marking the field as unknown.
- When accessing source files, **always use the `<skill_dir>/source/` relative path**; do not use absolute paths.
- For cross-document comparison queries (e.g. "difference between esp32 and esp32-s3"), you **must** use the `--compare` flag: `python <skill_dir>/scripts/agent.py --compare "esp32-s3,esp32"`. Do not do a plain search, otherwise you get scattered passages instead of a complete comparison.
- For error-code queries, it is best to combine the key hexadecimal value with the module name. Note: the RAG engine's BM25 hex boosting automatically prioritizes reference tables containing error codes (type `docs`), so no `--type` is needed.
- **After a forced index rebuild, you must delete the BM25 cache**: if the `.chroma_esp32_all/` directory was not emptied during the rebuild (e.g. only `chroma.sqlite3` was deleted but not the whole directory), the BM25 cache file `_bm25_cache.pkl` may remain. That cache is built from the document IDs of the chunks collection; if the chunks collection changes but the cache is not updated, BM25 retrieval will return wrong or empty results. Solution:
  Linux / macOS:
  ```bash
  rm -f <skill_dir>/.chroma_esp32_all/_bm25_cache.pkl
  ```

  Windows (PowerShell):
  ```powershell
  Remove-Item -Force <skill_dir>\.chroma_esp32_all\_bm25_cache.pkl
  ```
  It will be rebuilt automatically on the first `search()` call.

## Dependencies

Create the virtual environment (once) and install the Python packages:

Linux / macOS:
```bash
cd <skill_dir>
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

Windows (PowerShell):
```powershell
cd <skill_dir>
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Key dependencies: sentence-transformers, chromadb, PyMuPDF, rank-bm25, BeautifulSoup, markdownify, openpyxl.

## Directory structure

- `<skill_dir>/source/` — raw source documents (PDF, ZIP, XLSX), organized by SoC series and document type
- `<skill_dir>/.chroma_esp32_all/` — ChromaDB dense vector index (persisted, with file_hash (MD5) tracking + BM25 cache `_bm25_cache.pkl`)
- `<skill_dir>/config.yaml` — document classification, weights, SoC->datasheet mapping, etc.
- `<skill_dir>/scripts/main.py` — RAG engine implementation (index build, search, comparison, hex boosting, BM25 cache)
- `<skill_dir>/scripts/agent.py` — one-shot query entry point (a single Bash call completes search + output, avoiding repeated permission prompts)
- `<skill_dir>/models/` — local offline models
  - `models/dense/` — bi-encoder embedding models (bge-base-en-v1.5 is the default; alternatives: all-MiniLM-L6-v2, gte-base-en-v1.5, embeddinggemma-300m-npu)
  - `models/cross-encoder/` — cross-encoder reranking model (cross-encoder-ms-marco-MiniLM-L6-v2)

## Build/rebuild the index

If the source documents in `source/` are updated, rebuild the ChromaDB index. Activate the virtual environment first (see Workflow), then clear the potentially stale BM25 cache and build:

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
2. **Streaming** — process one document at a time (especially ZIP files: extract each inner sub-file one by one -> chunk -> vectorize -> index, freeing memory immediately afterward, without loading the whole ZIP into memory at once)
3. **Monitoring** — print index status (document count, chunk count) automatically every 10 minutes

Build a specific file or directory (after activation):
```bash
python -m scripts.build.run <path>
```

Full rebuild from scratch:

Linux / macOS:
```bash
rm -rf <skill_dir>/.chroma_esp32_all
python -m scripts.build.run
```

Windows (PowerShell):
```powershell
Remove-Item -Recurse -Force <skill_dir>\.chroma_esp32_all
python -m scripts.build.run
```
