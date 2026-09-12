# ESP32 RAG Skill — Index Building and Retrieval Engine

> **Note**: AI retrieval is sourced from the `ESP32` materials folder (excluding `01.Software Code`). To ensure answer accuracy, it strictly follows the original descriptions in the source documents and cites the content sources, with no AI speculation!

## 1. Overview

This skill is a retrieval-augmented generation (RAG) system for an ESP32 documentation knowledge base. It lets an AI assistant perform semantic search and question answering across a large collection of ESP32 datasheets, technical reference manuals (TRMs), hardware design guidelines, chip errata, and other technical documents.

Skill directory structure:

```
<skill_dir>/
├── SKILL.md                                       # Skill definition (including Workflow instructions)
├── config.yaml                                    # Document classification, weights, models, SoC→Datasheet mapping, etc.
├── requirements.txt                               # Python dependency list
├── scripts/
│   ├── main.py                                    # ChromaDB RAG engine (core implementation)
│   ├── agent.py                                   # One-shot query entry point (single Bash call, avoids repeated permission prompts)
│   ├── build/
│   │   └── run.py                                 # Index build entry script (supports the --model parameter)
│   └── __init__.py
├── models/                                        # Self-contained model files (usable offline)
│   ├── dense/
│   │   ├── bge-base-en-v1.5/                      # 419MB, 768-dim, default embedding model
│   │   ├── all-MiniLM-L6-v2/                      # 88MB, 384-dim, lightweight alternative
│   │   ├── gte-base-en-v1.5/                      # GTE base 768-dim
│   │   └── embeddinggemma-300m-npu/               # Google Gemma 300M NPU-optimized
│   └── cross-encoder/
│       └── cross-encoder-ms-marco-MiniLM-L-6-v2/  # 88MB, reranking model
├── source/                                        # Raw source documents (PDF, ZIP, XLSX, DOCX, MD)
│   ├── esp32-s3_datasheet_en.pdf                  # ESP32-S3 datasheet
│   ├── esp32-s3_technical_reference_manual.pdf    # ESP32-S3 technical reference manual
│   ├── esp32-s3_hardware_design_guidelines_en.pdf # Hardware design guidelines
│   ├── esp-chip-errata-en-master-esp32s3.pdf      # Chip errata
│   ├── ...                                        # Other chip/module/dev-board materials
│   └── (organized by hardware resources, learning resources, etc.)
└── .chroma_esp32_all/                             # ChromaDB persisted vector database
    ├── chroma.sqlite3                             # ChromaDB metadata store
    ├── _bm25_cache.pkl                            # BM25 index persistence cache (~50MB)
    └── <uuid>/                                    # ChromaDB segment directory
```

## 2. Retrieval Architecture

This skill uses a three-stage retrieval path of **ChromaDB dense vector retrieval + BM25 hybrid fusion + cross-encoder reranking**:

**Pipeline**:
1. **Document extraction**: read raw PDF/ZIP/XLSX/DOCX/Markdown source files from `source/` (or an external directory specified by `RAG_DOCS_DIR`)
2. **Format conversion**: PDF→Markdown (PyMuPDF + font-size heading detection + table-aware extraction), HTML→Markdown (BeautifulSoup + Markdownify), XLSX→Markdown (openpyxl pipe tables, with blank-row and decoration-row filtering), DOCX→Markdown, native Markdown read directly (image references stripped automatically, code blocks preserved)
3. **Text cleaning**: remove footer noise, page numbers, "CONFIDENTIAL" markers, and orphan character fragments
4. **Semantic chunking**: semantic chunking based on the Markdown heading hierarchy (H1/H2 boundary segments), with smart paragraph merging/splitting and table integrity protection
5. **Vectorization**: generate 768-dim embedding vectors using the `bge-base-en-v1.5` model (switchable via config.yaml or the build-time `--model` option)
6. **Index storage**: persisted in ChromaDB, supporting cosine similarity search

## 3. Index System Design

### 3.1 ChromaDB Multi-layer Index — Collection Architecture

```
ChromaDB (.chroma_esp32_all/)
├── chunks                   # Unified chunk index (primary index, backwards compatible)
├── docs                     # Document metadata index (summary, chunk list, file_hash per document)
├── spec_summary             # Structured spec index (for cross-document comparison)
├── datasheet_chunks         # Datasheet chunks
├── trm_chunks               # Technical reference manual chunks
├── guide_chunks             # Developer guide chunks
├── api_chunks               # API reference chunks
├── safety_chunks            # Safety document chunks
├── release_chunks           # Release note chunks
├── spec_chunks              # Specification document chunks
└── docs_chunks              # General document chunks
```

### 3.2 Classification by Document Type

Documents are classified automatically by path (based on `doc_type_rules` in `config.yaml`), with first-match priority:

| Path keyword | Document type | Index collection | Search weight |
|-----------|---------|---------|---------|
| `api_reference` | api_reference | api_chunks | 3.0 |
| `guide` | guide | guide_chunks | 3.0 |
| `docs` | docs | docs_chunks | 2.0 |
| `trm` / `technical_reference` | trm | trm_chunks | 2.0 |
| `datasheet` | datasheet | datasheet_chunks | 1.0 |
| `release` | release_notes | release_chunks | 2.0 |
| `spec` | specification | spec_chunks | 2.0 |

### 3.3 Semantic Chunking Strategy

The chunking algorithm (`chunk_text()`) follows a hierarchical design:

1. **Table protection**: first protect table blocks with `<!-- TABLE -->` / `<!-- /TABLE -->` markers to prevent them from being split
2. **Heading hierarchy parsing**: extract the H1-H6 heading structure of the Markdown
3. **Parent chunks**: split the document into "parent segments" at H1/H2 boundaries
4. **Child chunks**: merge/split paragraphs within each parent segment, targeting 1200 characters with a 300-character minimum
5. **Metadata injection**: prefix each chunk with `Doc: <title> | Type: <type> | Section: <path>` so the embedding is context-aware
6. **Low-value filtering**: discard register dumps, hex-dense lines, bare "Table N" headings, and other low-information content
7. **Standalone table chunks**: feature tables in datasheets are extracted as standalone chunks with a 1.2x weight boost

### 3.4 Query Expansion

A domain-specific synonym table is defined (`query_expansions` in `config.yaml`), e.g.:
- `flash` → `['bind', 'flash', 'flashing']`
- `boot` → `['boot', 'bootloader', 'startup']`
- `errcode` → `['errcode', 'error_code', 'error code', 'error id']`
- `errata` → `['errata', 'known_issues', 'bugs']`

Queries are expanded automatically to improve recall.

### 3.5 Multi-collection Retrieval and RRF Fusion

When no document type is specified, all collections are searched and the results are merged using **Reciprocal Rank Fusion (RRF)**:

```
RRF_score(d) = Σ 1/(60 + rank(d, collection_i))
```

### 3.6 Cross-document Spec Comparison

Structured spec extraction is implemented for datasheets:

- Automatically detect the "SoC Features" table
- Extract key-value pairs for CPU, memory, peripheral interfaces, and other key specs
- Store them in a separate `spec_summary` collection
- Support queries in the form "esp32 vs esp32-s3", generating a Markdown comparison table automatically

The SoC name to datasheet number mapping is configured under `soc_to_datasheet` in `config.yaml`.

### 3.7 Three-stage Retrieval Pipeline

**Phase 1 — Dense Retrieval**: vectorize the query with `bge-base-en-v1.5` (configurable via `models.dense.default` in config.yaml or switched at build time with `--model`), search across the ChromaDB collections, fuse multi-collection results with RRF, and expand the candidate set to `n_results × 3~4`.

**Phase 2 — BM25 Hybrid Fusion**: run BM25 keyword retrieval in parallel, adding score to matching dense results; BM25-specific high-score segments are injected to avoid missing exact matches. BM25 excels at exact keyword matching and complements the semantic understanding of dense vectors; it is especially important for pattern queries that dense vectors cannot handle well, such as hexadecimal error codes, register addresses, and part numbers.
- The BM25 index is lazily rebuilt when `ChromaIndex` initializes (reading all document text from the chunks collection) and, once built, persisted to the `_bm25_cache.pkl` cache file (~50MB). With a cache, loading takes about 1 second; without a cache, the first search requires a full rebuild from ChromaDB (~12 seconds for 48k chunks), and it is persisted automatically afterward for later use.
- **Not incremental**: when the index is updated (`add_chunks()`), BM25 performs a full rebuild using an accumulated token cache (`_BM25(token_cache + new_tokens)`) rather than modifying in place. The cache file is persisted after each rebuild.
- **Cache integrity check**: on load, the number of cached IDs is compared against the actual chunk count in ChromaDB. If they differ, the cache is rebuilt automatically, avoiding retrieval gaps caused by a stale cache.
- **Upper limit protection**: when the total number of chunks exceeds 200,000, BM25 retrieval is skipped (to avoid memory exhaustion); hybrid retrieval then degrades to pure dense vector retrieval.

**Phase 3 — Cross-Encoder Reranking**: use `cross-encoder-ms-marco-MiniLM-L-6-v2` (located in `models/cross-encoder/`, switchable via `models.cross_encoder.default` in config.yaml) to score the candidate set pairwise, re-rank by relevance score, and take the Top `n_results`.

All models are cached locally with no network dependency.

### 3.8 Table-to-Text Serialization

Convert each table row into a natural-language declarative sentence for embedding, improving short-window model comprehension of dense tables:

```
Input: pipe table          →  Output: In <SoC>, CPU: 240 MHz, 2 Cores.
| CPU | 240 MHz...|   →         In <SoC>, SRAM: 512 KB.
| SRAM | 512 KB...|
```

Pure hex/register tables are skipped automatically to avoid introducing noise.

### 3.9 XLSX Blank-row and Decoration-row Filtering

`xlsx_to_markdown()` automatically filters the following noise rows when processing Excel worksheets:

- **All-empty rows**: every cell is empty or None
- **Pure decoration rows**: cells contain only `-`, `|`, spaces, or combinations thereof (e.g. `---|---|---`)

These rows are common as group separators in XLSX error-code reference tables; filtering them reduces meaningless chunks and improves retrieval precision.

## 4. Incremental Build Mechanism

### 4.1 Design Goals

There are many source documents and a full rebuild takes a long time (about 12-13 hours). Incremental builds process only added, changed, or deleted documents, avoiding unnecessary repeated work.

### 4.2 Build Pipeline (three stages)

```
Phase 1: File inventory scan (pure os.walk, <1 second)
  → list all .pdf/.zip/.xlsx/.docx/.md files
  → record path, doc_id (hash of the file path), mtime

Phase 2: Compare against the old index to determine the to-do list
  → load the existing ChromaDB and get old metadata from the docs collection
  → for each file:
    - relative path not in the old index → new (to index)
    - relative path in the index, but the MD5 content hash changed → changed (to re-index)
    - relative path in the index, MD5 identical → unchanged (skip)
  → entries present in the old index but no longer on the filesystem → delete from the index

Phase 3: Stream-process documents to index (one document at a time)
  for each pending document:
    → extract (PDF→Markdown / ZIP→Markdown / XLSX→Markdown)
    → clean text
    → semantic chunking (chunk_text)
    → embedding computation (default bge-base-en-v1.5; other models via `--model`)
    → write to the ChromaDB collections + the unified chunks collection
    → update docs metadata (including the latest file_hash)
    → free memory (del + gc.collect)
```

### 4.3 MD5 Content Hash Integrity Check

When indexing a document, `add_doc_metadata()` reads the full source file content, computes its MD5 hash, and stores it in the docs collection metadata under the `file_hash` field. The next build compares the current file's MD5 against the stored value:

```python
# Comparison logic in Phase 2
cur_hash = hashlib.md5(open(entry['source'], 'rb').read()).hexdigest()
old_hash = old_entry.get('file_hash', '')
if cur_hash and old_hash and cur_hash == old_hash:
    continue  # content unchanged, skip
```

This means:
- **New file** → indexed automatically
- **Modified file** → MD5 changes → re-indexed automatically
- **Unchanged file** → MD5 identical → skipped
- **Deleted file** → relative path no longer appears → the corresponding document and all its chunks are removed from ChromaDB

Compared with mtime, MD5 verification is more reliable — when a file is copied or a git checkout switches branches, mtime may change even though the content does not, but the MD5 comparison still holds, avoiding unnecessary rebuilds.

## 5. Querying

### 5.1 Recommended: the agent script (single call)

Searching, formatting, and source-file reading are wrapped into a single Bash call, requiring only one permission approval:

```bash
cd <skill_dir>/scripts
python3 agent.py "<query>" [--top N] [--type <doc_type>] [--raw]
```

Parameters:
- `--top N` : return the top N results (default 10)
- `--type <doc_type>` : filter by document type (datasheet, trm, guide, api, safety, release_notes, specification)
- `--raw` : JSON lines output (machine-readable)

**Output notes**: the agent outputs in human-readable form by default; each result contains the chunk text (the `text` field) plus an excerpt from the corresponding non-PDF/ZIP/XLSX source file. If the agent output already includes an excerpt, prefer it over digging into the source file directly.

Examples:
```bash
python3 agent.py "ESP32-S3 boot process" --top 5
python3 agent.py "errata error code"
python3 agent.py "esp32-s3 vs esp32"
python3 agent.py "bootrom" --raw
```

### 5.2 Using the Python API directly (requires manual multi-step operations)

```python
import pysqlite3
import sys
sys.modules['sqlite3'] = pysqlite3
import os
os.environ['RAG_CHROMA_DIR'] = '<skill_dir>/.chroma_esp32_all'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_HUB_OFFLINE'] = '1'
sys.path.insert(0, '<skill_dir>/scripts')
from main import ChromaIndex

index = ChromaIndex('<skill_dir>/.chroma_esp32_all')
results = index.search("ESP32-S3 boot process", n_results=10, hybrid=True, rerank=True)
```

## 6. Index Building and Updating

### 6.1 First Build

All commands below run from the skill root. Activate the virtual environment first:

Linux / macOS:
```bash
cd <skill_dir>
source .venv/bin/activate
```

Windows (PowerShell):
```powershell
cd <skill_dir>
.\.venv\Scripts\Activate.ps1
```

Then build with the default model (bge-base-en-v1.5):
```bash
python -m scripts.build.run
```

Build with a specified model:
```bash
python -m scripts.build.run --model gte-base-en-v1.5
```

If you prefer not to activate the venv, call its interpreter directly:

Linux / macOS:
```bash
.venv/bin/python -m scripts.build.run
```

Windows (PowerShell):
```powershell
.\.venv\Scripts\python.exe -m scripts.build.run
```

Expected duration: depends on CPU performance and corpus size (source document volume).

### 6.2 Incremental Update

With the virtual environment activated (see 6.1):
```bash
python -m scripts.build.run
```
Changes are detected automatically; only added/modified/deleted documents are processed. A model can also be specified:
```bash
python -m scripts.build.run --model all-MiniLM-L6-v2
```

### 6.3 Forced Full Rebuild

Linux / macOS:
```bash
rm -rf .chroma_esp32_all
python -m scripts.build.run
```

Windows (PowerShell):
```powershell
Remove-Item -Recurse -Force .chroma_esp32_all
python -m scripts.build.run
```

Note: if you delete only `.chroma_esp32_all/chroma.sqlite3` but keep the `.chroma_esp32_all` directory, the old `_bm25_cache.pkl` may remain and cause BM25 to return wrong results. The safe approach is to delete the entire `.chroma_esp32_all/` directory, or delete the cache manually after rebuilding:

Linux / macOS:
```bash
rm -rf .chroma_esp32_all && python -m scripts.build.run
# or:
rm -f .chroma_esp32_all/chroma.sqlite3 .chroma_esp32_all/_bm25_cache.pkl && python -m scripts.build.run
```

Windows (PowerShell):
```powershell
Remove-Item -Recurse -Force .chroma_esp32_all
python -m scripts.build.run
# or:
Remove-Item -Force .chroma_esp32_all\chroma.sqlite3, .chroma_esp32_all\_bm25_cache.pkl
python -m scripts.build.run
```

### 6.4 Building an External Corpus / Separate Database (Markdown support + directory exclusion)

By default the build targets the repository's `source/` directory. You can also point environment variables at any external document directory and write to a separate vector database, avoiding mixing with the existing `.chroma_esp32_all/` and affecting retrieval ranking.

Environment variables:

| Variable | Description | Default |
|------|------|--------|
| `RAG_DOCS_DIR` | Source document directory (recursively scanned) | `<skill_dir>/source` |
| `RAG_CHROMA_DIR` | ChromaDB persistence directory | `<skill_dir>/.chroma_esp32_all` |
| `RAG_EXCLUDE_DIRS` | Excluded paths (semicolon-separated substrings, case-insensitive) | empty |

**Supported source file formats**: `.pdf`, `.zip`, `.xlsx`, `.docx`, `.md`, `.markdown`. Native Markdown is read directly, and image references (`![alt](url)`, `![alt][ref]`) and `<img>`/`<picture>`/`<source>` tags are stripped automatically; PDF denoising/cleaning is **not** applied, so literal content such as code blocks and numeric-only lines is preserved.

**Example: index external ESP32 materials into a separate database** (excluding the SDK source tree):

Linux / macOS:
```bash
export RAG_DOCS_DIR="/path/to/esp32"
export RAG_CHROMA_DIR="<skill_dir>/.chroma_esp32_all"
export RAG_EXCLUDE_DIRS="01. Software Code"
.venv/bin/python -m scripts.build.run
```

Windows (PowerShell):
```powershell
$env:RAG_DOCS_DIR="D:\01.亚马逊\ESP32\esp\ESP32"
$env:RAG_CHROMA_DIR="D:\09.WorkSpace\esp-rag\.chroma_esp32_all"
$env:RAG_EXCLUDE_DIRS="01. Software Code"
.\.venv\Scripts\python.exe -m scripts.build.run
```

Measured result: 233 documents / 22,313 chunks (about 103 minutes).

Notes:
- `RAG_EXCLUDE_DIRS` matches relative-path substrings; separate multiple entries with `;`, e.g. `"01. Software Code;node_modules"`. It only affects the inventory scan stage; excluded files are never extracted.
- To query a separate database, point `RAG_CHROMA_DIR` at that directory as well (for example, the `RAG_CHROMA_DIR` read by `scripts/agent.py`).
- Running the same command again performs an incremental update: added/changed/deleted documents are detected automatically, with no full rebuild needed.

## 7. Configuration (config.yaml)

All document-related configuration is managed centrally in `config.yaml`:

| Config item | Description | Purpose |
|--------|------|------|
| `models` | Embedding and reranking model configuration | Specify the default model in `config.yaml`; switch at build time with `--model` |
| `soc_to_datasheet` | SoC name → Datasheet number mapping | Find the corresponding datasheet by SoC name for comparison and spec queries |
| `doc_type_rules` | Path keyword → document type rules | Automatic classification during document extraction |
| `weight_rules` | Path keyword → search weight | Affects BM25/vector retrieval ranking |
| `collection_map` | Document type → ChromaDB collection name | Multi-collection routing isolation |

### 7.1 Model Configuration Details

The full structure of the `models` config block:

```yaml
models:
  dense:                              # Bi-encoder (embedding model)
    default: bge-base-en-v1.5         # Default embedding model
    available:                        # List of available models
      - name: all-MiniLM-L6-v2
      - name: bge-base-en-v1.5
      - name: gte-base-en-v1.5
      - name: embeddinggemma-300m-npu
  cross_encoder:                      # Cross-encoder (reranking model)
    default: cross-encoder-ms-marco-MiniLM-L-6-v2
    available:
      - name: cross-encoder-ms-marco-MiniLM-L-6-v2
```

**Change the default embedding model**: set `dense.default` to any name in `available`, e.g. switch to `gte-base-en-v1.5`:
```yaml
models:
  dense:
    default: gte-base-en-v1.5
```

**Switch temporarily at build time** (without editing config.yaml):
```bash
python3 -m scripts.build.run --model all-MiniLM-L6-v2
```

**Note**: `dense` and `cross_encoder` are configured independently; the build-time `--model` parameter only affects the choice of dense embedding model, while the cross-encoder always uses the `cross_encoder.default` configuration.

To add a new SoC or adjust classification rules or weights, just edit `config.yaml` — no code changes needed.

## 8. Design Decisions Summary

| Decision | Choice | Rationale |
|------|------|------|
| Retrieval method | ChromaDB dense vectors + BM25 hybrid | Semantic understanding + exact keyword matching complement each other |
| Vector model | Default bge-base-en-v1.5 (768-dim), configurable in config.yaml or switchable with `--model` | Supports alternatives such as all-MiniLM-L6-v2, gte-base-en-v1.5, embeddinggemma-300m-npu |
| Vector database | ChromaDB | Native Python, persistent, supports metadata filtering and RRF fusion |
| Chunking strategy | Semantic hierarchical (heading-aware) | Preserves context integrity, better than fixed-window chunking |
| Table handling | Standalone chunks + metadata markers + Table-to-Text serialization | Tables are information-dense and need special protection; NL serialization improves embedding comprehension |
| Retrieval enhancement | BM25 + Dense Hybrid + Cross-Encoder Reranker | Three stages: dense semantics → BM25 exactness → cross-encoder fine ranking |
| Error-code search | Automatic BM25 boost for hex codes + forced result injection | Dense vectors cannot match hex error codes/register patterns; BM25 matches tokens exactly and then weights them |
| Cross-document comparison | Structured spec extraction + spec_summary | Datasheet feature tables have a stable structure, enabling automated comparison |
| Incremental build | MD5 content hash + deletion detection | No need to re-index unchanged documents; MD5 is more reliable than mtime |
| Build-stage separation | File inventory scan → compare → stream processing | Phase 1 is pure filesystem work (<1s); only Phase 3 does the expensive extraction + embedding |
| Memory management | Streaming, one document at a time | Avoids OOM on large corpora |
| BM25 cache | Lazy rebuild on search + pickle persistence + load-time count check | With cache ~1s; without cache first build ~12s then persisted automatically; full rebuild is not incremental; auto-degrades/skips beyond 200k chunks |
| XLSX blank-row filtering | openpyxl row-level filtering | Filters all-empty rows and pure decoration rows, cutting ~30% of invalid chunks |
| Query enhancement | Agent automatically attaches source-file excerpts + allows access to source/ files | Provides more context when chunk text is insufficient; binary formats such as PDF are skipped automatically |
| Config management | Centralized `config.yaml` | SoC mapping, document classification, weights, etc. are configurable; adding a new SoC needs no code changes |

## 9. Registering as an opencode Skill

This repository itself is a standard skill directory layout: `SKILL.md` sits at the repository root (the folder name matches `name: esp-rag` in the frontmatter), and the scripts, config, models, and source documents are all organized as in-skill resources. Once registered, opencode loads the skill automatically when needed based on its `description` and uses its `agent.py` to query the knowledge base.

### 9.1 Skill Discovery Rules

opencode's skill loader recursively scans `**/SKILL.md` in the following locations:

| Scan scope | Path |
|---------|------|
| Global skills directory | `~/.config/opencode/skills/<name>/SKILL.md` (Windows: `%USERPROFILE%\.config\opencode\skills\...`) |
| Project skills directory | `.opencode/skills/<name>/SKILL.md` |
| Explicitly registered paths | Any directory listed under `skills.paths` in `opencode.json` (recursively scanned) |

`SKILL.md` must satisfy:
- The file name must be exactly `SKILL.md`, located in a folder named after the skill
- The frontmatter must contain at least `name` (lowercase hyphenated, must match the folder name) and `description` (explaining the skill's function and when to trigger it); this repository's `description` is "Search the ESP32 documentation knowledge base (ChromaDB RAG)"

### 9.2 Option 1: Project-local registration (built into this repository)

`opencode.json` at the repository root already configures `skills.paths` to point at itself:

```json
{
  "skills": { "paths": ["."] }
}
```

So starting opencode inside the `esp-rag` directory discovers the skill automatically. To use it temporarily in another project, point that project's `opencode.json` `paths` at this repository (see 9.4).

### 9.3 Option 2: Global registration (usable from any directory)

Using a directory junction rather than copying is recommended, to avoid `.chroma_esp32_all/`, `models/`, and `source/` taking up hundreds of MB of duplicate disk space:

```powershell
# Windows (cmd): create a junction in the global skills directory pointing to this repository
cmd /c mklink /J "$env:USERPROFILE\.config\opencode\skills\esp-rag" "D:\09.WorkSpace\esp-rag"
```

If you do not need to stay in sync with the repository, you can also copy the whole directory:

```powershell
Copy-Item -Recurse "D:\09.WorkSpace\esp-rag" "$env:USERPROFILE\.config\opencode\skills\esp-rag"
```

### 9.4 Option 3: Reference via skills.paths (no copy needed)

Add the following to the global config `%USERPROFILE%\.config\opencode\opencode.json` (effective everywhere) or a project's `opencode.json` (effective only in that project):

```json
{
  "$schema": "https://opencode.ai/config.json",
  "skills": { "paths": ["D:\\09.WorkSpace\\esp-rag"] }
}
```

### 9.5 Activation and Verification

- After modifying `opencode.json`, `SKILL.md`, or skill files, you must **exit and restart opencode** — configuration is loaded only at startup and is not hot-reloaded
- After restarting, the skill should appear in the available skills list (skill name `esp-rag`); triggering it in a conversation should call `scripts/agent.py` normally to perform retrieval
