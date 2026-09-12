"""
RAG pipeline for document knowledge base.

Uses ChromaDB as the primary index (replaces TF-IDF + BM25+ + FAISS).
Features semantic chunking, query expansion, metadata filtering,
and single-document incremental updates.

Usage (via CLI entry point):
  python3 -m scripts.rag build                 # Build index
  python3 -m scripts.rag build --keep-md       # Build + save Markdown to md/
  python3 -m scripts.rag query                 # Interactive query
  python3 -m scripts.rag query "your question" # One-shot query
  python3 -m scripts.rag update <path>         # Update/refresh single document
  python3 -m scripts.rag delete <path>         # Delete single document from index

Environment:
  RAG_DOCS_DIR     Source document directory (default: <project>/source)
  RAG_CHROMA_DIR   ChromaDB persist directory (default: <project>/.chroma_esp32_all)
  RAG_MD_DIR       Markdown output dir        (default: <project>/md)
  RAG_KEEP_MD=1    Save intermediate Markdown files
"""

import gc
import hashlib
import math
import os
import pickle
import re
import sys
import time
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder
from tqdm import tqdm

# ── BM25 sparse retrieval ────────────────────────────────────────────────
try:
    from rank_bm25 import BM25Okapi as _BM25
    _HAS_BM25 = True
except ImportError:
    _HAS_BM25 = False

# ── Cross-encoder reranker ────────────────────────────────────────────────
_RERANKER = None
_RERANKER_BATCH = 64


def _get_reranker():
    global _RERANKER
    if _RERANKER is None:
        model_name = _cfg.get('models', {}).get('cross_encoder', {}).get(
            'default', 'cross-encoder-ms-marco-MiniLM-L-6-v2')
        model_path = os.path.join(_MODELS_DIR, 'cross-encoder', model_name)
        if not os.path.isdir(model_path):
            raise FileNotFoundError(
                f"Cross-encoder model not found at {model_path}. "
                f"Expected directory: models/cross-encoder/{model_name}")
        _RERANKER = CrossEncoder(model_path, device='cpu')
    return _RERANKER


# ── pysqlite3 monkey-patch (ChromaDB needs sqlite3 >= 3.35) ──────────────
try:
    import pysqlite3 as _pysqlite3
    sys.modules['sqlite3'] = _pysqlite3
except ImportError:
    pass

# ── Suppress chromadb telemetry noise (must be before chromadb import) ──
os.environ.setdefault('ANONYMIZED_TELEMETRY', 'FALSE')

import chromadb
import logging
import warnings
import yaml
# Suppress ChromaDB warnings and telemetry print spam (posthog version mismatch)
logging.getLogger('chromadb').setLevel(logging.ERROR)
for _name in ['chromadb.segment.impl.metadata', 'chromadb.segment.impl.vector',
              'chromadb.telemetry', 'chromadb.telemetry.posthog',
              'chromadb.rate_limiting', 'posthog']:
    logging.getLogger(_name).setLevel(logging.CRITICAL)
warnings.filterwarnings('ignore', message='Delete of nonexisting embedding ID')
from chromadb.api.types import Documents, EmbeddingFunction, Embeddings


# ── Load config.yaml ────────────────────────────────────────────────────

_CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'config.yaml')
def _load_config():
    with open(_CONFIG_PATH, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

_cfg = _load_config()
SOC_TO_DATASHEET = _cfg.get('soc_to_datasheet', {})
DOC_TYPE_RULES = _cfg.get('doc_type_rules', [])
WEIGHT_RULES = _cfg.get('weight_rules', [])
CHUNK_SIZE = _cfg.get('chunk_size', 1200)
MIN_CHUNK = _cfg.get('min_chunk', 300)
BATCH_SIZE = _cfg.get('batch_size', 5)
_QUERY_EXPANSIONS = _cfg.get('query_expansions', {})
SPEC_FEATURE_KEYWORDS = _cfg.get('spec_feature_keywords',
    ['cpu', 'memory', 'display', 'video',
     'pcie', 'usb', 'ethernet', 'camera', 'isp', 'boot', 'security',
     'functional safety',
     'real-time', 'dmips', 'cache', 'l1', 'l2', 'l3',
     'arm cortex'])

# ── Constants ───────────────────────────────────────────────────────────

_MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models')



# ══════════════════════════════════════════════════════════════════════════════
# 1. Custom Embedding Function (uses local model)
# ══════════════════════════════════════════════════════════════════════════════

class LocalModelEmbeddingFunction(EmbeddingFunction):
    """ChromaDB embedding function using local sentence-transformers model.

    Resolves models from <skill_dir>/models/dense/<model_name>.
    All model files are self-contained within the skill directory —
    no HuggingFace download fallback.
    """

    def __init__(self, model_name: str = None):
        if model_name is None:
            model_name = _cfg.get('models', {}).get('dense', {}).get(
                'default', 'bge-base-en-v1.5')
        self._model_name = model_name
        model_path = os.path.join(_MODELS_DIR, 'dense', model_name)
        if not os.path.isdir(model_path):
            raise FileNotFoundError(
                f"Dense model not found at {model_path}. "
                f"Expected directory: models/dense/{model_name}")
        os.environ['TRANSFORMERS_OFFLINE'] = '1'
        os.environ['HF_HUB_OFFLINE'] = '1'
        self._model = SentenceTransformer(model_path, device='cpu', trust_remote_code=True)

    def __call__(self, input: Documents) -> Embeddings:
        embs = self._model.encode(
            list(input), show_progress_bar=False, normalize_embeddings=True
        )
        return embs.tolist()


# ══════════════════════════════════════════════════════════════════════════════
# 2. Document extraction (unchanged from original main.py)
# ══════════════════════════════════════════════════════════════════════════════

# ── PDF → Markdown (table-aware) ─────────────────────────────────────────────

def _is_in_table(block_bbox: Tuple[float, float, float, float],
                 table_bboxes: List[Tuple[float, float, float, float]]) -> bool:
    """Check if a text block's bbox overlaps significantly with any table bbox."""
    bx0, by0, bx1, by1 = block_bbox
    b_area = max((bx1 - bx0) * (by1 - by0), 1.0)
    for tx0, ty0, tx1, ty1 in table_bboxes:
        ox0 = max(bx0, tx0)
        oy0 = max(by0, ty0)
        ox1 = min(bx1, tx1)
        oy1 = min(by1, ty1)
        if ox0 < ox1 and oy0 < oy1:
            overlap_area = (ox1 - ox0) * (oy1 - oy0)
            if overlap_area / b_area > 0.3:
                return True
    return False


def _clean_table_cell(cell_text: Optional[str]) -> str:
    """Clean PDF table cell text: multi-pass noise removal.

    Applies repeated regex passes to strip:
    - Footer page-number digits at start/end of lines
    - Orphan single uppercase/lowercase characters from text fragments
    - Leading/trailing dash or underscore noise
    - Remaining embedded newlines → space

    Design principle: only strip chars at NEWLINE boundaries (where PDF text
    fragments leak into cells). NEVER strip chars from within inline text
    (e.g., keep "2" in "2 GPC" — that's legitimate content).
    """
    if cell_text is None:
        return ''

    text = cell_text.strip()
    if not text:
        return ''

    # === Phase 1: \n-boundary cleaning (before newline collapse) ===
    # In this phase, a single char between two \n chars is suspicious —
    # PDF often puts orphans on their own line. But a digit adjacent to
    # word chars on the same line is legitimate.
    prev = ''
    while prev != text:
        prev = text

        # 1. Strip leading digit+newline (footer page numbers at cell start)
        text = re.sub(r'^\d+\n', '', text)

        # 2. Strip trailing newline+digit (footer noise at cell end)
        text = re.sub(r'\n\d+$', '', text)

        # 3. Strip leading orphan single char + newline (e.g. "L\ntext")
        text = re.sub(r'^[A-Za-z0-9_-]\n', '', text)

        # 4. Strip trailing newline + orphan single char (e.g. "text\nt")
        text = re.sub(r'\n[A-Za-z0-9_-]$', '', text)

        # 5. Strip isolated leading dash or underscore
        text = re.sub(r'^[-_]\s*', '', text)

        # 6. Strip isolated trailing dash or underscore
        text = re.sub(r'\s*[-_]$', '', text)

        # 7. Strip a single orphan char on its own line embedded in text.
        #    This is SAFE because \n on both sides guarantees it's an isolated
        #    line — legitimate content doesn't appear as a lone char between
        #    newlines within a single cell.
        text = re.sub(r'\n[A-Za-z0-9_-]\n', '\n', text)

        text = text.strip()

    # === Phase 2: Collapse remaining newlines to space ===
    text = re.sub(r'\s+', ' ', text)

    # === Phase 3: Final pass — strip trailing orphans ONLY ===
    # At this point all content is inline. Strip any remaining trailing
    # orphan single chars that \n-boundary cleaning missed (e.g.,
    # trailing " t" from "Type 1 Hypervisor... t" where "t" is a fragment).
    # NOTE: we do NOT strip leading orphans here because a leading single
    # char without \n (e.g., "2" in "2 GPC") is legitimate content —
    # leading footer digits already have \n after them and were handled in Phase 1.
    text = re.sub(r'\s+[A-Za-z0-9_-]$', '', text)

    # 4. If text is just a single char/digit, return empty
    if len(text) <= 2 and re.match(r'^[A-Za-z0-9_-]$', text.strip()):
        return ''

    return text.strip()


def pdf_to_markdown(filepath: str, doc_type: str = 'docs') -> Tuple[Optional[str], Optional[List[dict]]]:
    """Convert PDF to Markdown with table-aware extraction.

    Uses page.find_tables() (PyMuPDF >= 1.23) to detect tables and extract
    them as structured pipe tables. Regular text uses font-size heading detection.

    Returns (markdown_text, tables_metadata) where tables_metadata is a list of
    dicts with structured row data, or (None, None) on failure.
    """
    import fitz  # PyMuPDF
    try:
        doc = fitz.open(filepath)
        pages_md = []
        all_tables: List[dict] = []
        fname = os.path.basename(filepath)

        print(f"  [{fname}] extracting {len(doc)} pages...", end=" ", flush=True)
        for page_num in range(len(doc)):
            if page_num > 0 and page_num % 100 == 0:
                print(f"{page_num}/{len(doc)}", end=" ", flush=True)

            try:
                page = doc[page_num]

                # 1. Detect tables first
                tables = page.find_tables()
                table_bboxes: List[Tuple[float, float, float, float]] = []
                page_tables_meta: List[dict] = []

                for t in tables.tables:
                    bbox = t.bbox  # (x0, y0, x1, y1)
                    table_bboxes.append(bbox)
                    rows = t.extract()

                    # Clean cells
                    cleaned_rows = []
                    for row in rows:
                        cleaned_rows.append([_clean_table_cell(c) for c in row])

                    if not cleaned_rows:
                        continue

                    header = cleaned_rows[0] if cleaned_rows else []
                    data_rows = cleaned_rows[1:] if len(cleaned_rows) > 1 else []

                    page_tables_meta.append({
                        'page': page_num,
                        'header': header,
                        'rows': data_rows,
                        'num_cols': len(header),
                        'num_rows': len(data_rows),
                    })

                    # Emit markdown pipe table
                    pipe_rows = []
                    pipe_rows.append('| ' + ' | '.join(header) + ' |')
                    pipe_rows.append('| ' + ' | '.join(['---'] * len(header)) + ' |')
                    for row in data_rows:
                        # Pad row to match header length
                        padded = row + [''] * (len(header) - len(row))
                        pipe_rows.append('| ' + ' | '.join(padded[:len(header)]) + ' |')
                    page_tables_meta[-1]['md_table'] = '\n'.join(pipe_rows)

                # 2. Extract non-table text blocks
                blocks = page.get_text('dict')['blocks']
                page_lines = []

                for block in blocks:
                    if 'lines' not in block:
                        continue
                    # Skip blocks that fall inside detected tables
                    block_bbox = block['bbox']
                    if _is_in_table(block_bbox, table_bboxes):
                        continue

                    for line in block['lines']:
                        parts = []
                        max_font_size = 0
                        is_bold = False
                        for span in line['spans']:
                            text = span['text'].strip()
                            if not text:
                                continue
                            parts.append(text)
                            max_font_size = max(max_font_size, span['size'])
                            if span['flags'] & 16:
                                is_bold = True

                        if not parts:
                            continue
                        text = " ".join(parts)

                        if max_font_size >= 18:
                            page_lines.append(f"# {text}")
                        elif max_font_size >= 14:
                            page_lines.append(f"## {text}")
                        elif max_font_size >= 12 or (is_bold and max_font_size >= 10):
                            page_lines.append(f"### {text}")
                        else:
                            page_lines.append(text)

                # 3. Insert table markers into text where tables appeared
                sorted_tables = sorted(
                    zip(table_bboxes, page_tables_meta),
                    key=lambda x: x[0][1],  # sort by y0 (top of table)
                )

                page_text = '\n'.join(page_lines)
                if sorted_tables:
                    for bbox, tmeta in sorted_tables:
                        table_marker = f"\n\n<!-- TABLE -->\n{tmeta['md_table']}\n<!-- /TABLE -->\n\n"
                        page_text += table_marker

                pages_md.append(page_text)
                all_tables.extend(page_tables_meta)
            except Exception as e:
                print(f"\n  [SKIP] Page {page_num}: {e}", flush=True)
                pages_md.append('')
                continue

        doc.close()
        result = "\n\n".join(pages_md)
        return (result if result.strip() else None,
                all_tables if all_tables else None)
    except Exception as e:
        print(f"  [SKIP] PDF error {filepath}: {e}")
        return None, None


# ── HTML → Markdown ──────────────────────────────────────────────────────

def html_to_markdown(html_content: str) -> str:
    """Convert HTML to clean Markdown, stripping nav/sidebar noise."""
    from bs4 import BeautifulSoup
    from markdownify import markdownify as md

    soup = BeautifulSoup(html_content, 'html.parser')

    content_candidates = [
        soup.select_one('#wh_topic_body'),
        soup.select_one('main'),
        soup.select_one('article'),
        soup.select_one('[role="main"]'),
        soup.find('body'),
    ]
    content_elem = next((c for c in content_candidates if c is not None), soup)

    for selector in ['.wh_search_input', '.breadcrumb', 'nav', '.nav',
                     '.sidebar', '.footer', 'footer', '.d-print-none']:
        for elem in content_elem.select(selector):
            elem.decompose()

    md_text = md(
        str(content_elem),
        heading_style='ATX',
        strip=['img', 'a'],
    )
    return md_text.strip()


# ── Native Markdown ────────────────────────────────────────────────────────

_MD_IMAGE_RE = re.compile(
    r'!\[[^\]]*\]\([^)]*\)'
    r'|!\[[^\]]*\]\[[^\]]*\]'
    r'|<img\b[^>]*>'
    r'|<picture\b[^>]*>|</picture\s*>'
    r'|<source\b[^>]*>',
    re.IGNORECASE,
)


def md_to_markdown(filepath: str) -> Optional[str]:
    """Read a native Markdown file verbatim.

    Image references (Markdown / HTML) are stripped so that picture assets
    are not indexed. No PDF-style cleaning is applied so that code blocks
    and literal content (e.g. numeric-only lines) are preserved.
    """
    try:
        with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
            text = f.read()
        text = _MD_IMAGE_RE.sub('', text).strip()
        return text if text else None
    except Exception as e:
        print(f"  [SKIP] MD error {filepath}: {e}")
        return None


# ── ZIP extraction ─────────────────────────────────────────────────────────

def extract_zip_markdown(filepath: str) -> Optional[str]:
    """Extract text from ZIP: convert HTML to MD, keep .txt/.md as-is."""
    _BINARY_EXTS = {'.jar', '.exe', '.dll', '.so', '.png', '.jpg', '.jpeg',
                    '.gif', '.bmp', '.class', '.pyc', '.o', '.a', '.lib',
                    '.ttf', '.woff', '.eot', '.zip', '.tar', '.gz'}
    texts = []
    processed = 0
    skipped = 0
    try:
        with zipfile.ZipFile(filepath, 'r') as zf:
            infos = [info for info in zf.infolist() if not info.is_dir()]
            total = len(infos)

            doc_exts = {'.html', '.htm', '.xhtml', '.txt', '.md', '.docx'}
            doc_count = sum(1 for info in infos
                            if Path(info.filename).suffix.lower() in doc_exts)
            doc_size = sum(info.file_size for info in infos
                           if Path(info.filename).suffix.lower() in doc_exts)
            if total > 100 and doc_count / total < 0.1:
                print(f"  [SKIP] ZIP appears to be source/firmware code "
                      f"({total} files, only {doc_count} docs)")
                return None
            if doc_size > 500 * 1024 * 1024:
                print(f"  [SKIP] ZIP doc content too large "
                      f"({doc_size/1024/1024:.0f}MB), would cause memory pressure")
                return None

            for info in infos:
                ext = Path(info.filename).suffix.lower()
                if ext in _BINARY_EXTS:
                    skipped += 1
                    continue
                if ext in ('.html', '.htm', '.xhtml'):
                    try:
                        raw = zf.read(info.filename)
                        content = raw.decode('utf-8', errors='replace')
                        md_text = html_to_markdown(content)
                        if md_text:
                            header = f"### Source: {info.filename}\n\n"
                            texts.append(header + md_text)
                        processed += 1
                    except Exception:
                        pass
                elif ext == '.docx':
                    try:
                        md_text = docx_bytes_to_markdown(zf.read(info.filename))
                        if md_text:
                            header = f"### Source: {info.filename}\n\n"
                            texts.append(header + md_text)
                        processed += 1
                    except Exception:
                        pass
                elif ext in ('.txt', '.md'):
                    try:
                        raw = zf.read(info.filename)
                        text = raw.decode('utf-8', errors='replace').strip()
                        if text:
                            header = f"### Source: {info.filename}\n\n"
                            texts.append(header + text)
                        processed += 1
                    except Exception:
                        pass
                else:
                    skipped += 1
    except Exception as e:
        print(f"  [SKIP] ZIP error {filepath}: {e}")
        return None

    if not texts:
        return None
    return "\n\n---\n\n".join(texts) if texts else None


# ── XLSX → Markdown ─────────────────────────────────────────────────────────

def xlsx_to_markdown(filepath: str) -> Optional[str]:
    try:
        import openpyxl
    except ImportError:
        print("  [SKIP] openpyxl not installed, skipping XLSX")
        return None
    try:
        wb = openpyxl.load_workbook(filepath, read_only=True, data_only=True)
        sheets_md = []
        for sheet_name in wb.sheetnames:
            ws = wb[sheet_name]
            rows = list(ws.iter_rows(values_only=True))
            if not rows:
                continue
            # Filter empty rows and pure-separator rows before processing
            filtered_rows = []
            for row in rows:
                values = [str(c) if c is not None else '' for c in row]
                stripped = [v.strip() for v in values]
                # Skip rows where all cells are empty, pure '|', or pure '---'
                if all(v == '' or v == '|' or v == '---' or set(v).issubset({'-', '|', ' '}) for v in stripped):
                    continue
                filtered_rows.append(values)
            if not filtered_rows:
                continue

            header = filtered_rows[0]
            table_lines = []
            table_lines.append('| ' + ' | '.join(header) + ' |')
            table_lines.append('| ' + ' | '.join('---' for _ in header) + ' |')
            # Skip rows that look like section headers (all bold caps, no real data)
            for row in filtered_rows[1:]:
                # Skip rows that are just sub-headers or empty after header (already filtered above)
                table_lines.append('| ' + ' | '.join(row) + ' |')
            sheet_md = f"## {sheet_name}\n\n" + '\n'.join(table_lines)
            sheets_md.append(sheet_md)
        wb.close()
        result = '\n\n'.join(sheets_md)
        return result if result.strip() else None
    except Exception as e:
        print(f"  [SKIP] XLSX error {filepath}: {e}")
        return None


# ── DOCX → Markdown ─────────────────────────────────────────────────────────

def _docx_xml_to_markdown(xml_bytes: bytes) -> Optional[str]:
    """Convert the raw word/document.xml of a DOCX into Markdown."""
    import re
    from xml.etree import ElementTree as ET
    W = '{http://schemas.openxmlformats.org/wordprocessingml/2006/main}'
    try:
        root = ET.fromstring(xml_bytes)
    except Exception:
        return None

    def _para_text(p) -> str:
        return ''.join(t.text or '' for t in p.iter(W + 't'))

    body = root.find(W + 'body')
    if body is None:
        return None

    lines = []
    for el in body:
        tag = el.tag
        if tag == W + 'p':
            style = el.find(W + 'pPr/' + W + 'pStyle')
            name = style.get(W + 'val') if style is not None else ''
            txt = _para_text(el).strip()
            if not txt:
                continue
            if name.startswith('Heading') or name in ('1', '2', '3'):
                level = ''.join(ch for ch in name if ch.isdigit()) or '1'
                lines.append('#' * int(level) + ' ' + txt)
            else:
                lines.append(txt)
        elif tag == W + 'tbl':
            for tr in el.findall(W + 'tr'):
                cells = [' '.join(_para_text(p).strip() for p in tc.findall(W + 'p'))
                         for tc in tr.findall(W + 'tc')]
                if any(cells):
                    lines.append('| ' + ' | '.join(cells) + ' |')
            lines.append('')  # blank line after each table

    text = '\n'.join(lines).strip()
    return text if text else None


def docx_to_markdown(filepath: str) -> Optional[str]:
    """Extract a standalone DOCX file to Markdown (stdlib only)."""
    import zipfile
    try:
        with zipfile.ZipFile(filepath, 'r') as zf:
            xml_bytes = zf.read('word/document.xml')
    except Exception as e:
        print(f"  [SKIP] DOCX error {filepath}: {e}")
        return None
    return _docx_xml_to_markdown(xml_bytes)


def docx_bytes_to_markdown(raw: bytes) -> Optional[str]:
    """Extract DOCX content from raw bytes (e.g. a member of a larger ZIP)."""
    import io
    import zipfile
    try:
        with zipfile.ZipFile(io.BytesIO(raw)) as zf:
            xml_bytes = zf.read('word/document.xml')
    except Exception:
        return None
    return _docx_xml_to_markdown(xml_bytes)


# ── Low-value content filtering ─────────────────────────────────────────────

_LOW_VALUE_PATTERNS = [
    re.compile(r'^0x[0-9a-fA-F]+\s+0x[0-9a-fA-F]+\s+\S+'),  # hex addr range + name
    re.compile(r'^Bit\s+\d+[:\d]*'),                          # Bit field headers
    re.compile(r'^Offset:\s*0x[0-9a-fA-F]+'),                 # Register offset
    re.compile(r'^Reset:\s*0x[0-9a-fA-F]+'),                  # Reset value
    re.compile(r'^\d+:\d+\s+\d+:\d+'),                        # Bit ranges (31:24 15:8 pattern)
    re.compile(r'^(Field|Reset|Type|Register)\s+'),           # Register description headers
    re.compile(r'^Table\s+\d+'),                              # Table headers without content
]

_LOW_VALUE_REGISTER_RATIO = float(os.environ.get('RAG_FILTER_RATIO', '0.3'))
_LOW_VALUE_HEX_RATIO = float(os.environ.get('RAG_FILTER_HEX_RATIO', '0.4'))


def _is_low_value(text: str) -> bool:
    """Detect if a text chunk is low-value register dump content."""
    lines = text.strip().split('\n')
    if not lines:
        return False

    # Don't filter chunks with markdown headings (they're section starts)
    if any(l.strip().startswith('#') for l in lines):
        return False

    # Don't filter table chunks (pipe table rows)
    if any(l.strip().startswith('|') and l.strip().endswith('|') for l in lines):
        return False

    register_lines = 0
    hex_dense_lines = 0
    total_meaningful = 0

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        total_meaningful += 1

        for pat in _LOW_VALUE_PATTERNS:
            if pat.match(stripped):
                register_lines += 1
                break

        hex_chars = sum(1 for c in stripped if c in '0123456789abcdefABCDEFxX')
        if len(stripped) > 10 and hex_chars / max(len(stripped), 1) > _LOW_VALUE_HEX_RATIO:
            hex_dense_lines += 1

    if total_meaningful == 0:
        return False

    register_ratio = register_lines / total_meaningful
    hex_ratio = hex_dense_lines / total_meaningful
    return register_ratio > _LOW_VALUE_REGISTER_RATIO or hex_ratio > _LOW_VALUE_REGISTER_RATIO


# ── Text cleaning ──────────────────────────────────────────────────────────

def _is_table_row(line: str) -> bool:
    stripped = line.strip()
    return stripped.startswith('|') and stripped.endswith('|') and stripped.count('|') >= 3


def clean_extracted_text(text: str) -> str:
    """Remove common PDF extraction noise from markdown text."""
    lines = text.split('\n')
    cleaned = []
    for line in lines:
        stripped = line.strip()
        if _is_table_row(stripped):
            cleaned.append(line)
            continue
        if re.match(r'^#?\s*CONFIDENTIAL\s*$', stripped):
            continue
        # Strip document classification / internal-use-only markers
        # that may appear as trailing artifacts in extracted text
        line = re.sub(
            r'\b(?:CONFIDENTIAL|INTERNAL\s*(?:USE\s+)?ONLY|PROPRIETARY|RESTRICTED)\s*\S*\b',
            '', line, flags=re.IGNORECASE).strip()
        if not line:
            continue
        if re.match(r'^\|?\s*\d+\s*\|?\s*$', stripped):
            continue
        if re.match(r'^Page\s+\d+\s+of\s+\d+$', stripped, re.IGNORECASE):
            continue
        if re.match(r'^[A-Z]{2,3}-\d{5,}(-\d{3})?(_v[\d.]+)?\s*(\||\d+|\s)*$', stripped):
            continue
        cleaned.append(line)
    return '\n'.join(cleaned)


# ── Document discovery & classification ────────────────────────────────────

def _classify_weight(relpath: str) -> float:
    path_lower = relpath.lower()
    for rule in WEIGHT_RULES:
        kw = rule.get('contains')
        if kw is None or kw in path_lower:
            return rule['weight']
    return 1.0


def _classify_doc_type(relpath: str) -> str:
    path_lower = relpath.lower()
    for rule in DOC_TYPE_RULES:
        kw = rule.get('contains')
        if kw is None or kw in path_lower:
            return rule['type']
    return 'docs'


def get_documents(base_dir: str):
    """Walk base_dir, collect PDFs, ZIPs, XLSXs, DOCXs and Markdown files,
    extract and clean Markdown.

    Generator: yields one document at a time to control memory pressure.
    """
    for root, _dirs, files in os.walk(base_dir):
        for fname in sorted(files):
            fpath = os.path.join(root, fname)
            ext = Path(fname).suffix.lower()
            tables_meta = None
            try:
                if ext == '.pdf':
                    text, tables_meta = pdf_to_markdown(fpath, doc_type=_classify_doc_type(
                        os.path.relpath(fpath, base_dir)))
                elif ext == '.zip':
                    text = extract_zip_markdown(fpath)
                elif ext == '.xlsx':
                    text = xlsx_to_markdown(fpath)
                elif ext == '.docx':
                    text = docx_to_markdown(fpath)
                elif ext in ('.md', '.markdown'):
                    text = md_to_markdown(fpath)
                else:
                    continue
            except Exception as e:
                print(f"  [SKIP] Error processing {fpath}: {e}")
                del text
                continue
            if text:
                if ext not in ('.md', '.markdown'):
                    text = clean_extracted_text(text)
                if not text.strip():
                    continue
                relpath = os.path.relpath(fpath, base_dir)
                doc_id = hashlib.md5(fpath.encode()).hexdigest()[:12]
                doc_type = _classify_doc_type(relpath)
                weight = _classify_weight(relpath)
                yield {
                    'id': doc_id,
                    'path': relpath,
                    'source': fpath,
                    'text': text,
                    'tables_meta': tables_meta,
                    'weight': weight,
                    'doc_type': doc_type,
                    'title': Path(fname).stem,
                }
                # free text as soon as the caller processes/receives the yielded doc
                del text


# ══════════════════════════════════════════════════════════════════════════════
# 3. Semantic Chunking (unchanged from original main.py)
# ══════════════════════════════════════════════════════════════════════════════

def _extract_sections(text: str) -> List[dict]:
    """Extract section hierarchy from markdown text with line ranges."""
    lines = text.split('\n')
    sections = [{'level': 0, 'heading': 'ROOT', 'line': -1, 'start': 0}]
    for i, line in enumerate(lines):
        m = re.match(r'^(#{1,6})\s+(.+)$', line.strip())
        if m:
            level = len(m.group(1))
            heading = m.group(2).strip()
            sections.append({'level': level, 'heading': heading, 'line': i, 'start': i})
            sections[-2]['end'] = i
    sections[-1]['end'] = len(lines)
    return sections


def _section_path_for_line(sections: List[dict], line_num: int) -> List[str]:
    path = []
    for sec in sections:
        if sec.get('start', 0) <= line_num < sec.get('end', float('inf')):
            if sec['level'] > 0:
                path.append(sec['heading'])
    return path


def _doc_title_from_text(text: str) -> str:
    m = re.search(r'^# (.+)$', text, re.MULTILINE)
    if m:
        return m.group(1).strip()
    return ''


def _protect_tables(lines: List[str]) -> List[str]:
    """Merge table lines and <!-- TABLE --> markers into single blocks."""
    merged = []
    i = 0
    while i < len(lines):
        if lines[i].strip().startswith('<!-- TABLE -->'):
            # Collect entire table block (marker + pipe table + closing marker)
            table_block = [lines[i]]
            i += 1
            while i < len(lines) and not lines[i].strip().startswith('<!-- /TABLE -->'):
                table_block.append(lines[i])
                i += 1
            if i < len(lines):
                table_block.append(lines[i])
                i += 1
            merged.append('\n'.join(table_block))
        elif _is_table_row(lines[i]):
            table_block = [lines[i]]
            i += 1
            while i < len(lines) and _is_table_row(lines[i]):
                table_block.append(lines[i])
                i += 1
            merged.append('\n'.join(table_block))
        else:
            merged.append(lines[i])
            i += 1
    return merged


def _split_paragraphs(
    text: str, target_size: int, min_size: int
) -> List[str]:
    """Split text into paragraphs, merging small ones and splitting large ones.

    Simple O(n) algorithm:
    1. Split by double-newline to get paragraphs
    2. Merge paragraphs below min_size with neighbors
    3. Split paragraphs above target_size at sentence boundaries
    """
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    if not paragraphs:
        return [text] if text.strip() else []

    merged = []
    buf = []

    def _flush():
        if buf:
            merged.append('\n\n'.join(buf))
            buf.clear()

    for p in paragraphs:
        # Heading: always starts a new group
        if re.match(r'^#{1,6}\s+', p):
            _flush()
            buf.append(p)
            continue

        if not buf:
            buf.append(p)
        else:
            current_len = sum(len(x) for x in buf) + (len(buf) - 1) * 2  # \n\n
            if current_len + len(p) <= target_size:
                buf.append(p)
            else:
                _flush()
                buf.append(p)

    _flush()

    # Split oversized groups at sentence boundaries
    result = []
    for group in merged:
        if len(group) <= target_size:
            result.append(group)
        else:
            # Try to split at periods or newlines
            sentences = re.split(r'(?<=[.!?])\s+', group)
            buf = []
            for s in sentences:
                buf.append(s)
                if sum(len(x) for x in buf) >= target_size:
                    result.append(' '.join(buf))
                    buf = []
            if buf:
                result.append(' '.join(buf))

    return result


def _build_metadata_prefix(doc_title: str, doc_type: str, section_path: List[str]) -> str:
    """Build metadata prefix that embedding can reference."""
    parts = []
    if doc_title:
        parts.append(f"Doc: {doc_title}")
    if doc_type:
        parts.append(f"Type: {doc_type}")
    if section_path:
        path_str = " > ".join(section_path)
        parts.append(f"Section: {path_str}")
    if parts:
        return " | ".join(parts) + "\n\n"
    return ""


# ✨ NEW: Table-to-Text serialization — converts a structured table into a
# list of natural language sentences, one per row.  This helps short-window
# embedding models understand dense tabular data.
def _table_to_natural_language(header: List[str], rows: List[List[str]],
                                doc_title: str = '') -> str:
    """Convert a structured table (header + rows) to natural language sentences.

    Each row becomes one sentence like:
      "In <SoC>, CPU: 240 MHz Xtensa Dual-Core, SRAM: 512 KB."

    Falls back to an empty string if the table looks like a register dump
    (header full of hex-like names).
    """
    if not header or not rows:
        return ''

    # Skip pure register/hex tables
    hex_like = sum(1 for h in header if re.match(r'^(0x)?[0-9a-fA-F]+$', h.strip()))
    if hex_like > len(header) // 2:
        return ''

    doc_prefix = f"In {doc_title}, " if doc_title else ""
    sentences = []

    for row in rows:
        parts = []
        for hi, cell in enumerate(row):
            cell = cell.strip()
            if not cell:
                continue
            h = header[hi].strip() if hi < len(header) else f"col{hi}"
            # Skip empty headers or pure hex values
            if not h or re.match(r'^(0x)?[0-9a-fA-F]+$', cell):
                continue
            # Clean orphan characters from PDF extraction
            cell_clean = re.sub(r'\s+[A-Za-z0-9_-]\s*$', '', cell).strip()
            if not cell_clean:
                continue
            parts.append(f"{h}: {cell_clean}")
        if parts:
            sentences.append(f"{doc_prefix}{'; '.join(parts)}.")

    return '\n'.join(sentences) if sentences else ''


def chunk_text(
    text: str,
    doc_id: str,
    source: str,
    doc_type: str = 'docs',
    doc_title: str = '',
    chunk_size: int = CHUNK_SIZE,
    min_chunk: int = MIN_CHUNK,
    weight: float = 1.0,
    tables_meta: Optional[List[dict]] = None,
) -> List[dict]:
    """Semantic heading-aware chunking.

    Returns list of chunk dicts, each with 'id', 'doc_id', 'source', 'text',
    'doc_type', 'doc_title', 'section_path', 'weight'.

    If tables_meta is provided (from table-aware PDF extraction), each table
    is also yielded as a standalone chunk with its structured data preserved
    in the chunk metadata under 'table_data'.
    """
    child_chunks: List[dict] = []

    # ── Emit structured table chunks ────────────────────────────────────
    if tables_meta:
        for tbl in tables_meta:
            header = tbl.get('header', [])
            rows = tbl.get('rows', [])
            if not header or not rows:
                continue
            # Build a compact text representation
            md_table = tbl.get('md_table', '')
            table_text = md_table if md_table else (
                '| ' + ' | '.join(header) + ' |\n'
                '| ' + ' | '.join(['---'] * len(header)) + ' |\n' +
                '\n'.join('| ' + ' | '.join(
                    row + [''] * (len(header) - len(row))
                )[:len(header)] + ' |' for row in rows)
            )
            # ✨ NEW: Table-to-Text serialization — convert each row to a
            # natural language sentence for better embedding comprehension.
            table_nl = _table_to_natural_language(header, rows, doc_title)
            cid = hashlib.md5(f"{doc_id}:tbl:{tbl.get('page', 0)}:{len(child_chunks)}".encode()).hexdigest()[:12]
            enriched = f"### Table (page {tbl.get('page', '?')})\n\n{table_text}"
            if doc_title:
                enriched = f"Doc: {doc_title} | Type: {doc_type}\n\n{enriched}"
            # NL version: same metadata prefix but with natural language text
            enriched_nl = f"{_build_metadata_prefix(doc_title, doc_type, [])}{table_nl}" if table_nl else enriched
            child_chunks.append({
                'id': cid,
                'doc_id': doc_id,
                'source': source,
                'text': enriched,
                'text_raw': table_text,
                # ✨ NEW: add NL-serialized version as a separate embed field
                'text_nl': enriched_nl,
                'doc_type': doc_type,
                'doc_title': doc_title,
                'section_path': [],
                'weight': weight * 1.2,  # Boost table chunks slightly
                'table_data': {  # structured data for comparison logic
                    'header': header,
                    'rows': rows,
                    'page': tbl.get('page'),
                },
            })

    lines = _protect_tables(text.split('\n'))
    sections = _extract_sections('\n'.join(lines))

    # Feature-table block detection (for legacy text-based tables)
    feature_block_range: Optional[Tuple[int, int]] = None
    for i, line in enumerate(lines):
        if re.search(r'SoC\s+Features', line.strip()):
            end = len(lines)
            for j in range(i + 1, len(lines)):
                if re.match(r'^#\s+', lines[j].strip()):
                    end = j
                    break
            feature_block_range = (i, end)
            break

    # Parent chunks: H1/H2 boundary segments
    parent_boundaries = [0]
    for sec in sections:
        if sec['level'] <= 2 and sec['start'] > 0:
            parent_boundaries.append(sec['start'])
    parent_boundaries.append(len(lines))

    parents = []
    for i in range(len(parent_boundaries) - 1):
        p_start = parent_boundaries[i]
        p_end = parent_boundaries[i + 1]
        seg_text = '\n'.join(lines[p_start:p_end]).strip()
        if seg_text:
            parents.append({
                'doc_id': doc_id,
                'source': source,
                'text': seg_text,
                'doc_type': doc_type,
                'doc_title': doc_title,
            })

    # Child chunks: use simple paragraph splitting (O(n), no recursion)
    child_chunks = []
    for parent in parents:
        p_text = parent['text']
        first_line = -1
        first_line_text = p_text.split('\n')[0].strip()
        for i, line in enumerate(lines):
            if line.strip() == first_line_text:
                first_line = i
                break
        section_path = _section_path_for_line(sections, first_line) if first_line >= 0 else []
        metadata_prefix = _build_metadata_prefix(doc_title, doc_type, section_path)

        segments = _split_paragraphs(p_text, chunk_size, min_chunk)

        for seg_text in segments:
            if not seg_text.strip():
                continue

            # Low-value content filter
            if _is_low_value(seg_text):
                continue

            char_len = len(seg_text)
            if char_len < min_chunk and child_chunks:
                child_chunks[-1]['text'] += '\n\n' + seg_text
                continue

            cid = hashlib.md5(f"{doc_id}:{len(child_chunks)}".encode()).hexdigest()[:12]
            enriched_text = f"{metadata_prefix}{seg_text}"
            child_chunks.append({
                'id': cid,
                'doc_id': doc_id,
                'source': source,
                'text': enriched_text,
                'text_raw': seg_text,
                'doc_type': doc_type,
                'doc_title': doc_title,
                'section_path': section_path,
                'weight': weight,
            })

    return child_chunks


# ══════════════════════════════════════════════════════════════════════════════
# 4. Query Expansion (unchanged from original main.py)
# ══════════════════════════════════════════════════════════════════════════════

def expand_query(query: str) -> str:
    """Expand query with domain-specific synonyms."""
    tokens = query.lower().split()
    expanded = list(tokens)
    for token in tokens:
        if token in _QUERY_EXPANSIONS:
            expanded.extend(_QUERY_EXPANSIONS[token])
    seen = set()
    deduped = []
    for t in expanded:
        if t not in seen:
            seen.add(t)
            deduped.append(t)
    return ' '.join(deduped)


# ══════════════════════════════════════════════════════════════════════════════
# 5a. Cross-document comparison utilities
# ══════════════════════════════════════════════════════════════════════════════


def _detect_soc_name(text: str) -> Optional[str]:
    """Detect SoC/product name from text."""
    # SoC pattern extracted from config.yaml soc_to_datasheet keys
    soc_list = '|'.join(re.escape(k) for k in SOC_TO_DATASHEET)
    m = re.search(
        rf'({soc_list})',
        text, re.IGNORECASE
    )
    if m:
        return m.group(1).replace('_', '-').replace(' ', '-').upper()
    return None


def _extract_spec_table(tables_meta: Optional[List[dict]]) -> Optional[dict]:
    """Extract structured specs from datasheet 'SoC Features' table.

    Scans tables_meta for the main SoC features table (the one with feature
    names like 'CPU', 'Memory' in its rows).
    Returns a dict of {feature_name: value_string} or None.
    """
    if not tables_meta:
        return None

    # Keywords that identify a feature name in the first column
    feature_kw = SPEC_FEATURE_KEYWORDS

    # Find the SoC features table — the one with the most feature-keyword
    # matches in its rows. It typically spans pages 14-16.
    best_table = None
    best_score = 0

    for tbl in tables_meta:
        all_rows = [tbl.get('header', [])] + tbl.get('rows', [])
        if len(all_rows) < 5:
            continue

        score = 0
        for row in all_rows:
            if not row:
                continue
            # Check ALL cells, not just col 0 (multi-page tables shift columns)
            for cell in row:
                if cell and any(kw in cell.strip().lower() for kw in feature_kw):
                    score += 1
                    break

        if score > best_score:
            best_score = score
            best_table = tbl

    if not best_table or best_score < 3:
        return None

    # Collect all rows from tables on adjacent pages (same table, multi-page)
    # Use a wider page range (6 pages) since SoC features tables can span
    # pages 10-12 in datasheets
    feature_rows: List[List[str]] = []
    seen_pages: Set[int] = set()
    for tbl in tables_meta:
        p = tbl.get('page')
        bpage = best_table.get('page')
        if bpage is not None and p is not None:
            if abs(p - bpage) > 6:
                continue
            if p in seen_pages:
                continue
            seen_pages.add(p)
        all_rows = [tbl.get('header', [])] + tbl.get('rows', [])
        feature_rows.extend(all_rows)

    # Deduplicate multi-page header rows
    def _deduplicate_headers(rows):
        """Remove header rows that repeat across page boundaries."""
        if len(rows) < 2:
            return rows
        # Collect fingerprints from the first page's rows
        first_page_fp = set()
        for row in rows[:3]:  # first few rows should include header
            fp = '|'.join(c.strip().lower()[:20] for c in row if c)
            first_page_fp.add(fp)
        result = [rows[0]]
        for row in rows[1:]:
            fp = '|'.join(c.strip().lower()[:20] for c in row if c)
            # Check if row looks header-like (short cells, no feature keywords)
            is_header_like = True
            if row:
                for c in row:
                    if c and any(kw in c.lower() for kw in feature_kw):
                        is_header_like = False
                        break
            if fp in first_page_fp and is_header_like:
                continue  # Skip duplicated header
            result.append(row)
        return result

    feature_rows = _deduplicate_headers(feature_rows)

    # Determine the most likely feature-name column index
    # by finding which column has the most keyword matches
    col_scores: Dict[int, int] = {}
    for row in feature_rows:
        if not row:
            continue
        for ci, cell in enumerate(row):
            if cell and any(kw in cell.strip().lower() for kw in feature_kw):
                col_scores[ci] = col_scores.get(ci, 0) + 1

    best_col = 0
    if col_scores:
        best_col = max(col_scores, key=col_scores.get)

    # Extract specs using the detected feature column
    specs: Dict[str, str] = {}
    current_feature = ''
    current_values: List[str] = []

    def _flush_feature():
        nonlocal current_feature, current_values
        if current_feature and current_values:
            specs[current_feature] = ' | '.join(current_values)
        elif current_feature and not current_values:
            pass  # Skip features with no description
        current_feature = ''
        current_values = []

    for row in feature_rows:
        if not row:
            continue
        # Get feature name from the best column
        feat_cell = row[best_col].strip() if best_col < len(row) and row[best_col] else ''
        if not feat_cell:
            # When the merged first column is empty, continue the current feature
            desc_cols = [c.strip() for c in row[best_col + 1:] if c and c.strip()]
            if desc_cols and current_feature:
                current_values.append(' | '.join(desc_cols))
            continue

        # Get description from the column right after the feature column
        desc_cols = [c.strip() for c in row[best_col + 1:] if c and c.strip()]

        # Clean the cell text
        feat_clean = ' '.join(feat_cell.split())

        # Skip header/empty rows
        if feat_clean.lower() in ('description', 'parameter', 'feature', 'item', ''):
            continue
        if 'soc maximum operation' in feat_clean.lower():
            continue

        # Detect if this is a feature header vs continuation
        # (including features that were merged into col 0 due to merged first column)
        is_feature_header = any(kw in feat_clean.lower() for kw in feature_kw)

        # Continuation rows: empty first col (handled above), or start with '-', '|', or digit
        is_continuation = (feat_clean.startswith('-') or feat_clean.startswith('|'))

        # Detect abbreviation/glossary rows (single uppercase acronym + definition)
        def _is_abbreviation_row(feat, descs):
            if not descs:
                return False
            desc_text = ' '.join(descs)
            # Pattern: feature is a short all-uppercase acronym (2-8 chars)
            # with a multi-word definition containing no technical values
            if re.match(r'^[A-Z][A-Z0-9]{1,7}$', feat.replace(' ', '')):
                has_tech = bool(re.search(
                    r'\d+\s*(?:GHz|MHz|GB|MB|KB|bit|core)',
                    desc_text, re.IGNORECASE))
                if not has_tech and len(desc_text.split()) >= 2:
                    return True
            # Pattern: multiple pipe-separated acronyms in feature cell
            if '|' in feat:
                parts = [p.strip() for p in feat.split('|')]
                acro_parts = sum(1 for p in parts if re.match(r'^[A-Z][A-Z0-9]{1,7}$', p))
                if acro_parts >= 2:
                    return True
            return False

        if _is_abbreviation_row(feat_clean, desc_cols):
            _flush_feature()
            continue

        if is_feature_header and not is_continuation:
            _flush_feature()
            current_feature = feat_clean
            if desc_cols:
                current_values.append(' | '.join(desc_cols))
        elif is_continuation and current_feature:
            # Continuation
            if desc_cols:
                current_values.append(feat_clean + ' | ' + ' | '.join(desc_cols))
            else:
                current_values.append(feat_clean)
        elif not is_continuation and current_feature and not is_feature_header:
            # Might be continuation without clear markers
            if desc_cols:
                current_values.append(feat_clean + ' | ' + ' | '.join(desc_cols))
            else:
                current_values.append(feat_clean)

    _flush_feature()

    # Clean up extracted spec values
    specs = _clean_spec_values(specs)

    return {'header': ['Feature', 'Description'], 'specs': specs,
            'raw_rows': feature_rows} if specs else None


def _clean_orphan_chars(text):
    """Aggressively strip PDF orphan artifacts from start/end of text.

    These are inline text fragments from PDF footer/page-number artifacts
    that PyMuPDF injects into table cells without newline boundaries
    (so _clean_table_cell's newline-boundary cleaning misses them).
    """
    _ORPHAN_PATTERNS = [
        # "I 0 ", "V 2 ", "t 6 ", "A 2 " — single-char space digit
        r'^[A-Za-z]\s+\d\s+',
        # "l 1 " — lowercase-L space digit (very common PDF artifact)
        r'^l\s+\d\s+',
        # "S: ", "y: " — single char colon-space
        r'^[A-Za-z]\s*:\s+',
        # trailing orphan single char
        r'\s+[A-Za-z0-9_-]\s*$',
        # leading underscore or dash
        r'^[-_]+\s*',
        # "a 2 s" three-char single-letter patterns
        r'^[a-z]\s+\d\s+[a-z]\s+',
    ]
    prev = None
    while prev != text:
        prev = text
        for pat in _ORPHAN_PATTERNS:
            text = re.sub(pat, '', text)
        text = text.strip()
    return text


def _is_abbreviation_list(text: str) -> bool:
    """Detect if a value is an abbreviation glossary list rather than a spec value."""
    segments = [s.strip() for s in text.split('|') if s.strip()]
    if len(segments) < 2:
        return False
    definition_count = 0
    for seg in segments:
        has_units = bool(re.search(r'\d+\s*(?:GHz|MHz|GB|MB|KB|GBps|Gbps)', seg, re.IGNORECASE))
        starts_upper = bool(re.match(r'^[A-Z]', seg))
        is_acronym = bool(re.match(r'^[A-Z][A-Z0-9]{1,7}$', seg))
        if (starts_upper and not has_units) or is_acronym:
            definition_count += 1
    return definition_count >= len(segments) * 0.6


def _clean_spec_values(specs: Dict[str, str]) -> Dict[str, str]:
    """Post-extraction cleanup of spec key-value pairs.

    Removes PDF footer artifacts, trailing/leading orphan chars,
    abbreviation glossary rows, and known document fingerprint strings.
    """
    cleaned: Dict[str, str] = {}
    # Known footer/document metadata patterns to strip from values
    footer_patterns = [
        r'DS-\d+-\d+_v[\d.]+',           # Document number like DS-10119-001_v1.2
        r'Confidential',                   # Confidential marker
        r'\d+\s*of\s*\d+',                # Page "X of Y"
        r'Nov\w*\s+\d{4}',                # "Nov 2024" date
    ]
    footer_re = re.compile('|'.join(footer_patterns), re.IGNORECASE)

    for feature, value in specs.items():
        if not value:
            continue

        v = value

        # 1. Strip known footer/document metadata patterns
        v = footer_re.sub('', v)

        # 2. Aggressive orphan cleaning on feature key and value
        feat = _clean_orphan_chars(feature.strip())
        v = _clean_orphan_chars(v)

        # 3. Split on '|', clean each segment, deduplicate
        segments = []
        seen_segments = set()
        for seg in v.split('|'):
            s = seg.strip()
            if not s:
                continue
            # Strip trailing orphan single char (common fragment artifact)
            s = re.sub(r'\s+[A-Za-z0-9_-]$', '', s)
            # Strip leading orphan single char
            s = re.sub(r'^[A-Za-z0-9_-]\s+', '', s)
            # Strip isolated dash/underscore segments
            if re.match(r'^[-_]+$', s.strip()):
                continue
            s = s.strip()
            if not s or len(s) <= 2 and re.match(r'^[A-Za-z0-9_-]$', s):
                continue
            # Deduplicate by lowercase
            s_lower = s.lower()
            if s_lower not in seen_segments:
                seen_segments.add(s_lower)
                segments.append(s)

        v = ' | '.join(segments)

        # 4. Clean up remnant whitespace from footer removal
        v = re.sub(r'\s{2,}', ' ', v).strip()

        # 5. Remove entries that became empty after cleaning
        if not v or v in ('-', '_', 'N/A'):
            continue

        # 6. If the value looks like an abbreviation glossary list, skip it entirely
        if _is_abbreviation_list(v):
            continue

        # 7. Merge orphan section-header rows back into the feature name
        #    If the value starts with a known feature keyword and seems short
        #    (under 30 chars, no data values), append to feature name instead.
        if v and len(v) < 30 and any(kw in v.lower() for kw in
                                     ['cpu', 'memory', 'display', 'video',
                                      'pcie', 'usb', 'ethernet',
                                      'camera', 'isp', 'boot']):
            feat = f"{feat} {v}"
            continue  # Don't set the value — it was really a sub-header

        # 8. Strip trailing orphan digits (footer remnants like " 8" at end)
        v = re.sub(r'\s+\d{1,2}$', '', v)

        # 9. Final: skip if key is empty after cleaning
        if not feat:
            continue

        cleaned[feat] = v

    return cleaned


def _detect_comparison_query(query: str) -> Optional[Tuple[List[str], str]]:
    """Detect if query is asking for cross-document comparison.

    Returns (doc_names, feature) or None.
    E.g., 'compare A vs B' -> (['A', 'B'], '')
           'compare A vs B feature' -> (['A', 'B'], 'feature')
    """
    q = query.strip()

    # Pattern 1: "compare X and Y" or "compare X vs Y"
    m = re.search(
        r'compare\s+(.+?)\s+(?:and|vs|versus)\s+(.+?)(?:\s+(.+))?$',
        q, re.IGNORECASE
    )
    if m:
        return ([m.group(1).strip().upper(), m.group(2).strip().upper()],
                (m.group(3) or '').strip())

    # Pattern 2: "X vs Y" or "X vs Y feature"
    m = re.search(
        r'^(.+?)\s+vs\.?\s+(.+?)(?:\s+(.+))?$',
        q, re.IGNORECASE
    )
    if m:
        return ([m.group(1).strip().upper(), m.group(2).strip().upper()],
                (m.group(3) or '').strip())

    # Pattern 3: "difference between X and Y" (English)
    m = re.search(
        r'difference[s]?\s+between\s+(.+?)\s+and\s+(.+?)(?:\s+(.+))?$',
        q, re.IGNORECASE
    )
    if m:
        return ([m.group(1).strip().upper(), m.group(2).strip().upper()],
                (m.group(3) or '').strip())

    # Pattern 5: Chinese comparison patterns
    #   "X and Y - what is the difference/diff/discrepancy/contrast"
    #   "the difference between X and Y"
    #   "compare X and Y"
    #   "X compared with Y"
    m = re.search(r'^(.+?)\s+(?:和|与|跟)\s+(.+?)\s+有.{0,2}(?:区别|差异|不同|差别)', q)
    if m:
        return ([m.group(1).strip().upper(), m.group(2).strip().upper()], '')

    m = re.search(r'^(.+?)\s+(?:和|与|跟)\s+(.+?)\s+的(?:区别|差异|不同|差别)', q)
    if m:
        return ([m.group(1).strip().upper(), m.group(2).strip().upper()], '')

    m = re.search(r'^对比\s+(.+?)\s+(?:和|与|跟)\s+(.+?)\s*$', q)
    if m:
        return ([m.group(1).strip().upper(), m.group(2).strip().upper()], '')

    m = re.search(r'^(.+?)\s+(?:相比|对比)\s+(.+?)\s*$', q)
    if m:
        return ([m.group(1).strip().upper(), m.group(2).strip().upper()], '')

    # Pattern 4: "X specs" or "X datasheet"
    m = re.search(r'^(.+?)\s+(?:specs?|specification|datasheet)s?\s*(.+)?$', q, re.IGNORECASE)
    if m:
        doc_name = m.group(1).strip().upper()
        feature = (m.group(2) or '').strip()
        return ([doc_name], feature)

    return None


def _format_comparison(specs_list: List[dict]) -> str:
    """Build a markdown comparison table from spec dicts."""
    if not specs_list:
        return ""

    # Collect all feature keys across all docs
    all_features: List[str] = []
    seen_features: Set[str] = set()
    for sd in specs_list:
        for k in sd.get('specs', {}):
            if k not in seen_features:
                all_features.append(k)
                seen_features.add(k)

    doc_titles = [sd['doc_title'] for sd in specs_list]

    lines = []
    lines.append('| Feature | ' + ' | '.join(doc_titles) + ' |')
    lines.append('|' + '|'.join(['---'] * (len(doc_titles) + 1)) + '|')

    for feat in all_features:
        row = [feat]
        for sd in specs_list:
            row.append(sd.get('specs', {}).get(feat, '—'))
        lines.append('| ' + ' | '.join(row) + ' |')

    return '\n'.join(lines)


# ── Doc name mapping for cross-document comparison ─────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

class ChromaIndex:
    """ChromaDB-based RAG index with multi-collection + incremental update.

    Collections by doc type:
      - 'chunks'           : unified (backwards compat, also written for insurance)
      - 'datasheet_chunks' : datasheet PDFs
      - 'trm_chunks'       : Technical Reference Manuals
      - 'guide_chunks'     : Developer guides
      - 'api_chunks'       : API reference docs
      - Other doc types get a generic 'docs_chunks' collection.

    Also maintains:
      - 'docs'             : document metadata index
      - 'spec_summary'     : structured spec key-value pairs (for cross-doc comparison)
    """

    COLLECTION_MAP = _cfg.get('collection_map', {})

    def __init__(self, chroma_dir: str, model_name: str = None):
        self._chroma_dir = chroma_dir
        self._client = chromadb.PersistentClient(path=chroma_dir)
        self._ef = LocalModelEmbeddingFunction(model_name)

        # Don't pass embedding_function to ChromaDB — its built-in ONNX embedder
        # tries to download from S3 and fails in this environment.
        # We handle all embedding ourselves externally.
        self.chunks = self._client.get_or_create_collection(
            'chunks',
            metadata={'hnsw:space': 'cosine'},
        )
        self.docs = self._client.get_or_create_collection('docs')

        # Create per-type collections
        self._collections: Dict[str, chromadb.Collection] = {}
        for key, col_name in self.COLLECTION_MAP.items():
            self._collections[key] = self._client.get_or_create_collection(
                col_name,
                metadata={'hnsw:space': 'cosine'},
            )

        # Spec summary collection (for cross-document comparison)
        self.specs = self._client.get_or_create_collection(
            'spec_summary',
            metadata={'hnsw:space': 'cosine'},
        )

        # BM25 index for hybrid retrieval (lazy-built from persisted chunks)
        self._bm25: Optional[_BM25] = None
        self._bm25_ids: List[str] = []
        self._bm25_built: bool = False

    def _bm25_cache_path(self) -> str:
        """Path to persistent BM25 state file."""
        return os.path.join(self._chroma_dir, '_bm25_cache.pkl')

    def _ensure_bm25(self):
        """Build BM25 index from persistent cache or ChromaDB chunks.

        Caches the constructed BM25 object + ID list as a pickle file
        inside the chroma directory. Updated by add_chunks() and
        invalidated by delete_document().

        First load: reads all chunks from ChromaDB (~12s for 48k chunks).
        Subsequent loads: reads pickle file (~1s).
        """
        if self._bm25_built or not _HAS_BM25:
            return
        self._bm25_built = True
        cache_path = self._bm25_cache_path()

        # Try loading from cache first
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'rb') as f:
                    cached = pickle.load(f)
                self._bm25 = cached['bm25']
                self._bm25_ids = cached['ids']
                # Integrity check: cached document count should match
                # the actual chunk count in ChromaDB. If chunks were
                # added incrementally by another process (or in a
                # previous session that didn't persist the cache),
                # the cache is stale and must be rebuilt.
                cached_count = len(self._bm25_ids)
                actual_count = self.chunks.count()
                if cached_count != actual_count and actual_count > 0 and actual_count <= 200000:
                    self._bm25 = None
                    self._bm25_ids = []
                else:
                    return
            except Exception:
                pass  # fall through to rebuild

        # Rebuild from ChromaDB
        try:
            total = self.chunks.count()
            if total == 0 or total > 200000:
                return
            all_docs = []
            all_ids = []
            offset = 0
            limit = 20000
            while offset < total:
                batch = self.chunks.get(limit=limit, offset=offset)
                if not batch or not batch['ids']:
                    break
                for i, doc_id in enumerate(batch['ids']):
                    doc_text = batch['documents'][i] if batch['documents'] else ''
                    all_docs.append(doc_text)
                    all_ids.append(doc_id)
                offset += limit
            if all_docs:
                import re
                tokenized = [re.findall(r'\w+', d.lower()) for d in all_docs]
                self._bm25 = _BM25(tokenized)
                self._bm25_ids = all_ids
                self._bm25_token_cache = tokenized
                # Persist cache (BM25 object + IDs, more compact than raw corpus)
                self._persist_static_bm25_cache(cache_path, self._bm25, self._bm25_ids)
        except Exception as e:
            print(f"  [BM25] Failed to build index: {e}", file=sys.stderr)
            self._bm25 = None
            self._bm25_ids = []
            try:
                os.remove(self._bm25_cache_path())
            except Exception:
                pass

    @staticmethod
    def _persist_static_bm25_cache(cache_path: str, bm25, ids: List[str]):
        """Write BM25 object + IDs to cache file."""
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump({'bm25': bm25, 'ids': ids}, f)
        except Exception:
            pass

    def _persist_bm25_cache(self, cache_path=None, bm25=None, ids=None):
        """Write current BM25 state to cache file."""
        if bm25 is not None and ids is not None:
            self._persist_static_bm25_cache(cache_path, bm25, ids)
            return
        if not _HAS_BM25 or not self._bm25_ids or self._bm25 is None:
            return
        self._persist_static_bm25_cache(self._bm25_cache_path(), self._bm25, self._bm25_ids)

    def _has_hex_code_query(self, query: str) -> bool:
        """Check if query contains hex code patterns."""
        return bool(re.search(r'0x[0-9a-fA-F]{2,8}', query))

    def _extract_hex_codes(self, query: str) -> List[str]:
        """Extract hex code literals from query."""
        return re.findall(r'0x[0-9a-fA-F]{2,8}', query)

    def _get_collection(self, doc_type: str) -> chromadb.Collection:
        """Return the appropriate collection for a given doc type."""
        col_name = self.COLLECTION_MAP.get(doc_type, 'docs_chunks')
        if doc_type not in self._collections:
            self._collections[doc_type] = self._client.get_or_create_collection(
                col_name,
                metadata={'hnsw:space': 'cosine'},
            )
        return self._collections[doc_type]

    # ── Query ────────────────────────────────────────────────────────────

    def search(self, query: str, n_results: int = 10,
               where: Optional[dict] = None,
               doc_type: Optional[str] = None,
               hybrid: bool = True, rerank: bool = True) -> List[dict]:
        """Search the index and return top-n results.

        If doc_type is specified, searches only that doc type's collection.
        If not specified, searches all collections and merges with RRF.

        When hybrid=True, also runs BM25 and fuses with dense results.
        When rerank=True, applies cross-encoder reranking on the
        hybrid-fused top results for final ordering.
        """
        # Ensure BM25 is loaded from persisted chunks on first search
        self._ensure_bm25()

        expanded = expand_query(query)
        q_emb = self._ef([expanded])[0]

        # Phase 1: collect candidates from dense vector search
        if doc_type:
            col = self._get_collection(doc_type)
            results = col.query(
                query_embeddings=[q_emb],
                n_results=n_results * 3,
                where=where,
            )
            candidate_list = self._format_results(results, n_results * 3)
        else:
            all_results: List[dict] = []
            for dtype, col in self._collections.items():
                try:
                    results = col.query(
                        query_embeddings=[q_emb],
                        n_results=n_results * 2,
                        where=where,
                    )
                    formatted = self._format_results(results, n_results * 2)
                    for r in formatted:
                        r['_rank'] = len(all_results)
                        r['_collection'] = dtype
                    all_results.extend(formatted)
                except Exception:
                    continue

            if not all_results:
                return []

            seen: Dict[str, float] = {}
            source_map: Dict[str, dict] = {}
            for r in all_results:
                rid = r['id']
                rrf_score = 1.0 / (60.0 + r.get('_rank', 0))
                if rid in seen:
                    seen[rid] += rrf_score
                else:
                    seen[rid] = rrf_score
                    source_map[rid] = r
                if rid in source_map:
                    if r.get('score', 0) > source_map[rid].get('score', 0):
                        source_map[rid] = r

            merged = sorted(seen.items(), key=lambda x: -x[1])
            candidate_list = []
            for rid, rrf_score in merged[:n_results * 3]:
                r = source_map[rid]
                r['score'] = rrf_score
                candidate_list.append(r)

        # Phase 2: hybrid BM25 + dense fusion (with hex-code boosting)
        if hybrid and _HAS_BM25 and self._bm25 is not None and self._bm25_ids:
            bm25_tokens = re.findall(r'\w+', expanded.lower())
            bm25_scores = self._bm25.get_scores(bm25_tokens)
            bm25_idx = np.argsort(bm25_scores)[::-1]
            bm25_top = []
            for idx in bm25_idx[:n_results * 2]:
                if bm25_scores[idx] > 0:
                    bm25_top.append({
                        'id': self._bm25_ids[idx],
                        'score': float(bm25_scores[idx]),
                        '_bm25_rank': len(bm25_top),
                    })

            # Hex-code boosting: if query contains 0x... patterns, do a
            # secondary BM25 pass for each hex literal independently and
            # promote matching chunks. This is critical for error code
            # lookups where dense vectors fail to capture hex patterns.
            hex_boosted_ids: Set[str] = set()
            if self._has_hex_code_query(query):
                hex_codes = self._extract_hex_codes(query)
                for hc in hex_codes:
                    hex_tokens = re.findall(r'\w+', hc.lower())
                    hc_scores = self._bm25.get_scores(hex_tokens)
                    hc_idx = np.argsort(hc_scores)[::-1]
                    for idx in hc_idx[:n_results]:
                        if hc_scores[idx] > 1:  # at least one occurrence
                            hex_boosted_ids.add(self._bm25_ids[idx])

            # Fuse: existing candidates get a bonus if also found by BM25
            seen_bm25 = {r['id'] for r in bm25_top}
            hybrid_bonus = n_results / max(len(bm25_top), 1)
            for r in candidate_list:
                if r['id'] in seen_bm25:
                    r['score'] += hybrid_bonus
                # Extra boost for hex-code matched chunks
                if r['id'] in hex_boosted_ids:
                    r['score'] += hybrid_bonus * 2

            # Also inject any high BM25 results that dense missed
            seen_dense = {r['id'] for r in candidate_list}
            for br in bm25_top[:n_results]:
                if br['id'] not in seen_dense:
                    br_result = self.chunks.get(ids=[br['id']])
                    br_text = br_result['documents'][0] if br_result.get('documents') else ''
                    br_meta = br_result['metadatas'][0] if br_result.get('metadatas') else {}
                    candidate_list.append({
                        'id': br['id'],
                        'text': br_text,
                        'score': br['score'] * 0.3,  # damped entry score
                        '_bm25_rank': br.get('_bm25_rank', 0),
                        'source': br_meta.get('source', ''),
                        'metadata': br_meta,
                        'doc_type': br_meta.get('doc_type', 'docs'),
                        'section_path': br_meta.get('section_path', ''),
                        'doc_title': br_meta.get('doc_title', ''),
                    })
                if br['id'] in hex_boosted_ids and br['id'] in seen_dense:
                    # Already accounted for above
                    pass

            # Inject hex-boosted results that BM25 found but dense missed
            for hc_id in hex_boosted_ids:
                if hc_id not in seen_dense and not any(r['id'] == hc_id for r in candidate_list):
                    try:
                        idx = self._bm25_ids.index(hc_id)
                        hc_doc_score = float(bm25_scores[idx]) * 0.5
                        # Fetch actual text and metadata from ChromaDB
                        hc_result = self.chunks.get(ids=[hc_id])
                        hc_text = hc_result['documents'][0] if hc_result.get('documents') else ''
                        hc_meta = hc_result['metadatas'][0] if hc_result.get('metadatas') else {}
                        candidate_list.append({
                            'id': hc_id,
                            'text': hc_text,
                            'score': hc_doc_score,
                            'source': hc_meta.get('source', ''),
                            'metadata': hc_meta,
                            'doc_type': hc_meta.get('doc_type', 'docs'),
                            'section_path': hc_meta.get('section_path', ''),
                            'doc_title': hc_meta.get('doc_title', ''),
                            '_hex_boosted': True,
                        })
                    except (ValueError, IndexError):
                        pass

            # Re-sort by fused score
            candidate_list.sort(key=lambda x: -x.get('score', 0))
            candidate_list = candidate_list[:n_results * 4]

        # Phase 3: cross-encoder reranking
        if rerank:
            reranker = _get_reranker()
            if reranker is not None and candidate_list:
                pairs = [(expanded, r.get('text', '')) for r in candidate_list]
                try:
                    ce_scores = reranker.predict(
                        pairs, batch_size=_RERANKER_BATCH, show_progress_bar=False
                    )
                    for i, r in enumerate(candidate_list):
                        r['ce_score'] = float(ce_scores[i]) if i < len(ce_scores) else 0.0
                    candidate_list.sort(key=lambda x: -x.get('ce_score', 0))
                except Exception:
                    pass  # fall back to current ordering

        return candidate_list[:n_results]

    def search_by_metadata(self, where: dict, n_results: int = 20) -> List[dict]:
        """Retrieve chunks by metadata filter only (no vector search).

        Useful for structured lookups (e.g. all chunks for a given doc).
        """
        results = self.chunks.get(where=where, limit=n_results)
        return self._format_get_results(results)

    # ── Build / Update ───────────────────────────────────────────────────

    def add_chunks(self, chunks: List[dict]):
        """Add chunks to the index in batch, routing to the correct collection."""
        if not chunks:
            return

        # Process in sub-batches to avoid long embedding stalls and persist progressively
        EMBED_BATCH = 256
        for start in range(0, len(chunks), EMBED_BATCH):
            sub = chunks[start:start + EMBED_BATCH]
            ids = [c['id'] for c in sub]
            documents = [c.get('text_nl', c['text']) for c in sub]
            metadatas = [{
                'doc_id': c['doc_id'],
                'source': c['source'],
                'doc_type': c.get('doc_type', 'docs'),
                'doc_title': c.get('doc_title', ''),
                'section_path': ' > '.join(c.get('section_path', [])) if c.get('section_path') else '',
                'weight': c.get('weight', 1.0),
                'has_table': str(c.get('table_data') is not None).lower(),
            } for c in sub]

            # Pre-compute embeddings
            embs = self._ef(documents)

            # Write to unified chunks collection
            batch_size = 166
            for i in range(0, len(ids), batch_size):
                self.chunks.add(
                    ids=ids[i:i + batch_size],
                    embeddings=embs[i:i + batch_size],
                    documents=documents[i:i + batch_size],
                    metadatas=metadatas[i:i + batch_size],
                )

            # BM25 index incremental
            if _HAS_BM25:
                tokenized = [re.findall(r'\w+', d.lower()) for d in documents]
                if self._bm25 is None:
                    self._bm25 = _BM25(tokenized)
                    self._bm25_ids = list(ids)
                    self._bm25_token_cache = tokenized
                else:
                    # Rebuild BM25 with accumulated tokens on each sub-batch.
                    # This is simpler and more reliable than trying to mutate
                    # BM25 internals across different library versions.
                    self._bm25 = _BM25(self._bm25_token_cache + tokenized)
                    self._bm25_ids.extend(ids)
                    self._bm25_token_cache.extend(tokenized)
                # Persist cache after each sub-batch so partial builds are saved
                self._persist_bm25_cache()

            # Route to per-type collection
            by_type: Dict[str, List[int]] = {}
            for i, c in enumerate(sub):
                dt = c.get('doc_type', 'docs')
                by_type.setdefault(dt, []).append(i)

            for dt, indices in by_type.items():
                col = self._get_collection(dt)
                t_ids = [ids[i] for i in indices]
                t_docs = [documents[i] for i in indices]
                t_embs = [embs[i] for i in indices]
                t_meta = [metadatas[i] for i in indices]
                for i in range(0, len(t_ids), batch_size):
                    col.add(
                        ids=t_ids[i:i + batch_size],
                        embeddings=t_embs[i:i + batch_size],
                        documents=t_docs[i:i + batch_size],
                        metadatas=t_meta[i:i + batch_size],
                    )

    def add_doc_metadata(self, doc: dict, chunk_ids: List[str]):
        """Record document metadata and which chunks belong to it."""
        did = doc['id']
        summary = doc['text'][:2000] if doc['text'] else ''
        # Pre-compute embedding for the doc metadata too, to avoid ChromaDB's
        # built-in ONNX embedder which tries to download from S3
        doc_emb = self._ef([summary])[0]
        # Compute file content hash for precise incremental change detection
        file_hash = ''
        try:
            with open(doc['source'], 'rb') as _f:
                file_hash = hashlib.md5(_f.read()).hexdigest()
        except OSError:
            pass
        self.docs.upsert(
            ids=[did],
            embeddings=[doc_emb],
            documents=[summary],
            metadatas=[{
                'source': doc['path'],
                'doc_type': doc.get('doc_type', 'docs'),
                'title': doc.get('title', ''),
                'chunk_count': len(chunk_ids),
                'chunk_ids': ','.join(chunk_ids),
                'file_hash': file_hash,
            }],
        )

    def update_document(self, doc: dict) -> int:
        """Update/replace a single document in the index.

        Returns number of chunks indexed.
        """
        # Delete existing chunks for this doc
        self.delete_document(doc['id'])

        # Chunk and re-add
        doc_title = doc.get('title', '') or _doc_title_from_text(doc['text'])
        chunks = chunk_text(
            doc['text'], doc['id'], doc['path'],
            doc_type=doc.get('doc_type', 'docs'),
            doc_title=doc_title,
            weight=doc.get('weight', 1.0),
            tables_meta=doc.get('tables_meta'),
        )

        if chunks:
            self.add_chunks(chunks)
            self.add_doc_metadata(doc, [c['id'] for c in chunks])

        return len(chunks)

    def delete_document(self, doc_id: str):
        """Delete all chunks for a given document from the index."""
        # Find all chunk IDs for this doc
        result = self.chunks.get(where={'doc_id': doc_id})
        if result['ids']:
            self.chunks.delete(ids=result['ids'])
            # Also delete from per-type collections
            for col in self._collections.values():
                try:
                    col.delete(ids=result['ids'])
                except Exception:
                    pass
        # Also remove from docs collection
        try:
            self.docs.delete(ids=[doc_id])
        except Exception:
            pass
        try:
            self.specs.delete(where={'doc_id': doc_id})
        except Exception:
            pass
        # Invalidate BM25 cache — next load will rebuild from scratch
        self._bm25 = None
        self._bm25_ids = []
        # Only remove cache if it exists and if this doc had chunks
        if result['ids']:
            try:
                os.remove(self._bm25_cache_path())
            except Exception:
                pass

    def get_document_list(self) -> List[dict]:
        """List all indexed documents with metadata."""
        results = self.docs.get()
        docs_list = []
        if results['ids']:
            for i, did in enumerate(results['ids']):
                md = results['metadatas'][i] if results['metadatas'] else {}
                docs_list.append({
                    'id': did,
                    'source': md.get('source', ''),
                    'doc_type': md.get('doc_type', ''),
                    'title': md.get('title', ''),
                    'chunk_count': md.get('chunk_count', 0),
                    'mtime': md.get('mtime', '0'),
                    'file_hash': md.get('file_hash', ''),
                })
        return sorted(docs_list, key=lambda x: x['source'])

    @property
    def chunk_count(self) -> int:
        return self.chunks.count()

    @property
    def spec_count(self) -> int:
        return self.specs.count()

    # ── Spec summary (for cross-document comparison) ────────────────────

    def add_specs(self, doc_id: str, doc_title: str, specs: dict):
        """Store structured spec key-value pairs in the spec_summary collection."""
        if not specs:
            return
        ids = []
        documents = []
        metadatas = []
        embs_list = []
        for key, value in specs.items():
            spec_id = hashlib.md5(f"{doc_id}:spec:{key}".encode()).hexdigest()[:12]
            ids.append(spec_id)
            documents.append(f"{key}: {value}")
            metadatas.append({
                'doc_id': doc_id,
                'doc_title': doc_title,
                'spec_name': key,
                'spec_value': value,
            })
        if ids:
            embs_list = self._ef(documents)
            batch_size = 166
            for i in range(0, len(ids), batch_size):
                self.specs.add(
                    ids=ids[i:i + batch_size],
                    embeddings=embs_list[i:i + batch_size],
                    documents=documents[i:i + batch_size],
                    metadatas=metadatas[i:i + batch_size],
                )

    def lookup_specs(self, doc_title_query: str) -> List[dict]:
        """Look up specs by doc title using metadata filter or substring match."""
        # Try exact metadata match first
        results = self.specs.get(where={'doc_title': doc_title_query})
        if results['ids']:
            return self._format_spec_results(results)

        # Try prefix match (e.g. "DS-11562" matches "DS-11562-001_v1.0")
        # This catches datasheet-number lookups before falling to substring.
        all_specs = self.specs.get()
        if all_specs['ids']:
            specs_by_doc: Dict[str, dict] = {}
            q_upper = doc_title_query.upper()
            for i in range(len(all_specs['ids'])):
                md = all_specs['metadatas'][i] if all_specs['metadatas'] else {}
                dt = md.get('doc_title', '')
                if dt.upper().startswith(q_upper):
                    spec_name = md.get('spec_name', '')
                    spec_value = md.get('spec_value', '')
                    if dt not in specs_by_doc:
                        specs_by_doc[dt] = {'doc_title': dt, 'specs': {}}
                    specs_by_doc[dt]['specs'][spec_name] = spec_value
            if specs_by_doc:
                return list(specs_by_doc.values())

        # Fall back to substring match on doc_title
        all_specs = self.specs.get()
        if not all_specs['ids']:
            return []

        specs_by_doc: Dict[str, dict] = {}
        q_upper = doc_title_query.upper()
        for i in range(len(all_specs['ids'])):
            md = all_specs['metadatas'][i] if all_specs['metadatas'] else {}
            dt = md.get('doc_title', '')
            # Substring match: the query (e.g. "ESP32 S3") appears in doc title,
            # OR the doc title prefix (e.g. "DS-11199") contains a token from the query.
            # Skip tokens shorter than 3 chars to avoid false matches from
            # generic prefixes like "DS" or single letters like "U", "X".
            tokens = [t for t in q_upper.replace('-', ' ').split() if len(t) >= 3]
            if q_upper in dt.upper() or any(qtok in dt.upper() for qtok in tokens):
                spec_name = md.get('spec_name', '')
                spec_value = md.get('spec_value', '')
                if dt not in specs_by_doc:
                    specs_by_doc[dt] = {'doc_title': dt, 'specs': {}}
                specs_by_doc[dt]['specs'][spec_name] = spec_value

        return list(specs_by_doc.values())

    @staticmethod
    def _format_spec_results(results) -> List[dict]:
        specs_by_doc: Dict[str, dict] = {}
        for i in range(len(results['ids'])):
            md = results['metadatas'][i] if results['metadatas'] else {}
            dt = md.get('doc_title', '')
            spec_name = md.get('spec_name', '')
            spec_value = md.get('spec_value', '')
            if dt not in specs_by_doc:
                specs_by_doc[dt] = {'doc_title': dt, 'specs': {}}
            specs_by_doc[dt]['specs'][spec_name] = spec_value
        return list(specs_by_doc.values())

    # ── Internal ─────────────────────────────────────────────────────────

    @staticmethod
    def _format_results(results, n_results: int) -> List[dict]:
        formatted = []
        if not results['ids'] or not results['ids'][0]:
            return formatted

        for i in range(len(results['ids'][0])):
            mid = i
            metadata = results['metadatas'][0][mid] if results['metadatas'] else {}
            formatted.append({
                'id': results['ids'][0][i],
                'source': metadata.get('source', ''),
                'score': 1.0 - results['distances'][0][i] if results['distances'] else 0.0,
                'text': results['documents'][0][i] if results['documents'] else '',
                'metadata': metadata,
                'doc_type': metadata.get('doc_type', 'docs'),
                'section_path': metadata.get('section_path', ''),
                'doc_title': metadata.get('doc_title', ''),
            })
        return formatted[:n_results]

    @staticmethod
    def _format_get_results(results) -> List[dict]:
        formatted = []
        if not results['ids']:
            return formatted
        for i in range(len(results['ids'])):
            metadata = results['metadatas'][i] if results['metadatas'] else {}
            formatted.append({
                'id': results['ids'][i],
                'source': metadata.get('source', ''),
                'text': results['documents'][i][:2000] if results['documents'] else '',
                'doc_type': metadata.get('doc_type', 'docs'),
                'section_path': metadata.get('section_path', ''),
                'doc_title': metadata.get('doc_title', ''),
            })
        return formatted


# ══════════════════════════════════════════════════════════════════════════════
# 6. Root directory discovery
# ══════════════════════════════════════════════════════════════════════════════

def _get_project_root() -> str:
    return str(Path(__file__).resolve().parent.parent)


def _get_docs_root() -> str:
    env = os.environ.get('RAG_DOCS_DIR')
    if env:
        return env
    return os.path.join(_get_project_root(), "source")


def _get_chroma_dir() -> str:
    env = os.environ.get('RAG_CHROMA_DIR')
    if env:
        return env
    return os.path.join(_get_project_root(), ".chroma_esp32_all")


def _get_md_dir() -> str:
    env = os.environ.get('RAG_MD_DIR')
    if env:
        return env
    return os.path.join(_get_project_root(), "md")


def _ensure_md_output_dir() -> str:
    out_dir = _get_md_dir()
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def _write_md_file(relpath: str, md_content: str):
    out_dir = _ensure_md_output_dir()
    md_relpath = Path(relpath).with_suffix(".md")
    out_path = out_dir / md_relpath
    out_path.parent.mkdir(parents=True, exist_ok=True)
    header = f"> Source: `{relpath}`\n\n"
    out_path.write_text(header + md_content, encoding='utf-8')


# ══════════════════════════════════════════════════════════════════════════════
# 7. Public API
# ══════════════════════════════════════════════════════════════════════════════

def build_index(docs_dir: Optional[str] = None,
                chroma_dir: Optional[str] = None,
                keep_md: bool = False,
                incremental: bool = True,
                model_name: str = None):
    """Extract, chunk, and build the ChromaDB index.

    True streaming: extracts → chunks → embeds → indexes one doc
    at a time. Never materializes the full corpus in memory.

    WARNING: Full rebuild (incremental=False) will NOT delete existing
    chroma_dir — it only adds/updates from current source files.
    To truly reset, delete .chroma_esp32_all/ manually.
    """

    docs_dir = docs_dir or _get_docs_root()
    chroma_dir = chroma_dir or _get_chroma_dir()

    # Optional path exclusions (semicolon-separated substrings, case-insensitive)
    exclude_dirs = [s.strip().lower() for s in
                    os.environ.get('RAG_EXCLUDE_DIRS', '').split(';') if s.strip()]

    # ── Phase 1: quick inventory (file listing only, NO extraction) ──
    print(f"Scanning {docs_dir}...")
    inventory: List[dict] = []
    for root, _dirs, files in os.walk(docs_dir):
        for fname in sorted(files):
            ext = Path(fname).suffix.lower()
            if ext not in ('.pdf', '.zip', '.xlsx', '.docx', '.md', '.markdown'):
                continue
            fpath = os.path.join(root, fname)
            relpath = os.path.relpath(fpath, docs_dir)
            if exclude_dirs and any(x in relpath.lower() for x in exclude_dirs):
                continue
            try:
                mtime = os.path.getmtime(fpath)
            except OSError:
                mtime = 0
            doc_id = hashlib.md5(fpath.encode()).hexdigest()[:12]
            inventory.append({
                'id': doc_id,
                'path': relpath,
                'source': fpath,
                'mtime': mtime,
            })
    total = len(inventory)
    print(f"Found {total} accessible documents")

    # ── Phase 2: determine which to index (incremental diff) ─────────
    if os.path.isdir(chroma_dir):
        existing = ChromaIndex(chroma_dir, model_name=model_name)
        old_docs = existing.get_document_list()
        old_by_relpath = {}
        for d in old_docs:
            old_by_relpath[os.path.normpath(d['source'])] = d

        current_relpaths = set()
        to_do: List[dict] = []
        for entry in inventory:
            relpath = os.path.normpath(entry['path'])
            current_relpaths.add(relpath)
            old_entry = old_by_relpath.get(relpath)
            if old_entry is None:
                entry['_action'] = 'new'
                to_do.append(entry)
                continue

            # Compare file content hash for change detection
            try:
                with open(entry['source'], 'rb') as _fh:
                    cur_hash = hashlib.md5(_fh.read()).hexdigest()
            except OSError:
                cur_hash = ''
            old_hash = old_entry.get('file_hash', '')
            if cur_hash and old_hash and cur_hash == old_hash:
                # Content unchanged, skip
                continue
            # Content changed, re-index
            entry['_action'] = 'changed'
            to_do.append(entry)

        # Remove deleted
        deleted_relpaths = set(old_by_relpath.keys()) - current_relpaths
        deleted_count = 0
        for relpath in deleted_relpaths:
            try:
                existing.delete_document(old_by_relpath[relpath].get('id', ''))
                deleted_count += 1
                print(f"  Removed deleted: {relpath}")
            except Exception:
                pass

        if not to_do and deleted_count == 0:
            print("Index is up to date.")
            return
        index = existing
        # Sort: PDFs first, ZIPs last (SDK API Reference ZIP takes ~1 hour)
        to_do.sort(key=lambda e: (1 if e['source'].lower().endswith('.zip') else 0, e['path']))
        docs_to_index = len(to_do)
        skipped = total - docs_to_index
        parts = []
        if docs_to_index:
            parts.append(f"{docs_to_index} to index")
        if skipped:
            parts.append(f"{skipped} unchanged skipped")
        if deleted_count:
            parts.append(f"{deleted_count} deleted removed")
        print(f"Incremental: {', '.join(parts)}")
    else:
        print("No existing index found. Performing full build.")
        index = ChromaIndex(chroma_dir, model_name=model_name)
        to_do = inventory
        docs_to_index = total

    # ── Phase 3: stream — extract → chunk → embed → index, one doc at a time ──
    total_chunks = 0
    docs_processed = 0
    start_time = time.time()

    for entry in to_do:
        # Re-extract this single document
        fpath = entry['source']
        doc_count_for_zip = 0
        for doc in get_documents_for_file(fpath, docs_dir):
            doc_title = doc.get('title', '') or _doc_title_from_text(doc['text'])

            sub_label = ""
            if fpath.lower().endswith('.zip') and doc_count_for_zip == 0:
                # Print progress for ZIP once, then emit sub-doc progress silently
                action = entry.get('_action', 'index')
                relpath = entry['path']
                print(f"  [{docs_processed + 1}/{docs_to_index}] {action}: {relpath} ...", flush=True)
            doc_count_for_zip += 1

            # Delete existing before re-adding
            try:
                index.delete_document(doc['id'])
            except Exception:
                pass

            # Extract structured specs from datasheet tables
            if doc.get('doc_type') == 'datasheet' and doc.get('tables_meta'):
                spec_data = _extract_spec_table(doc['tables_meta'])
                if spec_data:
                    index.add_specs(doc['id'], doc_title, spec_data['specs'])

            chunks = chunk_text(
                doc['text'], doc['id'], doc['path'],
                doc_type=doc.get('doc_type', 'docs'),
                doc_title=doc_title,
                weight=doc.get('weight', 1.0),
                tables_meta=doc.get('tables_meta'),
            )
            if keep_md:
                _write_md_file(doc['path'], doc['text'])

            chunk_count = 0
            if chunks:
                index.add_chunks(chunks)
                index.add_doc_metadata(doc, [c['id'] for c in chunks])
                chunk_count = len(chunks)
                total_chunks += chunk_count
                del chunks

            del doc['text'], doc
            gc.collect()

            if doc_count_for_zip % 500 == 0 and doc_count_for_zip > 0:
                elapsed = time.time() - start_time
                print(f"    sub-docs: {doc_count_for_zip} processed, {chunk_count} chunks ({elapsed:.0f}s elapsed)", flush=True)

        if doc_count_for_zip == 0:
            # No sub-documents yielded (PDF/XLSX returned None or ZIP had no content)
            docs_processed += 1
            continue

        docs_processed += 1
        elapsed = time.time() - start_time
        if doc_count_for_zip > 1:
            print(f"  Done: {doc_count_for_zip} sub-docs, {total_chunks} total chunks ({elapsed:.0f}s elapsed)", flush=True)

    spec_count = index.spec_count
    elapsed = time.time() - start_time
    print(f"\nDone! {docs_processed} docs, {total_chunks} chunks + {spec_count} spec entries "
          f"(total {index.chunk_count} chunks in db) in {elapsed:.0f}s")


def extract_zip_stream(filepath: str, base_dir: str):
    """Extract ZIP content and yield one document per HTML/TXT/MD file inside.

    This allows the build pipeline to process ZIP contents one file at a time
    (extract → chunk → embed → index) instead of materializing the entire
    ZIP content in memory before indexing.
    """
    _BINARY_EXTS = {'.jar', '.exe', '.dll', '.so', '.png', '.jpg', '.jpeg',
                    '.gif', '.bmp', '.class', '.pyc', '.o', '.a', '.lib',
                    '.ttf', '.woff', '.eot', '.zip', '.tar', '.gz'}
    import hashlib
    from pathlib import Path

    fname = Path(filepath).stem
    relpath = os.path.relpath(filepath, base_dir)
    doc_id_prefix = hashlib.md5(filepath.encode()).hexdigest()[:12]
    doc_type = _classify_doc_type(relpath)
    weight = _classify_weight(relpath)

    try:
        with zipfile.ZipFile(filepath, 'r') as zf:
            infos = [info for info in zf.infolist() if not info.is_dir()]
            total = len(infos)

            # Quick firmware check
            doc_exts = {'.html', '.htm', '.xhtml', '.txt', '.md', '.docx'}
            doc_count = sum(1 for info in infos
                            if Path(info.filename).suffix.lower() in doc_exts)
            doc_size = sum(info.file_size for info in infos
                           if Path(info.filename).suffix.lower() in doc_exts)
            if total > 100 and doc_count / total < 0.1:
                print(f"  [SKIP] ZIP appears to be source/firmware code "
                      f"({total} files, only {doc_count} docs)")
                return
            if doc_size > 500 * 1024 * 1024:
                print(f"  [SKIP] ZIP doc content too large "
                      f"({doc_size/1024/1024:.0f}MB), would cause memory pressure")
                return

            sub_count = 0
            for info in infos:
                ext = Path(info.filename).suffix.lower()
                if ext in _BINARY_EXTS:
                    continue
                text = None
                if ext in ('.html', '.htm', '.xhtml'):
                    try:
                        raw = zf.read(info.filename)
                        content = raw.decode('utf-8', errors='replace')
                        text = html_to_markdown(content)
                    except Exception as e:
                        print(f"  [WARN] extract failed: {info.filename}: {e}", flush=True)
                        pass
                elif ext in ('.txt', '.md'):
                    try:
                        raw = zf.read(info.filename)
                        text = raw.decode('utf-8', errors='replace').strip()
                    except Exception:
                        pass
                elif ext == '.docx':
                    try:
                        text = docx_bytes_to_markdown(zf.read(info.filename))
                    except Exception:
                        pass

                if not text or not text.strip():
                    continue

                text = clean_extracted_text(text)
                if not text.strip():
                    continue

                sub_count += 1

                if sub_count % 500 == 0:
                    print(f"  ZIP progress: {sub_count} files extracted from {fname}", flush=True)

                sub_id = hashlib.md5(f"{relpath}:{info.filename}".encode()).hexdigest()[:12]
                sub_title = f"{fname}/{info.filename}"

                yield {
                    'id': sub_id,
                    'path': relpath,
                    'source': filepath,
                    'text': text,
                    'tables_meta': None,
                    'weight': weight,
                    'doc_type': doc_type,
                    'title': sub_title,
                }
                del text

            if sub_count == 0:
                print(f"  [SKIP] No extractable content in ZIP {fname}")
    except Exception as e:
        print(f"  [SKIP] ZIP error {filepath}: {e}")


def get_documents_for_file(fpath: str, base_dir: str):
    """Extract a single document by file path, yielding one result."""
    import hashlib
    from pathlib import Path

    if not os.path.isfile(fpath):
        return

    ext = Path(fpath).suffix.lower()
    tables_meta = None
    try:
        if ext == '.pdf':
            text, tables_meta = pdf_to_markdown(fpath, doc_type=_classify_doc_type(os.path.relpath(fpath, base_dir)))
        elif ext == '.zip':
            yield from extract_zip_stream(fpath, base_dir)
            return
        elif ext == '.xlsx':
            text = xlsx_to_markdown(fpath)
        elif ext == '.docx':
            text = docx_to_markdown(fpath)
        elif ext in ('.md', '.markdown'):
            text = md_to_markdown(fpath)
        else:
            return
    except Exception as e:
        print(f"  [SKIP] Error processing {fpath}: {e}")
        return

    if not text:
        return

    if ext not in ('.md', '.markdown'):
        text = clean_extracted_text(text)
    if not text.strip():
        return

    relpath = os.path.relpath(fpath, base_dir)
    doc_id = hashlib.md5(fpath.encode()).hexdigest()[:12]
    doc_type = _classify_doc_type(relpath)
    weight = _classify_weight(relpath)
    yield {
        'id': doc_id,
        'path': relpath,
        'source': fpath,
        'text': text,
        'tables_meta': tables_meta,
        'weight': weight,
        'doc_type': doc_type,
        'title': Path(fpath).stem,
    }

    # NOTE: pbar is only defined in build_index() context, not when
    # get_documents_for_file is called standalone.


def query(index: ChromaIndex, question: str,
          n_results: int = 10, where: Optional[dict] = None,
          doc_type: Optional[str] = None) -> List[dict]:
    """Query the index and return top-n results.

    If doc_type is specified, searches only that doc type's collection.
    If the query looks like a cross-document comparison, returns structured
    comparison results.
    """
    # Check for comparison query
    comparison = _detect_comparison_query(question)
    if comparison:
        doc_names, feature = comparison
        specs_list = []
        for name in doc_names:
            name_upper = name.upper().replace('-', ' ').strip()
            matched = index.lookup_specs(name)
            if not matched:
                part_no = SOC_TO_DATASHEET.get(name_upper)
                if part_no:
                    matched = index.lookup_specs(part_no)
                else:
                    # Broader fallback: check if any datasheet prefix is in the name
                    for part_name, prefix in SOC_TO_DATASHEET.items():
                        if prefix in name_upper or part_name.replace(' ', '') in name_upper:
                            matched = index.lookup_specs(prefix)
                            break
            specs_list.extend(matched)
        if len(specs_list) >= 2:
            comparison_table = _format_comparison(specs_list)
            if comparison_table:
                return [{
                    'type': 'comparison',
                    'table': comparison_table,
                    'specs': specs_list,
                }]
        elif len(specs_list) == 1 and not feature:
            # Single doc specs request
            spec = specs_list[0]
            lines = [f"## {spec['doc_title']} Specifications\n"]
            lines.append('| Feature | Value |')
            lines.append('|---------|-------|')
            for k, v in spec['specs'].items():
                lines.append(f"| {k} | {v} |")
            return [{'type': 'comparison', 'table': '\n'.join(lines), 'specs': specs_list}]

    # Fall through to normal search
    return index.search(question, n_results=n_results, where=where, doc_type=doc_type)


def interactive(index: ChromaIndex):
    """Interactive query loop."""
    print("\nEntering interactive mode. Type 'quit' to exit.\n")
    print("Available commands:")
    print("  /type <doc_type>   — filter by doc type (datasheet, trm, guide, api)")
    print("  /all               — clear type filter")
    print("  /help              — show this help")
    print()
    current_type: Optional[str] = None
    while True:
        try:
            q = input(">>> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not q or q.lower() in ('quit', 'exit'):
            break
        if q.startswith('/'):
            parts = q.split(None, 1)
            cmd = parts[0].lower()
            arg = parts[1] if len(parts) > 1 else ''
            if cmd == '/type' and arg:
                current_type = arg
                print(f"  Filtering by type: {current_type}")
            elif cmd == '/all':
                current_type = None
                print("  Cleared type filter")
            elif cmd == '/help':
                print("  /type <doc_type>  — filter by doc type")
                print("  /all              — clear type filter")
                print("  /help             — show this help")
            continue
        results = query(index, q, doc_type=current_type)
        # Handle comparison results
        if results and results[0].get('type') == 'comparison':
            print(f"\n{results[0]['table']}\n")
            continue
        print(f"\nTop {len(results)} results" +
              (f" (type: {current_type})" if current_type else "") + ":\n")
        for r in results:
            parts = []
            if r.get('doc_type'):
                parts.append(f"[{r['doc_type']}]")
            if r.get('section_path'):
                parts.append(f"{r['section_path']}")
            if parts:
                print(f"  {r['score']:.3f} {' '.join(parts)}")
            print(f"       {r['source']}")
            print(f"       {r['text'][:200]}...")
            print()


if __name__ == '__main__':
    docs_root = _get_docs_root()
    chroma_dir = _get_chroma_dir()
    keep_md = os.environ.get('RAG_KEEP_MD', '').lower() in ('1', 'true', 'yes')

    if len(sys.argv) > 1 and sys.argv[1] == 'query':
        if not os.path.isdir(chroma_dir):
            print(f"ChromaDB not found at {chroma_dir}. Run `python3 -m scripts.rag build` first.")
            sys.exit(1)
        idx = ChromaIndex(chroma_dir)
        # Parse --type flag
        doc_type: Optional[str] = None
        query_args = sys.argv[2:]
        if '--type' in query_args:
            ti = query_args.index('--type')
            if ti + 1 < len(query_args):
                doc_type = query_args[ti + 1]
                query_args = query_args[:ti] + query_args[ti + 2:]
        if query_args:
            q = " ".join(query_args)
            results = query(idx, q, doc_type=doc_type)
            # Handle comparison results
            if results and results[0].get('type') == 'comparison':
                print(f"\nQuery: {q}\n")
                print(results[0]['table'])
                print()
                sys.exit(0)
            print(f"\nQuery: {q}\n")
            for r in results:
                parts = []
                if r.get('doc_type'):
                    parts.append(f"[{r['doc_type']}]")
                if r.get('section_path'):
                    parts.append(f"{r['section_path']}")
                print(f"  [{r['score']:.3f}] {' '.join(parts)}")
                print(f"       {r['source']}")
                print(f"       {r['text'][:500]}...\n")
        else:
            interactive(idx)
    elif len(sys.argv) > 1 and sys.argv[1] == 'update':
        if not os.path.isdir(chroma_dir):
            print(f"ChromaDB not found at {chroma_dir}. Run `python3 -m scripts.rag build` first.")
            sys.exit(1)
        target = " ".join(sys.argv[2:])
        # Find the doc by path (relative or absolute)
        doc = None
        for d in get_documents(docs_root):
            if os.path.normpath(target) in (d['path'], os.path.normpath(d['source'])):
                doc = d
                break
            # free text of non-matching docs immediately
            del d['text'], d
        if not doc:
            print(f"Document not found: {target}")
            sys.exit(1)
        idx = ChromaIndex(chroma_dir)
        n = idx.update_document(doc)
        print(f"Updated {doc['path']}: {n} chunks indexed")
    elif len(sys.argv) > 1 and sys.argv[1] == 'delete':
        if not os.path.isdir(chroma_dir):
            print(f"ChromaDB not found at {chroma_dir}")
            sys.exit(1)
        target = " ".join(sys.argv[2:])
        idx = ChromaIndex(chroma_dir)
        # Try to find doc_id from docs collection
        doc_list = idx.get_document_list()
        doc_id = None
        for d in doc_list:
            if target in (d['source'], d['id']):
                doc_id = d['id']
                break
        if not doc_id:
            print(f"Document not found in index: {target}")
            sys.exit(1)
        idx.delete_document(doc_id)
        print(f"Deleted {doc_id} from index")
    else:
        build_index(docs_root, chroma_dir, keep_md=keep_md)