#!/usr/bin/env python3
"""
Entry point for the RAG system (ChromaDB-backed).

Commands:
  build                     Build/rebuild the search index
  build --keep-md           Build index and save Markdown files to md/
  build --dense             (default) ChromaDB dense embedding index
  query [--type <type>]     Interactive or one-shot query with doc-type filter
  query <text>              One-shot query
  query --type datasheet <text>   Query only datasheets
  errcode <reporter_id> [error_code]    Lookup error code in Error ID Reference XLSX
  update <path>             Update/refresh a single document in the index
  delete <path>             Delete a document from the index
  list                      List indexed documents

Paths are configurable via environment variables:
  RAG_DOCS_DIR     Source document directory (default: <project>/source)
  RAG_CHROMA_DIR   ChromaDB persist directory (default: <project>/.chroma_esp32_all)
  RAG_MD_DIR       Markdown output dir        (default: <project>/md)
"""
import sys
import os
import re
import glob as glob_mod

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from main import build_index, query as _query, ChromaIndex, \
    _get_docs_root, _get_chroma_dir, get_documents


# ── Error ID Reference XLSX scanner ──────────────────────────────────────

_ERROR_ID_XLSX_CACHE = None


def _find_error_id_xlsx() -> str:
    """Locate the Error ID Reference XLSX under docs/."""
    docs_root = _get_docs_root()
    matches = glob_mod.glob(os.path.join(docs_root, 'docs', '**',
                                          '*Error_ID_Reference*.xlsx'),
                            recursive=True)
    if matches:
        return matches[0]
    # fallback: search entire source tree
    matches = glob_mod.glob(os.path.join(docs_root, '**',
                                          '*Error_ID_Reference*.xlsx'),
                            recursive=True)
    return matches[0] if matches else ''


def _load_error_id_xlsx(filepath: str) -> dict:
    """Load and index all SW_ sheets from Error ID Reference XLSX.

    Returns a dict keyed by reporter_id, each value being a list of
    (error_code, error_name, meta_data, description, sheet_name).
    """
    import openpyxl
    wb = openpyxl.load_workbook(filepath, data_only=True)
    index: dict = {}

    for sheet_name in wb.sheetnames:
        if not sheet_name.startswith('SW_'):
            continue
        ws = wb[sheet_name]

        # Find header row: look for row containing "Reporter ID" and "Error Code"
        header_row = None
        header_cols = {}
        for ri, row in enumerate(ws.iter_rows(values_only=True), 1):
            if row is None:
                continue
            for ci, cell in enumerate(row):
                if cell is None:
                    continue
                cell_s = str(cell).strip().lower()
                if 'reporter id' in cell_s:
                    header_cols['reporter'] = ci
                if 'error code' in cell_s:
                    header_cols['error_code'] = ci
                if 'error name' in cell_s:
                    header_cols['error_name'] = ci
                if 'meta data' in cell_s:
                    header_cols['meta_data'] = ci
                if 'description' in cell_s:
                    header_cols['description'] = ci
            if len(header_cols) >= 3:
                header_row = ri
                break

        if header_row is None:
            continue

        # Scan data rows
        for row in ws.iter_rows(min_row=header_row + 1, values_only=True):
            if row is None:
                continue
            rid_idx = header_cols.get('reporter')
            ec_idx = header_cols.get('error_code')
            if rid_idx is None or ec_idx is None:
                continue
            if rid_idx >= len(row) or ec_idx >= len(row):
                continue
            rid = row[rid_idx]
            ec = row[ec_idx]
            if rid is None or ec is None:
                continue
            rid_s = str(rid).strip().lower()
            ec_s = str(ec).strip().lower()
            if not rid_s or not ec_s:
                continue

            en = str(row[header_cols.get('error_name', 0)])[:80] if header_cols.get('error_name', 0) < len(row) and row[header_cols.get('error_name', 0)] else ''
            md = str(row[header_cols.get('meta_data', 0)])[:80] if header_cols.get('meta_data', 0) < len(row) and row[header_cols.get('meta_data', 0)] else ''
            desc = str(row[header_cols.get('description', 0)])[:300] if header_cols.get('description', 0) < len(row) and row[header_cols.get('description', 0)] else ''

            entry = {
                'error_name': en.strip(),
                'error_code': ec_s,
                'meta_data': md.strip(),
                'description': desc.strip(),
                'sheet': sheet_name,
            }

            # Also index by the Reporter ID column from Reporter IDs sheet
            # SW_ sheets use hex strings like '0x8213'
            rid_key = rid_s.lower()
            # Normalize: strip leading '0x' padded to 4 hex digits
            if rid_key.startswith('0x'):
                rid_key = rid_key[2:].zfill(4)
            else:
                # Some use decimal or have prefix
                pass

            if rid_key not in index:
                index[rid_key] = []
            index[rid_key].append(entry)

            # Also index error code values for matching
            # Some Error Code values are like '0x10' in the sheet
            if ec_s.startswith('0x'):
                ec_key = ec_s[2:].zfill(2)
                composite = f"{rid_key}:{ec_key}"
                if composite not in index:
                    index[composite] = []
                index[composite].append(entry)

    wb.close()
    return index


def _lookup_error_code(rid_hex: str, ec_hex: str = '') -> list:
    """Lookup error code in the Error ID Reference XLSX.

    Args:
        rid_hex: Reporter ID in hex (e.g. '0x8213', '8213', '0x8213')
        ec_hex: Error Code in hex (e.g. '0x10', '10')

    Returns:
        List of matching entries, each with error_name, error_code, meta_data,
        description, sheet.
    """
    global _ERROR_ID_XLSX_CACHE

    if _ERROR_ID_XLSX_CACHE is None:
        xlsx_path = _find_error_id_xlsx()
        if not xlsx_path:
            print("  [WARN] Error ID Reference XLSX not found in source tree.")
            return []
        _ERROR_ID_XLSX_CACHE = _load_error_id_xlsx(xlsx_path)

    idx = _ERROR_ID_XLSX_CACHE

    # Normalize reporter ID
    rid = rid_hex.lower().replace('0x', '').zfill(4)
    results = idx.get(rid, [])

    # If error code specified, filter
    if ec_hex:
        ec = ec_hex.lower().replace('0x', '').zfill(2)
        # Try composite key first
        composite = f"{rid}:{ec}"
        composite_results = idx.get(composite, [])
        if composite_results:
            return composite_results
        # Fall back to filtering
        results = [r for r in results if r['error_code'].replace('0x', '').zfill(2) == ec]

    return results


def cmd_errcode(argv):
    """Lookup error codes in the Error ID Reference XLSX.

    Usage: rag errcode <ReporterId> [ErrorCode]
    Example: rag errcode 0x8213 0x10
    """
    if not argv:
        print("Usage: rag errcode <ReporterId> [ErrorCode]")
        print("Example: rag errcode 0x8213 0x10")
        sys.exit(1)

    rid = argv[0]
    ec = argv[1] if len(argv) > 1 else ''

    # Allow "ReptrId-0x8213" style input
    m = re.search(r'(?:ReptrId[_-])?0x[0-9a-fA-F]+', rid)
    if m:
        rid = m.group(0)

    results = _lookup_error_code(rid, ec)

    if not results:
        print(f"\nNo matches found for Reporter ID: {rid}" +
              (f", Error Code: {ec}" if ec else ""))
        print("Tip: Try a broader search, or check if the error comes from")
        print("  a HW source (reporter IDs 0xE000-0xEFFF) or SW source (0x8000-0x8FFF).")
        print(f"  Use `rag errcode {rid}` without error code to see all errors from this reporter.")
        return

    print(f"\nError ID Reference matches for Reporter ID: {rid}" +
          (f", Error Code: {ec}" if ec else "") +
          f"  ({len(results)} match(es))\n")

    for r in results:
        print(f"  Error Name:    {r['error_name']}")
        print(f"  Reporter ID:   {rid}")
        print(f"  Error Code:    {r['error_code']}")
        print(f"  Meta Data:     {r['meta_data']}")
        print(f"  Sheet:         {r['sheet']}")
        print(f"  Description:   {r['description']}")
        print()


# ── Original commands ────────────────────────────────────────────────────

def cmd_build(argv):
    keep_md = '--keep-md' in argv or os.environ.get('RAG_KEEP_MD', '').lower() in ('1', 'true', 'yes')
    build_index(keep_md=keep_md)


def cmd_query(argv):
    chroma_dir = _get_chroma_dir()
    if not os.path.isdir(chroma_dir):
        print("ChromaDB not found. Run `rag build` first.")
        sys.exit(1)
    idx = ChromaIndex(chroma_dir)

    # Parse --type flag
    doc_type = None
    query_args = list(argv)
    if '--type' in query_args:
        ti = query_args.index('--type')
        if ti + 1 < len(query_args):
            doc_type = query_args[ti + 1]
            query_args = query_args[:ti] + query_args[ti + 2:]

    if query_args:
        q = " ".join(query_args)

        # ── Auto-detect error code format queries ──────────────────────
        # If query looks like an error code lookup, supplement with XLSX scan
        err_parts = _detect_error_code_query(q)
        xlsx_results = []
        if err_parts:
            rid, ec = err_parts
            xlsx_results = _lookup_error_code(rid, ec)

        results = _query(idx, q, doc_type=doc_type)

        # Handle comparison results
        if results and results[0].get('type') == 'comparison':
            print(f"\nQuery: {q}\n")
            print(results[0]['table'])
            print()
            return results

        if xlsx_results:
            print(f"\nQuery: {q}\n")
            print(f"  === Error ID Reference Match ===\n")
            for r in xlsx_results:
                print(f"  Error Name:    {r['error_name']}")
                print(f"  Reporter ID:   {rid}")
                print(f"  Error Code:    {r['error_code']}")
                print(f"  Meta Data:     {r['meta_data']}")
                print(f"  Sheet:         {r['sheet']}")
                print(f"  Description:   {r['description']}")
                print()
            print(f"  === RAG Search Results ({len(results)} hits) ===\n")

        if results:
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
        elif not xlsx_results:
            print(f"\nQuery: {q}\n")
            print("  (No results found)\n")
        return results
    else:
        # Interactive — import and use the interactive function from main
        from main import interactive
        interactive(idx)


def _detect_error_code_query(q: str):
    """Detect if the query is an error code format.

    Returns (reporter_id, error_code) tuple, or None.
    Supports formats:
      - ErrCode-0x10 ReptrId-0x8213 ErrAttr-0x11c
      - 0x10 0x8213
      - ReptrId:0x8213
      - error code 0x10 reporter 0x8213
    """
    ql = q.lower()

    # Pattern 1: "ErrCode-0xNN ReptrId-0xNNNN ..."
    m = re.search(r'(?:errcode|ec)[\s_-]*0x([0-9a-f]+)', ql)
    m2 = re.search(r'(?:reptrid|rid)[\s_-]*0x([0-9a-f]+)', ql)
    if m and m2:
        return (f"0x{m2.group(1)}", f"0x{m.group(1)}")

    # Pattern 2: "0xNNNN 0xNN" or "0xNN 0xNNNN" (two hex values)
    hexes = re.findall(r'0x([0-9a-f]+)', ql)
    if len(hexes) >= 2:
        # Assume the shorter one is the error code (1-2 bytes) and the longer is reporter ID
        hexes_sorted = sorted(hexes, key=lambda x: len(x))
        if len(hexes_sorted[0]) <= 4:
            return (f"0x{hexes_sorted[1]}", f"0x{hexes_sorted[0]}")
        return (f"0x{hexes_sorted[0]}", f"0x{hexes_sorted[1]}")

    # Pattern 3: standalone "ReptrId-0xNNNN" or "reporter 0xNNNN"
    m3 = re.search(r'reporter[^0-9a-f]*0x([0-9a-f]+)', ql)
    if m3:
        return (f"0x{m3.group(1)}", '')

    return None


def cmd_update(argv):
    if not argv:
        print("Usage: rag update <doc-path>")
        sys.exit(1)
    target = " ".join(argv)
    chroma_dir = _get_chroma_dir()
    if not os.path.isdir(chroma_dir):
        print("ChromaDB not found. Run `rag build` first.")
        sys.exit(1)

    # Find the doc by scanning source
    docs_root = _get_docs_root()
    documents = get_documents(docs_root)
    doc = None
    for d in documents:
        if target in (d['path'], os.path.normpath(d['source']), os.path.basename(d['path'])):
            doc = d
            break
    if not doc:
        print(f"Document not found: {target}")
        sys.exit(1)

    idx = ChromaIndex(chroma_dir)
    n = idx.update_document(doc)
    print(f"Updated {doc['path']}: {n} chunks indexed")


def cmd_delete(argv):
    if not argv:
        print("Usage: rag delete <doc-path-or-id>")
        sys.exit(1)
    target = " ".join(argv)
    chroma_dir = _get_chroma_dir()
    if not os.path.isdir(chroma_dir):
        print("ChromaDB not found.")
        sys.exit(1)

    idx = ChromaIndex(chroma_dir)
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


def cmd_list(argv):
    chroma_dir = _get_chroma_dir()
    if not os.path.isdir(chroma_dir):
        print("ChromaDB not found.")
        sys.exit(1)
    idx = ChromaIndex(chroma_dir)
    doc_list = idx.get_document_list()
    print(f"\nIndexed documents ({len(doc_list)} total, {idx.chunk_count} chunks, {idx.spec_count} spec entries):\n")
    for d in doc_list:
        print(f"  [{d['doc_type']:15s}] {d['source']:60s} ({d['chunk_count']} chunks)")


CMDS = {
    'build': cmd_build,
    'query': cmd_query,
    'errcode': cmd_errcode,
    'update': cmd_update,
    'delete': cmd_delete,
    'list': cmd_list,
}


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in CMDS:
        print(f"Usage: {sys.argv[0]} <{'|'.join(CMDS)}> [args...]")
        sys.exit(1)
    CMDS[sys.argv[1]](sys.argv[2:])


if __name__ == '__main__':
    main()