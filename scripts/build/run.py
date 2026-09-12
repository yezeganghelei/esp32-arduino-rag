#!/usr/bin/env python3
"""
Build entry — provides streaming incremental build with monitoring.

Usage:
  python3 -m scripts.build.run                     # Incremental build (full source/)
  python3 -m scripts.build.run <path>              # Build only specified file/dir
  python3 -m scripts.build.run --model <name>      # Build with specified dense model
  python3 -m scripts.build.run <path> --model <name>
"""
import sys, os, time, threading, argparse


def _build(target_path=None, model_name=None):
    try:
        import pysqlite3
        sys.modules['sqlite3'] = pysqlite3
    except ImportError:
        pass  # fall back to built-in sqlite3 (3.40.1, sufficient for chromadb)

    # Derive paths from script location
    _SKILL_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    CHROMA_DIR = os.environ.get('RAG_CHROMA_DIR', os.path.join(_SKILL_DIR, '.chroma_esp32_all'))
    DOCS_DIR = os.environ.get('RAG_DOCS_DIR', os.path.join(_SKILL_DIR, 'source'))

    os.environ['RAG_DOCS_DIR'] = DOCS_DIR
    os.environ['RAG_CHROMA_DIR'] = CHROMA_DIR

    sys.path.insert(0, os.path.join(_SKILL_DIR, 'scripts'))
    from scripts.main import ChromaIndex, get_documents_for_file, chunk_text, build_index

    start_time = time.time()

    if target_path:
        target_path = os.path.abspath(target_path)
        index = ChromaIndex(CHROMA_DIR, model_name=model_name)
        total_chunks = 0
        sub_count = 0
        total_to_index = None

        if os.path.isdir(target_path):
            from scripts.main import get_documents
            total_to_index = sum(1 for _ in get_documents(target_path))
            doc_gen = get_documents(target_path)
        else:
            doc_gen = get_documents_for_file(target_path, os.path.dirname(target_path))

        for doc in doc_gen:
            sub_count += 1
            doc_title = doc.get('title', '') or ''
            try:
                index.delete_document(doc['id'])
            except Exception:
                pass

            chunks = chunk_text(
                doc['text'], doc['id'], doc['path'],
                doc_type=doc.get('doc_type', 'docs'),
                doc_title=doc_title,
                weight=doc.get('weight', 1.0),
            )

            chunk_count = 0
            if chunks:
                index.add_chunks(chunks)
                index.add_doc_metadata(doc, [c['id'] for c in chunks])
                chunk_count = len(chunks)
                total_chunks += chunk_count
                del chunks
            del doc['text'], doc

            # Per-file progress
            progress_pct = ""
            if total_to_index and total_to_index > 0:
                progress_pct = f" ({sub_count}/{total_to_index})"
            print(f"  [{sub_count}]{progress_pct} {chunk_count} chunks this doc, "
                  f"{total_chunks} total ({time.time()-start_time:.0f}s)", flush=True)

        elapsed = time.time() - start_time
        print(f"\nDone! {sub_count} sub-docs, {total_chunks} chunks added ({elapsed:.0f}s)", flush=True)
        print(f"Total chunks in db: {index.chunk_count}", flush=True)
    else:
        build_index(docs_dir=DOCS_DIR, chroma_dir=CHROMA_DIR, incremental=True, model_name=model_name)


def _monitor(_DOCS_DIR, _CHROMA_DIR, total_to_index=None):
    """Monitor progress, print every 5 minutes."""
    prev_chunk_count = 0
    stable_count = 0
    while True:
        time.sleep(300)  # 5 minutes
        try:
            try:
                import pysqlite3
                sys.modules['sqlite3'] = pysqlite3
            except ImportError:
                pass  # fall back to built-in sqlite3
            sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'scripts'))
            from scripts.main import ChromaIndex
            idx = ChromaIndex(_CHROMA_DIR)
            doc_list = idx.get_document_list()
            indexed_docs = [d for d in doc_list if d['chunk_count'] > 0]
            total_chunks = idx.chunk_count

            progress = ""
            if total_to_index:
                progress = f" [{len(indexed_docs)}/{total_to_index} docs]"

            # Stalled detection
            if total_chunks == prev_chunk_count:
                stable_count += 1
            else:
                stable_count = 0
            prev_chunk_count = total_chunks
            stalled = " [STALLED?]" if stable_count >= 3 else ""

            print(f"\n=== [MONITOR @ {time.strftime('%H:%M:%S')}] "
                  f"{len(indexed_docs)} documents, {total_chunks} total chunks"
                  f"{progress}{stalled} ===", flush=True)

            # Group by series prefix
            by_prefix = {}
            for d in indexed_docs:
                prefix = d['source'].split('/')[0] if '/' in d['source'] else 'other'
                by_prefix.setdefault(prefix, {'docs': 0, 'chunks': 0})
                by_prefix[prefix]['docs'] += 1
                by_prefix[prefix]['chunks'] += d['chunk_count']
            for prefix in sorted(by_prefix.keys()):
                info = by_prefix[prefix]
                print(f"  {prefix}/: {info['docs']} docs, {info['chunks']} chunks", flush=True)

            # Show ZIP builds in progress
            source_counts = {}
            for d in indexed_docs:
                if d['source'].endswith('.zip'):
                    # For ZIPs, aggregate by source path
                    src = d['source']
                    source_counts[src] = source_counts.get(src, 0) + d['chunk_count']
            if source_counts:
                for src, cc in sorted(source_counts.items()):
                    short = src[-60:] if len(src) > 60 else src
                    print(f"  ZIP: {cc} chunks from ...{short}", flush=True)
        except Exception as e:
            print(f"\n=== [MONITOR] Error: {e} ===", flush=True)


if __name__ == '__main__':
    if '--help' in sys.argv or '-h' in sys.argv:
        print(__doc__)
        sys.exit(0)

    parser = argparse.ArgumentParser(description='Build RAG index', add_help=False)
    parser.add_argument('target', nargs='?', default=None, help='File or directory to build')
    parser.add_argument('--model', default=None, help='Dense model name (e.g. gte-base-en-v1.5, default from config.yaml)')
    args = parser.parse_args()

    target = args.target

    _SKILL_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    CHROMA_DIR = os.environ.get('RAG_CHROMA_DIR', os.path.join(_SKILL_DIR, '.chroma_esp32_all'))
    DOCS_DIR = os.environ.get('RAG_DOCS_DIR', os.path.join(_SKILL_DIR, 'source'))

    t = threading.Thread(
        target=_monitor, args=(DOCS_DIR, CHROMA_DIR), daemon=True
    )
    t.start()
    _build(target, model_name=args.model)