#!/usr/bin/env python3
"""
One-shot data setup: download the models and the prebuilt vector databases.

Models are fetched from HuggingFace (see scripts/download_models.py). The
vector databases are fetched from a GitHub Release produced by
scripts/package_data.py. Everything is written into the directories the app
expects:

    models/dense/<name>/              embedding models
    models/cross-encoder/<name>/      reranking model
    .chroma_esp32_all/                prebuilt documentation DB
    .chroma_esp32_code/               prebuilt source-code DB (optional)

Usage:
  python -m scripts.setup_data                 # models + docs DB
  python -m scripts.setup_data --db all        # + code DB
  python -m scripts.setup_data --db none       # models only
  python -m scripts.setup_data --no-models --db docs
  python -m scripts.setup_data --force         # overwrite existing DBs
  python -m scripts.setup_data --mirror        # HF mirror (China)

This is normally run inside the container via setup.ps1 / setup.sh.
"""
import argparse
import hashlib
import json
import os
import shutil
import sys
import tarfile
import tempfile
import urllib.request
import zipfile

SKILL_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if SKILL_DIR not in sys.path:
    sys.path.insert(0, SKILL_DIR)

DB_DIRS = {
    'docs': '.chroma_esp32_all',
    'code': '.chroma_esp32_code',
}
DEFAULT_DATA_URL = (
    'https://github.com/yezeganghelei/esp32-arduino-rag/releases/download/data-v1'
)
DEFAULT_CORPUS_URL = (
    'https://github.com/yezeganghelei/ESP32/archive/refs/heads/main.zip'
)
# The corpus repo is a single tree; this is the documented split:
#   "01. Software Code" -> source_code/
#   every other top-level directory -> source/
CODE_DIR_NAME = '01. Software Code'
# Top-level corpus directories skipped by default (not relevant to retrieval).
DEFAULT_EXCLUDE_DIRS = {
    '05. Common Tools',
    '08. ESP32-Arduino-AI-Agent-RAG',
    'Products',
}
UA = {'User-Agent': 'esp-rag-setup/1.0'}


def log(msg, **kwargs):
    print(msg, flush=True, **kwargs)


def _open(url):
    req = urllib.request.Request(url, headers=UA)
    return urllib.request.urlopen(req, timeout=60)


def _sha256_file(path, chunk_size=1024 * 1024):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(chunk_size), b''):
            h.update(block)
    return h.hexdigest()


def _download(url, dest, expected_sha=None, expected_size=None):
    """Download with retries/backoff (network and DNS can be flaky)."""
    import time
    retries = max(1, int(os.environ.get('RAG_DOWNLOAD_RETRIES', '6')))
    delay = 3
    last = None
    for attempt in range(1, retries + 1):
        try:
            return _download_once(url, dest, expected_sha, expected_size)
        except Exception as exc:
            last = exc
            log(f"[warn] download failed ({attempt}/{retries}): {exc}")
            if attempt < retries:
                log(f"       retrying in {delay}s ...")
                time.sleep(delay)
                delay = min(delay * 2, 30)
    raise last


def _download_once(url, dest, expected_sha=None, expected_size=None):
    log(f"[download] {url}")
    h = hashlib.sha256()
    with _open(url) as r, open(dest, 'wb') as f:
        total = int(r.headers.get('Content-Length') or 0)
        done = 0
        while True:
            chunk = r.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)
            h.update(chunk)
            done += len(chunk)
            if total:
                pct = done * 100 // total
                print(f"\r  {done / 1e6:.0f}/{total / 1e6:.0f} MB ({pct}%)",
                      end='', flush=True)
    if total:
        print()
    size = os.path.getsize(dest)
    if expected_size is not None and size != expected_size:
        raise ValueError(f"size mismatch for {url}: {size} != {expected_size}")
    if expected_sha and h.hexdigest() != expected_sha:
        raise ValueError(f"sha256 mismatch for {url}")
    return dest


def _setup_db(kind, base_url, force):
    arcname = DB_DIRS[kind]
    target = os.path.join(SKILL_DIR, arcname)
    manifest_url = f"{base_url.rstrip('/')}/esp32-rag-{kind}-manifest.json"

    import time
    manifest = None
    last = None
    for attempt in range(1, 4):
        try:
            with _open(manifest_url) as r:
                manifest = json.load(r)
            break
        except Exception as exc:
            last = exc
            log(f"[warn] {kind}: manifest fetch failed ({attempt}/3): {exc}")
            time.sleep(2 * attempt)
    if manifest is None:
        log(f"[warn] {kind}: cannot fetch manifest ({last})")
        log("       Prebuilt data is not published (or the URL is wrong).")
        return False

    if (os.path.isfile(os.path.join(target, 'chroma.sqlite3')) and not force):
        log(f"[skip] {kind}: {target} already exists (use --force to overwrite)")
        return True

    tmpdir = tempfile.mkdtemp(prefix=f'esp32rag-{kind}-')
    try:
        tar_path = os.path.join(tmpdir, f'esp32-rag-{kind}.tar.gz')
        with open(tar_path, 'wb') as out:
            for part in manifest['parts']:
                p = os.path.join(tmpdir, part['name'])
                _download(f"{base_url.rstrip('/')}/{part['name']}", p,
                          part.get('sha256'), part.get('size'))
                with open(p, 'rb') as pf:
                    shutil.copyfileobj(pf, out)
                os.remove(p)

        if manifest.get('sha256'):
            actual = _sha256_file(tar_path)
            if actual != manifest['sha256']:
                raise ValueError(f"{kind}: archive sha256 mismatch")

        if force and os.path.isdir(target):
            shutil.rmtree(target)

        log(f"[extract] {kind} -> {target}")
        with tarfile.open(tar_path, 'r:gz') as tar:
            for member in tar.getmembers():
                if not (member.name == arcname
                        or member.name.startswith(arcname + '/')):
                    raise ValueError(f"unsafe path in archive: {member.name}")
            tar.extractall(SKILL_DIR)

        log(f"[ok] {kind} ready at {target}")
        return True
    except Exception as exc:
        log(f"[error] {kind}: {exc}")
        return False
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def _setup_corpus(url, force, exclude_dirs):
    """Download the ESP32 corpus zip and split it into source/ + source_code/."""
    src_root = os.path.join(SKILL_DIR, 'source')
    code_root = os.path.join(SKILL_DIR, 'source_code')

    def _has_content(path):
        if not os.path.isdir(path):
            return False
        for entry in os.scandir(path):
            # A tracked readme.md (or placeholder) does not count as corpus.
            if entry.name.lower() in ('readme.md', '.gitkeep'):
                continue
            return True
        return False

    if not force and (_has_content(src_root) or _has_content(code_root)):
        log("[skip] corpus: source/ or source_code/ already has content "
            "(use --force to replace)")
        return True

    tmpdir = tempfile.mkdtemp(prefix='esp32rag-corpus-')
    try:
        zip_path = os.path.join(tmpdir, 'esp32.zip')
        _download(url, zip_path)
        log("[extract] corpus archive")
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(tmpdir)

        roots = [os.path.join(tmpdir, e) for e in os.listdir(tmpdir)
                 if os.path.isdir(os.path.join(tmpdir, e)) and e != '__MACOSX']
        if not roots:
            raise ValueError('cannot locate extracted corpus root')
        root = roots[0]

        os.makedirs(src_root, exist_ok=True)
        os.makedirs(code_root, exist_ok=True)
        exclude = set(DEFAULT_EXCLUDE_DIRS) | set(exclude_dirs or ())
        moved = 0
        for name in sorted(os.listdir(root)):
            src = os.path.join(root, name)
            if not os.path.isdir(src) or name in exclude:
                continue
            dest = os.path.join(
                code_root if name == CODE_DIR_NAME else src_root, name)
            if os.path.exists(dest):
                shutil.rmtree(dest)
            log(f"[corpus] {name} -> {dest}")
            shutil.move(src, dest)
            moved += 1
        log(f"[ok] corpus split into source/ and source_code/ ({moved} dirs)")
        return moved > 0
    except Exception as exc:
        log(f"[error] corpus: {exc}")
        return False
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def _setup_models(args):
    from scripts import download_models as dm

    cfg = dm._load_config_models()
    dense_default, ce_default = dm._defaults(cfg)

    if args.all_models:
        dense = dm._available(cfg, 'dense') or list(dm.DENSE_REPOS)
        ce = dm._available(cfg, 'cross_encoder') or list(dm.CE_REPOS)
    else:
        dense = args.model or [dense_default]
        ce = [ce_default]

    ok = True
    for name in dense:
        ok = dm._download('dense', name) and ok
    for name in ce:
        ok = dm._download('cross-encoder', name) and ok
    return ok


def main():
    ap = argparse.ArgumentParser(
        description="Download models and prebuilt vector databases for ESP-RAG.")
    ap.add_argument('--db', choices=['none', 'docs', 'code', 'all'], default='docs',
                    help='which prebuilt vector DB(s) to download (default: docs)')
    ap.add_argument('--no-models', dest='models', action='store_false',
                    help='skip the model download')
    ap.add_argument('--all-models', action='store_true',
                    help='download every supported model, not just the defaults')
    ap.add_argument('--model', action='append', default=[], metavar='NAME',
                    help='dense model name to download (repeatable)')
    ap.add_argument('--mirror', action='store_true',
                    help='use https://hf-mirror.com (useful in China)')
    ap.add_argument('--base-url', default=os.environ.get('RAG_DATA_URL', DEFAULT_DATA_URL),
                    help='base URL of the release assets (or env RAG_DATA_URL)')
    ap.add_argument('--corpus', action='store_true',
                    help='also download the ESP32 corpus and split it into '
                         'source/ and source_code/ (large)')
    ap.add_argument('--corpus-url',
                    default=os.environ.get('RAG_CORPUS_URL', DEFAULT_CORPUS_URL),
                    help='corpus zip URL (or env RAG_CORPUS_URL)')
    ap.add_argument('--exclude-dir', action='append', default=[],
                    dest='exclude_dirs', metavar='NAME',
                    help='additional top-level corpus directory to skip '
                         '(repeatable; 05. Common Tools, '
                         '08. ESP32-Arduino-AI-Agent-RAG and Products are '
                         'always skipped)')
    ap.add_argument('--force', action='store_true',
                    help='overwrite existing vector databases and corpus')
    args = ap.parse_args()

    if args.mirror:
        os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
        log("[info] using HF mirror: https://hf-mirror.com")

    ok = True

    if args.models:
        log("== models ==")
        ok = _setup_models(args) and ok

    if args.db != 'none':
        kinds = list(DB_DIRS) if args.db == 'all' else [args.db]
        for kind in kinds:
            log(f"== {kind} database ==")
            ok = _setup_db(kind, args.base_url, args.force) and ok

    if args.corpus:
        log("== corpus ==")
        ok = _setup_corpus(args.corpus_url, args.force,
                           set(args.exclude_dirs)) and ok

    log("")
    if ok:
        log("Done. Start the service with:  docker compose up -d")
        return 0
    log("Finished with warnings/errors — see messages above.", file=sys.stderr)
    return 1


if __name__ == '__main__':
    sys.exit(main())
