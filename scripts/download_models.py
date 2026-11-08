#!/usr/bin/env python3
"""
Download the embedding / reranking models required by ESP-RAG.

Models are NOT committed to Git (see .gitignore). Run this once before the
first index build. Files are placed under:

    models/dense/<name>/              (embedding models)
    models/cross-encoder/<name>/      (reranking model)

which is exactly where scripts/main.py looks for them.

Usage:
  python -m scripts.download_models             # defaults from config.yaml
  python -m scripts.download_models --all       # every supported model
  python -m scripts.download_models --model gte-base-en-v1.5
  python -m scripts.download_models --list      # show status, download nothing
  python -m scripts.download_models --mirror    # use https://hf-mirror.com (China)
"""
import argparse
import os
import sys

# Model name (as used in config.yaml) -> HuggingFace repo id
DENSE_REPOS = {
    'all-MiniLM-L6-v2': 'sentence-transformers/all-MiniLM-L6-v2',
    'bge-base-en-v1.5': 'BAAI/bge-base-en-v1.5',
    'gte-base-en-v1.5': 'Alibaba-NLP/gte-base-en-v1.5',
}
CE_REPOS = {
    'cross-encoder-ms-marco-MiniLM-L-6-v2': 'cross-encoder/ms-marco-MiniLM-L-6-v2',
}
# Models that are not plain HuggingFace repos and must be fetched manually.
UNSUPPORTED = {
    'embeddinggemma-300m-npu':
        'Nexa NPU format; download manually and place it under models/dense/.',
}

SKILL_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(SKILL_DIR, 'models')


def _load_config_models():
    try:
        import yaml
        with open(os.path.join(SKILL_DIR, 'config.yaml'), encoding='utf-8') as f:
            cfg = yaml.safe_load(f) or {}
        return cfg.get('models', {}) or {}
    except Exception:
        return {}


def _available(cfg, key):
    names = []
    for m in (cfg.get(key) or {}).get('available', []) or []:
        if isinstance(m, dict) and m.get('name'):
            names.append(m['name'])
    return names


def _defaults(cfg):
    dense = ((cfg.get('dense') or {}).get('default')
             or 'bge-base-en-v1.5')
    ce = ((cfg.get('cross_encoder') or {}).get('default')
          or 'cross-encoder-ms-marco-MiniLM-L-6-v2')
    return dense, ce


def _dir_size(path):
    total = 0
    for root, _dirs, files in os.walk(path):
        for fn in files:
            try:
                total += os.path.getsize(os.path.join(root, fn))
            except OSError:
                pass
    return total


def _download(kind, name):
    """Download one model. `kind` is 'dense' or 'cross-encoder'."""
    repos = DENSE_REPOS if kind == 'dense' else CE_REPOS
    if name in UNSUPPORTED:
        print(f"[skip] {name}: {UNSUPPORTED[name]}")
        return True
    repo_id = repos.get(name)
    if not repo_id:
        print(f"[error] unknown {kind} model '{name}'", file=sys.stderr)
        return False

    dest = os.path.join(MODELS_DIR, kind, name)
    print(f"[download] {name}\n           repo: {repo_id}\n           dest: {dest}")
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("[error] huggingface_hub is missing. "
              "Run: pip install -r requirements.txt", file=sys.stderr)
        return False

    # Allow downloading even if a sourced .env exported offline flags.
    saved = {k: os.environ.get(k) for k in ('HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE')}
    os.environ['HF_HUB_OFFLINE'] = '0'
    os.environ['TRANSFORMERS_OFFLINE'] = '0'
    os.makedirs(dest, exist_ok=True)
    try:
        snapshot_download(repo_id=repo_id, local_dir=dest)
    except Exception as exc:
        print(f"[error] failed to download {name}: {exc}", file=sys.stderr)
        return False
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    print(f"[ok] {name} ({_dir_size(dest) / 1e6:.1f} MB)")
    return True


def _list(cfg):
    print("dense (embedding) models:")
    for name in _available(cfg, 'dense'):
        repo = DENSE_REPOS.get(name) or ('manual' if name in UNSUPPORTED else 'unknown')
        present = os.path.isdir(os.path.join(MODELS_DIR, 'dense', name))
        print(f"  {name:<34} {repo:<48} {'[present]' if present else ''}")
    print("cross-encoder (reranking) models:")
    for name in _available(cfg, 'cross_encoder'):
        repo = CE_REPOS.get(name) or 'unknown'
        present = os.path.isdir(os.path.join(MODELS_DIR, 'cross-encoder', name))
        print(f"  {name:<34} {repo:<48} {'[present]' if present else ''}")


def main():
    cfg = _load_config_models()
    dense_default, ce_default = _defaults(cfg)

    ap = argparse.ArgumentParser(
        description="Download ESP-RAG models (they are not committed to Git).")
    ap.add_argument('--model', action='append', default=[], metavar='NAME',
                    help='dense model name to download (repeatable)')
    ap.add_argument('--all', action='store_true',
                    help='download all supported models')
    ap.add_argument('--list', action='store_true',
                    help='list models and whether they are present locally')
    ap.add_argument('--mirror', action='store_true',
                    help='use https://hf-mirror.com (useful in China)')
    args = ap.parse_args()

    if args.list:
        _list(cfg)
        return 0

    if args.mirror:
        os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
        print("[info] using HF mirror: https://hf-mirror.com")

    if args.all:
        dense = _available(cfg, 'dense') or list(DENSE_REPOS)
        ce = _available(cfg, 'cross_encoder') or list(CE_REPOS)
    else:
        dense = args.model or [dense_default]
        ce = [ce_default]

    ok = True
    for name in dense:
        ok = _download('dense', name) and ok
    for name in ce:
        ok = _download('cross-encoder', name) and ok

    if ok:
        print("\nDone. Next step: build the index with  python -m scripts.build.run")
        return 0
    print("\nSome models failed. See errors above.", file=sys.stderr)
    return 1


if __name__ == '__main__':
    sys.exit(main())
