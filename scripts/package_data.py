#!/usr/bin/env python3
"""
Package the prebuilt vector databases into release-ready, split archives.

Maintainer tool. For each database it produces:

    dist/<kind>/esp32-rag-<kind>-manifest.json
    dist/<kind>/esp32-rag-<kind>.tar.gz.part00
    dist/<kind>/esp32-rag-<kind>.tar.gz.part01
    ...

Users consume these with scripts/setup_data.py (see README).

Usage:
  python scripts/package_data.py --kind docs
  python scripts/package_data.py --kind all --version data-v1
"""
import argparse
import hashlib
import json
import os
import sys
import tarfile
import tempfile

SKILL_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KINDS = {
    'docs': '.chroma_esp32_all',
    'code': '.chroma_esp32_code',
}
DEFAULT_PART_MB = 1800  # keep each asset under the 2 GB GitHub Release limit


def sha256_file(path, chunk_size=1024 * 1024):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(chunk_size), b''):
            h.update(block)
    return h.hexdigest()


def make_tar(src, arcname, tar_path):
    print(f"[tar] {src} -> {tar_path} (compression may take a while)")
    with tarfile.open(tar_path, 'w:gz', compresslevel=6) as tar:
        tar.add(src, arcname=arcname)


def split_file(path, part_size, out_dir, base_name):
    parts = []
    full = hashlib.sha256()
    with open(path, 'rb') as f:
        idx = 0
        while True:
            chunk = f.read(part_size)
            if not chunk:
                break
            full.update(chunk)
            name = f"{base_name}.part{idx:02d}"
            with open(os.path.join(out_dir, name), 'wb') as o:
                o.write(chunk)
            parts.append({
                "name": name,
                "size": len(chunk),
                "sha256": hashlib.sha256(chunk).hexdigest(),
            })
            print(f"[split] {name}  {len(chunk) / 1e6:.1f} MB")
            idx += 1
    return parts, full.hexdigest()


def package_kind(kind, version, out_dir, part_mb):
    src = os.path.join(SKILL_DIR, KINDS[kind])
    if not os.path.isdir(src):
        print(f"[skip] {kind}: {src} not found", file=sys.stderr)
        return False

    kdir = os.path.join(out_dir, kind)
    os.makedirs(kdir, exist_ok=True)
    base = f"esp32-rag-{kind}.tar.gz"

    with tempfile.TemporaryDirectory() as tmpdir:
        tar_path = os.path.join(tmpdir, base)
        make_tar(src, KINDS[kind], tar_path)
        parts, full_hash = split_file(
            tar_path, part_mb * 1024 * 1024, kdir, base)

    manifest = {
        "kind": kind,
        "version": version,
        "arcname": KINDS[kind],
        "sha256": full_hash,
        "total_size": sum(p["size"] for p in parts),
        "parts": parts,
    }
    mpath = os.path.join(kdir, f"esp32-rag-{kind}-manifest.json")
    with open(mpath, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2)

    print(f"[ok] {kind}: {len(parts)} part(s), "
          f"{manifest['total_size'] / 1e9:.2f} GB -> {kdir}")
    return True


def main():
    ap = argparse.ArgumentParser(
        description="Package prebuilt ESP-RAG vector databases for release.")
    ap.add_argument('--kind', choices=['docs', 'code', 'all'], default='docs')
    ap.add_argument('--version', default='data-v1',
                    help='release tag used by setup_data.py (default: data-v1)')
    ap.add_argument('--out', default=os.path.join(SKILL_DIR, 'dist'))
    ap.add_argument('--part-size-mb', type=int, default=DEFAULT_PART_MB)
    args = ap.parse_args()

    kinds = list(KINDS) if args.kind == 'all' else [args.kind]
    ok = all(package_kind(k, args.version, args.out, args.part_size_mb)
             for k in kinds)

    if ok:
        print("\nNext: create a GitHub Release tagged "
              f"'{args.version}' and upload every file under "
              f"{args.out}/<kind>/")
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
