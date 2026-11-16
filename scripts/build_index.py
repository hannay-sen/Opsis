"""Weekend 1: embed every image and build the FAISS index.

Expects images in folders, e.g. the Bonn Furniture Styles dataset:
    data/images/<category>/<style>/<file>.jpg     (beds/Modern/123.jpg)
Folder names become labels, which evaluate.py uses to score results.

Usage (from project root):
    python -m scripts.build_index --images data/images --out data/index
    python -m scripts.build_index --images data/images --limit 5000   # quick first run
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import faiss
import numpy as np
from PIL import Image, UnidentifiedImageError

from backend.search import INDEX_FILE, META_FILE

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def find_images(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*") if p.suffix.lower() in IMAGE_EXTS)


def labels_from_path(rel: Path) -> dict:
    parts = rel.parts[:-1]  # folders only
    return {
        "category": parts[0] if len(parts) > 0 else None,
        "style": parts[1] if len(parts) > 1 else None,
    }


def build(image_root: Path, out_dir: Path, embedder, batch_size: int = 64, limit: int | None = None):
    paths = find_images(image_root)
    if not paths:
        raise SystemExit(f"No images found under {image_root}")
    if limit and limit < len(paths):
        random.Random(42).shuffle(paths)
        paths = sorted(paths[:limit])
    print(f"Embedding {len(paths)} images from {image_root} ...")

    vectors, items = [], []
    started = time.perf_counter()
    for start in range(0, len(paths), batch_size):
        batch_paths = paths[start:start + batch_size]
        images, kept = [], []
        for p in batch_paths:
            try:
                images.append(Image.open(p).convert("RGB"))
                kept.append(p)
            except (UnidentifiedImageError, OSError):
                print(f"  skipping unreadable file: {p}")
        if not images:
            continue
        vectors.append(embedder.embed_images(images))
        for p in kept:
            rel = p.relative_to(image_root)
            items.append({"path": rel.as_posix(), **labels_from_path(rel)})
        done = start + len(batch_paths)
        rate = done / (time.perf_counter() - started)
        print(f"  {done}/{len(paths)}  ({rate:.0f} img/s)", end="\r")

    matrix = np.vstack(vectors).astype("float32")
    # Exact inner-product search. Vectors are normalized, so this is cosine similarity.
    # Fast enough for ~100k items; swap for IndexHNSWFlat if you scale up.
    index = faiss.IndexFlatIP(matrix.shape[1])
    index.add(matrix)

    out_dir.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(out_dir / INDEX_FILE))
    (out_dir / META_FILE).write_text(json.dumps({
        "image_root": str(image_root.resolve()),
        "dim": int(matrix.shape[1]),
        "items": items,
    }))
    print(f"\nSaved {index.ntotal} vectors to {out_dir} in {time.perf_counter() - started:.0f}s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", type=Path, default=Path("data/images"))
    ap.add_argument("--out", type=Path, default=Path("data/index"))
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--limit", type=int, default=None, help="Only index N random images (for a quick first run)")
    args = ap.parse_args()

    from backend.embedder import ClipEmbedder
    build(args.images, args.out, ClipEmbedder(), args.batch_size, args.limit)


if __name__ == "__main__":
    main()
