"""Loads the FAISS index + metadata and answers nearest-neighbor queries."""
from __future__ import annotations

import json
from pathlib import Path

import faiss
import numpy as np

INDEX_FILE = "index.faiss"
META_FILE = "metadata.json"


class Searcher:
    def __init__(self, index_dir: str | Path, embedder):
        index_dir = Path(index_dir)
        self.index = faiss.read_index(str(index_dir / INDEX_FILE))
        meta = json.loads((index_dir / META_FILE).read_text())
        self.items: list[dict] = meta["items"]
        self.image_root = Path(meta["image_root"])
        self.embedder = embedder

    def _search(self, vecs: np.ndarray, k: int, exclude: int | None = None) -> list[dict]:
        # Ask for one extra so we can drop the query item itself in "more like this".
        scores, ids = self.index.search(vecs.astype("float32"), k + 1)
        results = []
        for score, i in zip(scores[0], ids[0]):
            if i < 0 or i == exclude:
                continue
            results.append({**self.items[i], "id": int(i), "score": float(score)})
        return results[:k]

    def by_text(self, query: str, k: int = 20) -> list[dict]:
        return self._search(self.embedder.embed_text([query]), k)

    def by_image(self, image, k: int = 20) -> list[dict]:
        return self._search(self.embedder.embed_images([image]), k)

    def similar_to(self, item_id: int, k: int = 20) -> list[dict]:
        vec = self.index.reconstruct(item_id).reshape(1, -1)
        return self._search(vec, k, exclude=item_id)
