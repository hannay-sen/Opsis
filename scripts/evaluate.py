"""Weekend 3: measure search quality and speed. These numbers go on your resume.

precision@k = of the top k "more like this" results, what share have the same label
as the query image (e.g. same style, or same category). Uses the dataset's folder
labels, so no hand-labeling needed.

Usage:
    python -m scripts.evaluate --label style --k 10 --queries 500
"""
from __future__ import annotations

import argparse
import json
import random
import time
from collections import defaultdict
from pathlib import Path

import faiss
import numpy as np

from backend.search import INDEX_FILE, META_FILE


def evaluate(index_dir: Path, label: str = "style", k: int = 10, n_queries: int = 500, seed: int = 0) -> dict:
    index = faiss.read_index(str(index_dir / INDEX_FILE))
    items = json.loads((index_dir / META_FILE).read_text())["items"]

    candidates = [i for i, it in enumerate(items) if it.get(label)]
    if not candidates:
        raise SystemExit(f"No items have a '{label}' label")
    queries = random.Random(seed).sample(candidates, min(n_queries, len(candidates)))

    per_label = defaultdict(list)
    latencies = []
    for qid in queries:
        vec = index.reconstruct(qid).reshape(1, -1)
        t = time.perf_counter()
        _, ids = index.search(vec, k + 1)
        latencies.append((time.perf_counter() - t) * 1000)
        neighbors = [i for i in ids[0] if i != qid and i >= 0][:k]
        hits = sum(items[i].get(label) == items[qid][label] for i in neighbors)
        per_label[items[qid][label]].append(hits / k)

    all_scores = [s for scores in per_label.values() for s in scores]
    return {
        "label": label,
        "k": k,
        "queries": len(queries),
        "index_size": index.ntotal,
        f"precision@{k}": round(float(np.mean(all_scores)), 4),
        "latency_ms_p50": round(float(np.percentile(latencies, 50)), 2),
        "latency_ms_p95": round(float(np.percentile(latencies, 95)), 2),
        "per_label": {lab: round(float(np.mean(s)), 3) for lab, s in sorted(per_label.items())},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=Path, default=Path("data/index"))
    ap.add_argument("--label", default="style", choices=["style", "category"])
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--queries", type=int, default=500)
    args = ap.parse_args()

    r = evaluate(args.index, args.label, args.k, args.queries)
    precision = r["precision@%d" % r["k"]]
    print(f"\nIndex size: {r['index_size']} images, {r['queries']} queries")
    print(f"precision@{r['k']} (same {r['label']}): {precision:.1%}")
    print(f"search latency: p50 {r['latency_ms_p50']} ms, p95 {r['latency_ms_p95']} ms")
    print(f"\nPer {r['label']} (weakest first):")
    for lab, score in sorted(r["per_label"].items(), key=lambda x: x[1]):
        print(f"  {lab:<20} {score:.1%}")


if __name__ == "__main__":
    main()
