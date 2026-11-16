from __future__ import annotations

import argparse
import json
from pathlib import Path

from huggingface_hub import HfApi

# The Space reads its settings from the top of its README.
SPACE_README = """---
title: Opsis
emoji: 🛋️
colorFrom: yellow
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
---

Opsis (Greek for "sight"): visual "more like this" search for furniture, using CLIP embeddings and FAISS.
"""

SKIP = [
    ".venv/*", ".git/*", ".github/*", "tests/*", "**/__pycache__/*", ".pytest_cache/*",
    "frontend/node_modules/*", "frontend/dist/*", "README.md", "*.zip",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--space", required=True, help="e.g. hannay-sen/opsis")
    args = ap.parse_args()

    meta = Path("data/index/metadata.json")
    if not meta.exists():
        raise SystemExit("No index found. Run: python -m scripts.build_index --limit 2000")
    n = len(json.loads(meta.read_text())["items"])
    print(f"Uploading app with {n} indexed images to {args.space} ...")

    api = HfApi()
    api.create_repo(args.space, repo_type="space", space_sdk="docker", exist_ok=True)
    api.upload_folder(
        folder_path=".",
        repo_id=args.space,
        repo_type="space",
        ignore_patterns=SKIP,
        commit_message="Deploy Opsis",
    )
    api.upload_file(
        path_or_fileobj=SPACE_README.encode(),
        path_in_repo="README.md",
        repo_id=args.space,
        repo_type="space",
    )
    print(f"Done. Watch it build at https://huggingface.co/spaces/{args.space}")


if __name__ == "__main__":
    main()