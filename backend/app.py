from __future__ import annotations

import io
import os
import time
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError

from .search import Searcher

INDEX_DIR = os.environ.get("INDEX_DIR", "data/index")
MAX_UPLOAD_BYTES = 10 * 1024 * 1024


def create_app(searcher: Searcher | None = None) -> FastAPI:
    if searcher is None:
        from .embedder import ClipEmbedder
        searcher = Searcher(INDEX_DIR, ClipEmbedder())
        # Run one search at startup so the first real user doesn't pay the warm-up cost.
        searcher.by_text("warm up", k=1)

    app = FastAPI(title="Opsis")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=os.environ.get("CORS_ORIGINS", "http://localhost:5173").split(","),
        allow_methods=["*"],
        allow_headers=["*"],
    )
    # Serves the dataset images at /images/<relative path>
    app.mount("/images", StaticFiles(directory=searcher.image_root), name="images")

    def respond(results, started):
        return {"results": results, "took_ms": round((time.perf_counter() - started) * 1000, 1)}

    @app.get("/health")
    def health():
        return {"ok": True, "items": len(searcher.items)}

    @app.get("/search/text")
    def search_text(q: str = Query(..., min_length=1, max_length=200), k: int = Query(20, ge=1, le=100)):
        t = time.perf_counter()
        return respond(searcher.by_text(q, k), t)

    @app.post("/search/image")
    async def search_image(file: UploadFile = File(...), k: int = Query(20, ge=1, le=100)):
        data = await file.read()
        if len(data) > MAX_UPLOAD_BYTES:
            raise HTTPException(413, "Image too large (max 10 MB)")
        try:
            image = Image.open(io.BytesIO(data))
            # CLIP only looks at 224x224 pixels, so shrink big phone photos early.
            # draft() lets JPEGs decode at reduced size, which is much faster.
            image.draft("RGB", (448, 448))
            image.thumbnail((448, 448))
        except (UnidentifiedImageError, OSError):
            raise HTTPException(400, "Could not read that file as an image")
        t = time.perf_counter()
        return respond(searcher.by_image(image, k), t)

    @app.get("/similar/{item_id}")
    def similar(item_id: int, k: int = Query(20, ge=1, le=100)):
        if not 0 <= item_id < len(searcher.items):
            raise HTTPException(404, "Unknown item")
        t = time.perf_counter()
        return respond(searcher.similar_to(item_id, k), t)

    # In the deployed version, the built React app is served from the same server.
    dist = Path(os.environ.get("FRONTEND_DIST", "frontend/dist"))
    if dist.is_dir():
        app.mount("/", StaticFiles(directory=dist, html=True), name="frontend")

    return app


# Built lazily so tests can import create_app without loading CLIP.
app = create_app() if os.environ.get("DECOR_LENS_NO_AUTOLOAD") != "1" else None