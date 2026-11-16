"""End-to-end tests with a tiny fake embedder, so they run in seconds without CLIP.

The fake embedder turns an image into a vector from its average color, so images
in the same "style" folder (same color) should find each other.
Run: pytest -q
"""
import io
import os

import numpy as np
import pytest
from PIL import Image

os.environ["DECOR_LENS_NO_AUTOLOAD"] = "1"

from backend.app import create_app  # noqa: E402
from backend.search import Searcher  # noqa: E402
from scripts.build_index import build  # noqa: E402
from scripts.evaluate import evaluate  # noqa: E402

STYLES = {"Modern": (220, 220, 230), "Rustic": (140, 90, 40), "Boho": (210, 120, 160)}
CATEGORIES = ["chairs", "beds"]


class FakeEmbedder:
    dim = 8

    def embed_images(self, images):
        vecs = []
        for im in images:
            rgb = np.asarray(im.convert("RGB").resize((8, 8)), dtype="float32").mean(axis=(0, 1)) / 255
            vecs.append(np.concatenate([rgb, rgb ** 2, [0.1, 0.1]]))
        return _norm(np.array(vecs))

    def embed_text(self, texts):
        return np.vstack([self.embed_images([Image.new("RGB", (8, 8), STYLES.get(t, (0, 0, 0)))]) for t in texts])


def _norm(x):
    x = x.astype("float32")
    return x / np.linalg.norm(x, axis=1, keepdims=True)


@pytest.fixture(scope="module")
def index_dir(tmp_path_factory):
    root = tmp_path_factory.mktemp("data")
    images = root / "images"
    rng = np.random.default_rng(0)
    for cat in CATEGORIES:
        for style, color in STYLES.items():
            d = images / cat / style
            d.mkdir(parents=True)
            for i in range(6):
                jitter = rng.integers(-10, 10, size=3)
                c = tuple(int(v) for v in np.clip(np.array(color) + jitter, 0, 255))
                Image.new("RGB", (32, 48), c).save(d / f"{i}.jpg")
    (images / "chairs" / "Modern" / "broken.jpg").write_text("not an image")
    out = root / "index"
    build(images, out, FakeEmbedder(), batch_size=7)
    return out


def test_index_skips_broken_files_and_keeps_labels(index_dir):
    s = Searcher(index_dir, FakeEmbedder())
    assert len(s.items) == len(CATEGORIES) * len(STYLES) * 6
    assert {it["style"] for it in s.items} == set(STYLES)
    assert {it["category"] for it in s.items} == set(CATEGORIES)


def test_more_like_this_excludes_itself_and_matches_style(index_dir):
    s = Searcher(index_dir, FakeEmbedder())
    results = s.similar_to(0, k=5)
    assert len(results) == 5
    assert all(r["id"] != 0 for r in results)
    assert all(r["style"] == s.items[0]["style"] for r in results)


def test_evaluate_reports_precision_and_latency(index_dir):
    r = evaluate(index_dir, label="style", k=5, n_queries=20)
    assert r["precision@5"] > 0.9
    assert r["latency_ms_p95"] >= 0
    assert set(r["per_label"]) == set(STYLES)


def test_api_endpoints(index_dir):
    from fastapi.testclient import TestClient

    client = TestClient(create_app(Searcher(index_dir, FakeEmbedder())))
    assert client.get("/health").json()["items"] == 36

    r = client.get("/search/text", params={"q": "Rustic", "k": 4}).json()
    assert [x["style"] for x in r["results"]] == ["Rustic"] * 4
    assert "took_ms" in r

    buf = io.BytesIO()
    Image.new("RGB", (20, 20), STYLES["Boho"]).save(buf, "PNG")
    r = client.post("/search/image", files={"file": ("q.png", buf.getvalue(), "image/png")}).json()
    assert r["results"][0]["style"] == "Boho"

    first = r["results"][0]
    assert client.get(f"/images/{first['path']}").status_code == 200

    assert client.post("/search/image", files={"file": ("x.txt", b"hello", "text/plain")}).status_code == 400
    assert client.get("/similar/9999").status_code == 404
    assert client.get("/search/text", params={"q": ""}).status_code == 422
