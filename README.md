# Decor Lens

**Visual "more like this" search for home decor.** Upload a photo of a chair or a room, or type a style like *"cozy boho bedroom,"* and get a grid of visually similar furniture.

<!-- Add a demo GIF here once it's running: ![demo](docs/demo.gif) -->
<!-- Live demo: https://... -->

## How it works

```
image or text ──► CLIP ──► 512-d vector ──► FAISS nearest-neighbor search ──► top-k similar images
```

1. **Embed:** CLIP (ViT-B/32) maps every image to a vector. Similar-looking images land close together, and because CLIP was trained on image–caption pairs, text queries land in the same space.
2. **Index:** All vectors go into a FAISS inner-product index. Vectors are L2-normalized, so inner product = cosine similarity.
3. **Search:** A query (uploaded photo, text, or an existing item) is embedded and matched against the index in milliseconds.
4. **Display:** A React masonry grid. Click any result for "more like this."

## Results

<!-- Fill in after running scripts/evaluate.py -->
| Metric | Value |
|---|---|
| Index size | — images |
| Precision@10 (same style) | —% |
| Precision@10 (same category) | —% |
| Search latency p50 / p95 | — ms / — ms |

**What I tried:** <!-- e.g. ViT-B/32 vs ViT-L/14, what changed and by how much -->

## Run it locally

**1. Get the data.** Download the [Bonn Furniture Styles dataset](https://www.kaggle.com/datasets/) from Kaggle and unzip it so images sit at `data/images/<category>/<style>/*.jpg`. Any folder of images works; folder names become labels.

**2. Backend**
```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Start with 5k images to check everything works (a few minutes on CPU), then run on all of them
python -m scripts.build_index --images data/images --limit 5000
python -m scripts.evaluate --label style --k 10
uvicorn backend.app:app --reload
```

**3. Frontend**
```bash
cd frontend
npm install
npm run dev        # http://localhost:5173
```

## API

| Endpoint | Description |
|---|---|
| `GET /search/text?q=boho+bedroom&k=20` | Text-to-image search |
| `POST /search/image` (multipart `file`) | Search by uploaded photo |
| `GET /similar/{id}` | "More like this" for an indexed item |
| `GET /images/{path}` | Serves dataset images |

## Tests

```bash
pip install -r requirements-dev.txt
pytest -q
```
Tests use a small fake embedder, so they run in under a second without downloading CLIP. They cover indexing (including skipping corrupt files), search, evaluation, and every API endpoint. CI runs them on every push.

## How I used AI tools

<!-- Be specific and honest: what you asked an AI assistant for, what you changed, what it got wrong and how you caught it. -->

## Next steps
- Crop a single object out of a room photo and search for just that ("shop the look")
- Switch to `IndexHNSWFlat` for sub-linear search at 1M+ images
- Filter results by category and style
