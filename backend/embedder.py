"""CLIP embedder: turns images and text into vectors in the same space.

Similar-looking images (and text that describes them) end up close together,
which is what makes "more like this" and text search work.
"""
from __future__ import annotations

import numpy as np
from PIL import Image

MODEL_NAME = "ViT-B-32"
PRETRAINED = "laion2b_s34b_b79k"


class ClipEmbedder:
    def __init__(self, model_name: str = MODEL_NAME, pretrained: str = PRETRAINED):
        # Imported here so the rest of the app (and tests) can load without torch.
        import open_clip
        import torch

        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else (
            "mps" if torch.backends.mps.is_available() else "cpu"
        )
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        self.model = self.model.to(self.device).eval()
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.dim = self.model.visual.output_dim

    def embed_images(self, images: list[Image.Image]) -> np.ndarray:
        batch = self.torch.stack([self.preprocess(im.convert("RGB")) for im in images])
        with self.torch.no_grad():
            feats = self.model.encode_image(batch.to(self.device))
        return _normalize(feats.cpu().numpy())

    def embed_text(self, texts: list[str]) -> np.ndarray:
        tokens = self.tokenizer(texts).to(self.device)
        with self.torch.no_grad():
            feats = self.model.encode_text(tokens)
        return _normalize(feats.cpu().numpy())


def _normalize(x: np.ndarray) -> np.ndarray:
    """L2-normalize so inner product == cosine similarity."""
    x = x.astype("float32")
    return x / np.linalg.norm(x, axis=1, keepdims=True).clip(min=1e-12)
