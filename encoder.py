"""Open vision-language encoder f_theta (OpenCLIP) producing L2-normalised embeddings.

Images are resized/cropped to the encoder's fixed input size by the model's
own preprocessing transform, so any upload resolution is accepted.
"""

from __future__ import annotations

import logging
import random
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from .images import load_rgb

log = logging.getLogger(__name__)

PROMPT_TEMPLATES = (
    "{}",
    "a photo of {}.",
    "a photo of {}, from Wikimedia Commons.",
    "an image in the category: {}.",
)


def pick_device(name: str = "auto") -> str:
    if name != "auto":
        return name
    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"


class _ImageFiles(torch.utils.data.Dataset):
    """Loads image files, applies an optional PIL transform, then the encoder preprocessing."""

    def __init__(self, paths: Sequence[str], preprocess: Callable, transform: Optional[Callable], seed: int):
        self.paths = list(paths)
        self.preprocess = preprocess
        self.transform = transform
        self.seed = seed

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int):
        try:
            img = load_rgb(self.paths[i])
            if self.transform is not None:
                img = self.transform(img, random.Random(self.seed * 1000003 + i))
            return self.preprocess(img), True
        except Exception:  # unreadable file: return a blank image and flag it
            return self.preprocess(Image.new("RGB", (224, 224))), False


class Encoder:
    def __init__(self, model_name: str, pretrained: str, device: str = "auto",
                 batch_size: int = 64, num_workers: int = 0):
        import open_clip

        self.model_name, self.pretrained = model_name, pretrained
        self.device = pick_device(device)
        model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
        self.model = model.to(self.device).eval()
        self.preprocess = preprocess
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.batch_size = int(batch_size)
        self.num_workers = int(num_workers)
        self.dim = int(self.encode_texts(["dimension probe"]).shape[1])
        log.info("encoder %s/%s on %s, dim=%d", model_name, pretrained, self.device, self.dim)

    @torch.no_grad()
    def _encode_image_tensor(self, x: torch.Tensor) -> np.ndarray:
        feats = self.model.encode_image(x.to(self.device))
        return F.normalize(feats.float(), dim=-1).cpu().numpy().astype(np.float32)

    def encode_pil(self, images: Sequence[Image.Image]) -> np.ndarray:
        out: List[np.ndarray] = []
        for start in range(0, len(images), self.batch_size):
            batch = [self.preprocess(im.convert("RGB")) for im in images[start:start + self.batch_size]]
            out.append(self._encode_image_tensor(torch.stack(batch)))
        if not out:
            return np.zeros((0, self.dim), np.float32)
        return np.concatenate(out, axis=0)

    def encode_paths(self, paths: Sequence[str], transform: Optional[Callable] = None,
                     seed: int = 0, progress: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """Encode image files. Returns (embeddings (n, d), ok mask (n,))."""
        ds = _ImageFiles(paths, self.preprocess, transform, seed)
        loader = torch.utils.data.DataLoader(ds, batch_size=self.batch_size, shuffle=False,
                                             num_workers=self.num_workers)
        embs: List[np.ndarray] = []
        oks: List[np.ndarray] = []
        iterator = loader
        if progress:
            from tqdm import tqdm
            iterator = tqdm(loader, desc="encode images", unit="batch")
        for x, ok in iterator:
            embs.append(self._encode_image_tensor(x))
            oks.append(ok.numpy().astype(bool))
        if not embs:
            return np.zeros((0, self.dim), np.float32), np.zeros(0, bool)
        return np.concatenate(embs, axis=0), np.concatenate(oks, axis=0)

    @torch.no_grad()
    def encode_texts(self, texts: Sequence[str], batch_size: int = 256) -> np.ndarray:
        """Encode texts; empty strings map to all-zero vectors (no signal)."""
        texts = list(texts)
        dim = getattr(self, "dim", None)
        out: List[np.ndarray] = []
        for start in range(0, len(texts), batch_size):
            chunk = texts[start:start + batch_size]
            tokens = self.tokenizer([t if t.strip() else "." for t in chunk]).to(self.device)
            feats = F.normalize(self.model.encode_text(tokens).float(), dim=-1).cpu().numpy()
            for j, t in enumerate(chunk):
                if not t.strip():
                    feats[j] = 0.0
            out.append(feats.astype(np.float32))
        if not out:
            return np.zeros((0, dim or 0), np.float32)
        return np.concatenate(out, axis=0)

    def category_prototypes(self, names: Sequence[str], batch_size: int = 256) -> np.ndarray:
        """t_c: mean of normalised prompt embeddings, re-normalised."""
        protos = np.zeros((len(names), 0), np.float32)
        acc = None
        for template in PROMPT_TEMPLATES:
            emb = self.encode_texts([template.format(n) for n in names], batch_size=batch_size)
            acc = emb if acc is None else acc + emb
        if acc is not None:
            norms = np.linalg.norm(acc, axis=1, keepdims=True)
            protos = (acc / np.maximum(norms, 1e-12)).astype(np.float32)
        return protos
