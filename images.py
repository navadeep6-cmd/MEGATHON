"""Image utilities: loading, perceptual hashing and quality degradations.

Degradations are used twice: as training augmentation for the supervised
head (robustness to low quality uploads) and as fixed corruptions in the
robustness evaluation.
"""

from __future__ import annotations

import io
import random
from typing import Callable, Dict, Optional

from PIL import Image, ImageFile, ImageFilter

ImageFile.LOAD_TRUNCATED_IMAGES = True
_RS = Image.Resampling


def load_rgb(path: str) -> Image.Image:
    with Image.open(path) as im:
        return im.convert("RGB")


def to_rgb_jpeg_bytes(data: bytes, quality: int = 92) -> bytes:
    """Decode any supported image bytes and re-encode as RGB JPEG (validates the file)."""
    with Image.open(io.BytesIO(data)) as im:
        im.load()
        if im.mode in ("RGBA", "LA") or (im.mode == "P" and "transparency" in im.info):
            rgba = im.convert("RGBA")
            background = Image.new("RGB", rgba.size, (255, 255, 255))
            background.paste(rgba, mask=rgba.split()[-1])
            rgb = background
        else:
            rgb = im.convert("RGB")
    out = io.BytesIO()
    rgb.save(out, format="JPEG", quality=quality)
    return out.getvalue()


def dhash(img: Image.Image, size: int = 8) -> int:
    """64-bit difference hash: compare horizontally adjacent pixels of a
    (size+1) x size grayscale thumbnail (algorithm by N. Krawetz)."""
    small = img.convert("L").resize((size + 1, size), _RS.BILINEAR)
    px = list(small.getdata())
    width = size + 1
    bits = 0
    for row in range(size):
        base = row * width
        for col in range(size):
            bits = (bits << 1) | (1 if px[base + col] > px[base + col + 1] else 0)
    return bits


# -- degradations -------------------------------------------------------------

def gaussian_blur(img: Image.Image, radius: float) -> Image.Image:
    return img.filter(ImageFilter.GaussianBlur(radius=radius))


def jpeg_reencode(img: Image.Image, quality: int) -> Image.Image:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=int(quality))
    buf.seek(0)
    with Image.open(buf) as im:
        return im.convert("RGB")


def down_up(img: Image.Image, long_side: int) -> Image.Image:
    """Downscale so the long side is long_side px, then upscale back (loss of detail)."""
    w, h = img.size
    scale = float(long_side) / max(w, h)
    if scale >= 1.0:
        return img
    small = img.resize((max(1, int(round(w * scale))), max(1, int(round(h * scale)))), _RS.BILINEAR)
    return small.resize((w, h), _RS.BICUBIC)


def center_crop(img: Image.Image, area_frac: float) -> Image.Image:
    w, h = img.size
    side = area_frac ** 0.5
    cw, ch = max(1, int(w * side)), max(1, int(h * side))
    left, top = (w - cw) // 2, (h - ch) // 2
    return img.crop((left, top, left + cw, top + ch))


def random_resized_crop(img: Image.Image, rng: random.Random, min_area: float = 0.4) -> Image.Image:
    w, h = img.size
    area = rng.uniform(min_area, 1.0)
    side = area ** 0.5
    cw, ch = max(1, int(w * side)), max(1, int(h * side))
    left = rng.randint(0, w - cw)
    top = rng.randint(0, h - ch)
    return img.crop((left, top, left + cw, top + ch))


def random_degrade(img: Image.Image, rng: random.Random) -> Image.Image:
    """Training augmentation: 1-2 random degradations (blur, JPEG, down/up-scale, crop)."""
    ops = rng.sample(["blur", "jpeg", "downscale", "crop"], k=rng.choice([1, 2]))
    for op in ops:
        if op == "blur":
            img = gaussian_blur(img, rng.uniform(0.5, 3.0))
        elif op == "jpeg":
            img = jpeg_reencode(img, rng.randint(10, 60))
        elif op == "downscale":
            img = down_up(img, rng.randint(32, 128))
        else:
            img = random_resized_crop(img, rng)
    return img


CORRUPTIONS: Dict[str, Optional[Callable[[Image.Image], Image.Image]]] = {
    "clean": None,
    "blur_r2": lambda im: gaussian_blur(im, 2.0),
    "blur_r4": lambda im: gaussian_blur(im, 4.0),
    "jpeg_q20": lambda im: jpeg_reencode(im, 20),
    "jpeg_q5": lambda im: jpeg_reencode(im, 5),
    "lowres_112": lambda im: down_up(im, 112),
    "lowres_48": lambda im: down_up(im, 48),
    "crop_50pct": lambda im: center_crop(im, 0.5),
}


class Corruption:
    """Picklable transform wrapper (DataLoader workers need picklable callables)."""

    def __init__(self, name: Optional[str] = None, augment: bool = False):
        if name is not None and name not in CORRUPTIONS:
            raise ValueError("unknown corruption %r" % name)
        self.name = name
        self.augment = augment

    def __call__(self, img: Image.Image, rng: random.Random) -> Image.Image:
        if self.augment:
            return random_degrade(img, rng)
        fn = CORRUPTIONS.get(self.name) if self.name else None
        return fn(img) if fn is not None else img
