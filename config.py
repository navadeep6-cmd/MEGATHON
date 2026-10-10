"""Configuration loading and canonical artifact paths."""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

from .io_utils import ensure_dir

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_config(path: Optional[str] = None) -> Dict[str, Any]:
    """Load config.yaml (or $CATSUGGEST_CONFIG) and make paths absolute."""
    import yaml

    path = path or os.environ.get("CATSUGGEST_CONFIG") or os.path.join(ROOT, "config.yaml")
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    for key in ("data_dir", "artifacts_dir", "reports_dir"):
        value = cfg["paths"][key]
        if not os.path.isabs(value):
            cfg["paths"][key] = os.path.join(ROOT, value)
    return cfg


class Paths:
    """Every file the pipeline reads or writes, in one place."""

    def __init__(self, cfg: Dict[str, Any]):
        data = cfg["paths"]["data_dir"]
        art = cfg["paths"]["artifacts_dir"]
        rep = cfg["paths"]["reports_dir"]
        self.data, self.artifacts, self.reports = data, art, rep
        # raw crawl
        self.raw_dir = os.path.join(data, "raw")
        self.crawl_members = os.path.join(self.raw_dir, "crawl_members.json")
        self.raw_files = os.path.join(self.raw_dir, "files.jsonl")
        self.raw_parents = os.path.join(self.raw_dir, "category_parents.json")
        self.raw_redirects = os.path.join(self.raw_dir, "category_redirects.json")
        self.images = os.path.join(data, "images")
        # processed dataset
        self.processed = os.path.join(data, "processed")
        self.dataset = os.path.join(self.processed, "dataset.jsonl")
        self.vocab = os.path.join(self.processed, "categories.json")
        # embeddings
        self.emb_dir = os.path.join(art, "embeddings")
        self.encoder_info = os.path.join(art, "encoder.json")
        self.protos = os.path.join(self.emb_dir, "category_protos.npy")
        self.dedup_dropped = os.path.join(self.emb_dir, "dedup_dropped.json")
        # models
        self.head = os.path.join(art, "head.npz")
        self.store_dir = os.path.join(art, "categories")
        self.memory_eval = os.path.join(art, "memory_eval")
        self.memory_serve = os.path.join(art, "memory_serve")
        self.serve_lock = os.path.join(art, "memory_serve.lock")  # PID of the single writer
        self.fusion_dir = os.path.join(art, "fusion")
        # feedback / continual learning
        self.feedback_dir = os.path.join(data, "feedback")
        self.feedback_log = os.path.join(self.feedback_dir, "feedback.jsonl")
        self.query_cache = os.path.join(self.feedback_dir, "query_cache")
        self.feedback_labels = os.path.join(self.feedback_dir, "feedback_labels.jsonl")
        self.feedback_state = os.path.join(self.feedback_dir, "applied_offset.json")
        self.live_suggestions = os.path.join(data, "live", "suggestions.jsonl")

    def emb(self, kind: str, split: str) -> str:
        """kind: 'img', 'meta' or 'img_aug0', 'img_aug1', ..."""
        return os.path.join(self.emb_dir, "%s_%s.npy" % (kind, split))

    def emb_ids(self, split: str) -> str:
        return os.path.join(self.emb_dir, "ids_%s.json" % split)

    def fusion(self, kind: str) -> str:
        return os.path.join(self.fusion_dir, "fusion_%s.joblib" % kind)

    def report(self, name: str) -> str:
        return os.path.join(self.reports, name)

    def image_path(self, pageid) -> str:
        return os.path.join(self.images, "%s.jpg" % pageid)

    def ensure(self) -> "Paths":
        for d in (self.raw_dir, self.images, self.processed, self.emb_dir, self.store_dir,
                  self.fusion_dir, self.feedback_dir, self.query_cache, self.reports,
                  os.path.dirname(self.live_suggestions)):
            ensure_dir(d)
        return self
