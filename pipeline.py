"""Shared helpers for the offline scripts (fit, evaluate, robustness, benchmark)."""

from __future__ import annotations

import logging
import os
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from .config import Paths
from .head import LinearHead
from .io_utils import read_json, read_jsonl, setup_logging  # noqa: F401  (re-exported for scripts)
from .store import CategoryStore, LabelMemory
from .suggester import CandidateSet, Query, Scorer, ann_kwargs

log = logging.getLogger(__name__)


def load_dataset(P: Paths) -> List[dict]:
    if not os.path.exists(P.dataset):
        raise FileNotFoundError("%s missing - run scripts/s03_build_dataset.py first" % P.dataset)
    return read_jsonl(P.dataset)


def load_split(P: Paths, split: str, drop_dedup: bool = True) -> Tuple[List[dict], np.ndarray, np.ndarray]:
    """Records of a split aligned with their image and metadata-text embeddings.

    Rows whose image failed to decode and (for val/test) rows flagged as
    embedding near-duplicates of training images are removed.
    """
    ids = read_json(P.emb_ids(split))
    Z = np.load(P.emb("img", split))
    Zm = np.load(P.emb("meta", split))
    ok_path = P.emb("ok", split)
    ok = np.load(ok_path) if os.path.exists(ok_path) else np.ones(len(ids), bool)
    by_id = {str(r["id"]): r for r in load_dataset(P) if r["split"] == split}
    dropped = set()
    if drop_dedup and split != "train" and os.path.exists(P.dedup_dropped):
        dropped = set(str(x) for x in read_json(P.dedup_dropped).get(split, []))
    keep = [i for i, rid in enumerate(ids) if ok[i] and str(rid) not in dropped and str(rid) in by_id]
    records = [by_id[str(ids[i])] for i in keep]
    return records, Z[keep], Zm[keep]


def truths_for(records: Sequence[dict], store: CategoryStore) -> List[Set[int]]:
    return [set(store.id_of[c] for c in r["labels"] if c in store.id_of) for r in records]


def build_scorer(cfg: Dict, memory: str = "eval") -> Tuple[Scorer, CategoryStore, LabelMemory, Optional[LinearHead]]:
    P = Paths(cfg)
    store = CategoryStore.load(P.store_dir)
    mem_dir = P.memory_eval if memory == "eval" else P.memory_serve
    mem = LabelMemory.load(mem_dir, cfg["knn"]["index"], **ann_kwargs(cfg))
    head = LinearHead.load(P.head) if os.path.exists(P.head) else None
    return Scorer(store, mem, head, cfg), store, mem, head


def score_in_batches(scorer: Scorer, Z: np.ndarray, Zm: np.ndarray, queries: Sequence[Query],
                     batch: int = 256, progress: bool = True) -> List[CandidateSet]:
    out: List[CandidateSet] = []
    starts = range(0, Z.shape[0], batch)
    if progress:
        from tqdm import tqdm
        starts = tqdm(starts, desc="score", unit="batch")
    for s in starts:
        out.extend(scorer.score(Z[s:s + batch], Zm[s:s + batch], queries[s:s + batch]))
    return out
