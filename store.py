"""Category vocabulary and labelled-image memory.

CategoryStore: names, text prototypes t_c, training frequencies, lexical
inverted index. New categories can be added at any time (no retraining):
they are immediately reachable by the text-prototype and lexical scorers.

LabelMemory: ANN index of categorised image embeddings plus their labels,
titles and coordinates. Appending rows makes new images / categories
available to the k-NN and geo scorers immediately.
"""

from __future__ import annotations

import json
import os
from collections import defaultdict
from typing import Dict, FrozenSet, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .ann import AnnIndex
from .geo import haversine_km_np
from .labels import category_type
from .textutil import category_tokens, find_year


class CategoryStore:
    def __init__(self, names: Sequence[str], protos: np.ndarray,
                 train_freq: Optional[Sequence[int]] = None, types: Optional[Sequence[str]] = None):
        if len(names) != protos.shape[0]:
            raise ValueError("names and prototypes differ in length")
        self.names: List[str] = list(names)
        self.id_of: Dict[str, int] = {n: i for i, n in enumerate(self.names)}
        self.protos = np.ascontiguousarray(protos, dtype=np.float32)
        n = len(self.names)
        self.train_freq = (np.zeros(n, np.int64) if train_freq is None
                           else np.asarray(train_freq, dtype=np.int64))
        self.types: List[str] = list(types) if types is not None else [category_type(x) for x in self.names]
        self.tokens: List[FrozenSet[str]] = [category_tokens(x) for x in self.names]
        self.years: List[Optional[int]] = [find_year(x) for x in self.names]
        self.token_index: Dict[str, List[int]] = defaultdict(list)
        for i, toks in enumerate(self.tokens):
            for t in toks:
                self.token_index[t].append(i)

    def __len__(self) -> int:
        return len(self.names)

    @property
    def dim(self) -> int:
        return int(self.protos.shape[1])

    def add(self, name: str, proto: np.ndarray, category_kind: Optional[str] = None) -> int:
        """Register a new category (idempotent). Returns its id."""
        if name in self.id_of:
            return self.id_of[name]
        cid = len(self.names)
        self.names.append(name)
        self.id_of[name] = cid
        p = np.asarray(proto, dtype=np.float32).reshape(1, -1)
        p = p / max(float(np.linalg.norm(p)), 1e-12)
        self.protos = np.ascontiguousarray(np.vstack([self.protos, p]))
        self.train_freq = np.concatenate([self.train_freq, np.zeros(1, np.int64)])
        self.types.append(category_kind or category_type(name))
        toks = category_tokens(name)
        self.tokens.append(toks)
        self.years.append(find_year(name))
        for t in toks:
            self.token_index[t].append(cid)
        return cid

    def lexical_candidates(self, query_tokens: FrozenSet[str], max_df: int, min_overlap: float,
                           limit: int) -> Dict[int, float]:
        """Categories whose name tokens appear in the query text (overlap >= min_overlap)."""
        if not query_tokens:
            return {}
        hits: Dict[int, int] = defaultdict(int)
        for tok in query_tokens:
            posting = self.token_index.get(tok)
            if posting and len(posting) <= max_df:
                for cid in posting:
                    hits[cid] += 1
        scored = []
        for cid, h in hits.items():
            denom = len(self.tokens[cid]) or 1
            frac = h / float(denom)
            if frac >= min_overlap:
                scored.append((frac, int(self.train_freq[cid]), cid))
        scored.sort(reverse=True)
        return {cid: frac for frac, _, cid in scored[:limit]}

    def save(self, directory: str) -> None:
        os.makedirs(directory, exist_ok=True)
        np.save(os.path.join(directory, "protos.npy"), self.protos)
        meta = {"names": self.names, "train_freq": [int(x) for x in self.train_freq], "types": self.types}
        with open(os.path.join(directory, "categories.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False)

    @classmethod
    def load(cls, directory: str) -> "CategoryStore":
        with open(os.path.join(directory, "categories.json"), encoding="utf-8") as f:
            meta = json.load(f)
        protos = np.load(os.path.join(directory, "protos.npy"))
        return cls(meta["names"], protos, meta["train_freq"], meta["types"])


class LabelMemory:
    """Labelled image embeddings for k-NN voting, geo voting and explanations."""

    def __init__(self, dim: int, index_kind: str = "flat", **ann_kwargs):
        self.dim = int(dim)
        self.ann = AnnIndex(dim, index_kind, **ann_kwargs)
        self._ann_kwargs = dict(ann_kwargs)
        self.labels: List[List[int]] = []
        self.titles: List[str] = []
        self.thumbs: List[str] = []
        self.paths: List[str] = []
        self._lat: List[float] = []
        self._lon: List[float] = []
        self._geo_cache: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None

    def __len__(self) -> int:
        return len(self.labels)

    def add(self, embeddings: np.ndarray, labels: Sequence[Iterable[int]], titles: Sequence[str],
            thumbs: Optional[Sequence[str]] = None, paths: Optional[Sequence[str]] = None,
            lats: Optional[Sequence[Optional[float]]] = None,
            lons: Optional[Sequence[Optional[float]]] = None) -> None:
        n = len(labels)
        if embeddings.shape[0] != n or len(titles) != n:
            raise ValueError("embeddings, labels and titles must have the same length")
        thumbs = list(thumbs) if thumbs is not None else [""] * n
        paths = list(paths) if paths is not None else [""] * n
        lats = list(lats) if lats is not None else [None] * n
        lons = list(lons) if lons is not None else [None] * n
        self.ann.add(embeddings)
        self.labels.extend([sorted(set(int(c) for c in ls)) for ls in labels])
        self.titles.extend(titles)
        self.thumbs.extend(thumbs)
        self.paths.extend(paths)
        self._lat.extend([float("nan") if v is None else float(v) for v in lats])
        self._lon.extend([float("nan") if v is None else float(v) for v in lons])
        self._geo_cache = None

    def search(self, queries: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        return self.ann.search(queries, k)

    def _geo_arrays(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self._geo_cache is None:
            lat = np.asarray(self._lat, dtype=np.float64)
            lon = np.asarray(self._lon, dtype=np.float64)
            rows = np.nonzero(np.isfinite(lat) & np.isfinite(lon))[0]
            self._geo_cache = (rows, lat[rows], lon[rows])
        return self._geo_cache

    def geo_neighbours(self, lat: float, lon: float, radius_km: float, max_n: int) -> List[Tuple[int, float]]:
        """Labelled rows within radius_km of (lat, lon), nearest first."""
        rows, lats, lons = self._geo_arrays()
        if rows.size == 0:
            return []
        d = haversine_km_np(lat, lon, lats, lons)
        within = np.nonzero(d <= radius_km)[0]
        if within.size == 0:
            return []
        order = within[np.argsort(d[within])][:max_n]
        return [(int(rows[i]), float(d[i])) for i in order]

    def save(self, directory: str) -> None:
        os.makedirs(directory, exist_ok=True)
        self.ann.save(directory)
        flat = np.asarray([c for ls in self.labels for c in ls], dtype=np.int64)
        offsets = np.cumsum([0] + [len(ls) for ls in self.labels]).astype(np.int64)
        np.savez(os.path.join(directory, "labels.npz"), flat=flat, offsets=offsets,
                 lat=np.asarray(self._lat, np.float64), lon=np.asarray(self._lon, np.float64))
        with open(os.path.join(directory, "rows.json"), "w", encoding="utf-8") as f:
            json.dump({"dim": self.dim, "titles": self.titles, "thumbs": self.thumbs,
                       "paths": self.paths}, f, ensure_ascii=False)

    @classmethod
    def load(cls, directory: str, index_kind: str = "flat", **ann_kwargs) -> "LabelMemory":
        with open(os.path.join(directory, "rows.json"), encoding="utf-8") as f:
            rows = json.load(f)
        obj = cls(rows["dim"], index_kind, **ann_kwargs)
        obj.ann = AnnIndex.load(directory, rows["dim"], index_kind, **ann_kwargs)
        data = np.load(os.path.join(directory, "labels.npz"))
        flat, offsets = data["flat"], data["offsets"]
        obj.labels = [flat[offsets[i]:offsets[i + 1]].tolist() for i in range(len(offsets) - 1)]
        obj.titles, obj.thumbs, obj.paths = rows["titles"], rows["thumbs"], rows["paths"]
        obj._lat = data["lat"].tolist()
        obj._lon = data["lon"].tolist()
        if obj.ann.ntotal != len(obj.labels):
            raise RuntimeError("index has %d rows but %d label rows" % (obj.ann.ntotal, len(obj.labels)))
        return obj
