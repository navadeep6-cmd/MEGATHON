"""Approximate nearest-neighbour index over L2-normalised embeddings.

Uses FAISS when installed (Flat / HNSW / IVF-PQ, inner-product metric) and
falls back to exact numpy search otherwise, so the pipeline always runs.
Row ids are sequential: row i of the index is row i of the label memory.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Optional, Tuple

import numpy as np

log = logging.getLogger(__name__)

try:  # optional dependency
    import faiss  # type: ignore
except ImportError:  # pragma: no cover - depends on environment
    faiss = None


def _f32(x: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(x, dtype=np.float32)


class AnnIndex:
    def __init__(self, dim: int, kind: str = "flat", hnsw_m: int = 32, ivf_nlist: int = 1024,
                 pq_m: int = 64, nprobe: int = 16, ef_search: int = 128):
        self.dim = int(dim)
        self.kind = kind
        self.hnsw_m, self.ivf_nlist, self.pq_m = hnsw_m, ivf_nlist, pq_m
        self.nprobe, self.ef_search = nprobe, ef_search
        self._index = None                    # faiss index
        self._mat: Optional[np.ndarray] = None  # numpy fallback matrix
        self.backend = "faiss" if faiss is not None else "numpy"
        if self.backend == "numpy" and kind != "flat":
            log.warning("faiss not installed: using exact numpy search instead of %s", kind)

    # -- construction -------------------------------------------------------
    @property
    def ntotal(self) -> int:
        if self._index is not None:
            return int(self._index.ntotal)
        return 0 if self._mat is None else int(self._mat.shape[0])

    def _create_faiss(self, first_batch: np.ndarray):
        kind = self.kind
        n = first_batch.shape[0]
        if kind == "ivfpq":
            nlist = max(1, min(self.ivf_nlist, n // 39))
            if n < 10000 or self.dim % self.pq_m != 0:
                log.warning("ivfpq needs >=10k vectors and dim %% pq_m == 0; using flat index")
                kind = "flat"
            else:
                index = faiss.index_factory(self.dim, "IVF%d,PQ%d" % (nlist, self.pq_m),
                                            faiss.METRIC_INNER_PRODUCT)
                index.train(first_batch)
                faiss.extract_index_ivf(index).nprobe = self.nprobe
                self.kind = kind
                return index
        if kind == "hnsw":
            index = faiss.index_factory(self.dim, "HNSW%d" % self.hnsw_m, faiss.METRIC_INNER_PRODUCT)
            index.hnsw.efConstruction = 200
            index.hnsw.efSearch = self.ef_search
            self.kind = kind
            return index
        self.kind = "flat"
        return faiss.IndexFlatIP(self.dim)

    def add(self, vectors: np.ndarray) -> None:
        x = _f32(vectors)
        if x.ndim != 2 or x.shape[1] != self.dim:
            raise ValueError("expected (n, %d) vectors, got %s" % (self.dim, x.shape))
        if x.shape[0] == 0:
            return
        if self.backend == "faiss":
            if self._index is None:
                self._index = self._create_faiss(x)
            self._index.add(x)
        else:
            self._mat = x.copy() if self._mat is None else np.vstack([self._mat, x])

    # -- search -------------------------------------------------------------
    def search(self, queries: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """Return (similarities, row ids), shape (nq, k'); k' = min(k, ntotal); missing = -1."""
        q = _f32(queries)
        if q.ndim == 1:
            q = q[None, :]
        k = int(min(k, self.ntotal))
        if k <= 0:
            return np.zeros((q.shape[0], 0), np.float32), np.zeros((q.shape[0], 0), np.int64)
        if self.backend == "faiss":
            sims, idx = self._index.search(q, k)
            return sims.astype(np.float32), idx.astype(np.int64)
        return self._numpy_search(q, k)

    def _numpy_search(self, q: np.ndarray, k: int, chunk: int = 1024):
        sims_out = np.empty((q.shape[0], k), np.float32)
        idx_out = np.empty((q.shape[0], k), np.int64)
        for start in range(0, q.shape[0], chunk):
            s = q[start:start + chunk] @ self._mat.T
            if k < s.shape[1]:
                part = np.argpartition(-s, k - 1, axis=1)[:, :k]
            else:
                part = np.tile(np.arange(s.shape[1]), (s.shape[0], 1))
            part_s = np.take_along_axis(s, part, axis=1)
            order = np.argsort(-part_s, axis=1)
            idx_out[start:start + chunk] = np.take_along_axis(part, order, axis=1)
            sims_out[start:start + chunk] = np.take_along_axis(part_s, order, axis=1)
        return sims_out, idx_out

    # -- persistence --------------------------------------------------------
    def nbytes(self) -> int:
        if self._index is not None:
            return int(faiss.serialize_index(self._index).nbytes)
        return 0 if self._mat is None else int(self._mat.nbytes)

    def save(self, directory: str) -> None:
        os.makedirs(directory, exist_ok=True)
        for stale in ("index.faiss", "index.npy"):
            path = os.path.join(directory, stale)
            if os.path.exists(path):
                os.remove(path)
        if self._index is not None:
            faiss.write_index(self._index, os.path.join(directory, "index.faiss"))
        elif self._mat is not None:
            np.save(os.path.join(directory, "index.npy"), self._mat)
        with open(os.path.join(directory, "ann.json"), "w", encoding="utf-8") as f:
            json.dump({"kind": self.kind, "dim": self.dim, "backend": self.backend}, f)

    @classmethod
    def load(cls, directory: str, dim: int, kind: str = "flat", **kwargs) -> "AnnIndex":
        meta_path = os.path.join(directory, "ann.json")
        if os.path.exists(meta_path):
            with open(meta_path, encoding="utf-8") as f:
                kind = json.load(f).get("kind", kind)  # the kind actually built (may have fallen back)
        obj = cls(dim, kind, **kwargs)
        fpath = os.path.join(directory, "index.faiss")
        npath = os.path.join(directory, "index.npy")
        if os.path.exists(fpath):
            if faiss is None:
                raise RuntimeError("index was built with faiss but faiss is not installed")
            obj._index = faiss.read_index(fpath)
            obj.backend = "faiss"
            if obj.kind == "hnsw" and hasattr(obj._index, "hnsw"):
                obj._index.hnsw.efSearch = obj.ef_search
            if obj.kind == "ivfpq":
                faiss.extract_index_ivf(obj._index).nprobe = obj.nprobe
        elif os.path.exists(npath):
            mat = _f32(np.load(npath))
            if faiss is not None:
                obj._index = obj._create_faiss(mat)
                obj._index.add(mat)
            else:
                obj._mat = mat
        return obj
