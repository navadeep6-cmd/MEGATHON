"""Candidate generation, feature assembly and the end-to-end Suggester.

Per image:
  1. k-NN vote over labelled image embeddings        -> s_knn  (+ neighbours as explanation)
  2. zero-shot text prototypes  z_x . t_c            -> s_txt
  3. supervised head  sigmoid(w_c . z_x + b_c)       -> s_cls
  4. metadata: text-tower embedding of title/description, lexical overlap,
     year match and a geo vote of nearby geotagged images -> s_meta features
Candidates = union of each scorer's top-N (a few hundred); the fusion model
scores only these, never all of C.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Optional, Sequence, Tuple

import numpy as np

from .config import Paths
from .fusion import FEATURES, FusionModel
from .head import LinearHead
from .hierarchy import CategoryGraph, rerank_with_hierarchy
from .io_utils import ensure_dir, read_json
from .scoring import describe_signals, geo_vote, knn_vote, lexical_overlap, year_features
from .store import CategoryStore, LabelMemory
from .textutil import build_meta_text, clean_file_title, content_tokens, meta_year, strip_html

FI = {name: i for i, name in enumerate(FEATURES)}


@dataclass
class Query:
    title: str = ""
    description: str = ""
    year: Optional[int] = None
    lat: Optional[float] = None
    lon: Optional[float] = None
    tokens: FrozenSet[str] = field(default_factory=frozenset)

    @property
    def text(self) -> str:
        return build_meta_text(self.title, self.description)

    @property
    def has_geo(self) -> bool:
        return (self.lat is not None and self.lon is not None
                and math.isfinite(self.lat) and math.isfinite(self.lon))

    @classmethod
    def from_metadata(cls, title: str = "", description: str = "", date: str = "",
                      lat: Optional[float] = None, lon: Optional[float] = None) -> "Query":
        title_c = clean_file_title(title) if title else ""
        desc = strip_html(description)
        return cls(title_c, desc, meta_year(date), lat, lon, content_tokens(title_c + " " + desc))

    @classmethod
    def from_record(cls, rec: dict) -> "Query":
        title_c = rec.get("title_clean", "")
        desc = rec.get("description", "")
        return cls(title_c, desc, rec.get("year"), rec.get("lat"), rec.get("lon"),
                   content_tokens(title_c + " " + desc))


@dataclass
class CandidateSet:
    cat_ids: np.ndarray                 # (n,) category ids
    X: np.ndarray                       # (n, len(FEATURES)) features
    neighbours: List[Tuple[int, float]]  # (memory row, similarity), best first


def _topn(row: np.ndarray, n: int) -> np.ndarray:
    n = min(int(n), row.shape[0])
    if n <= 0:
        return np.zeros(0, np.int64)
    if n == row.shape[0]:
        return np.arange(n)
    return np.argpartition(-row, n - 1)[:n]


class Scorer:
    def __init__(self, store: CategoryStore, memory: LabelMemory, head: Optional[LinearHead], cfg: Dict):
        self.store, self.memory, self.head = store, memory, head
        self.k = int(cfg["knn"]["k"])
        self.tau = float(cfg["knn"]["tau"])
        self.geo_radius = float(cfg["geo"]["radius_km"])
        self.geo_sigma = float(cfg["geo"]["sigma_km"])
        self.geo_max = int(cfg["geo"]["max_neighbours"])
        self.top_n = int(cfg["candidates"]["per_scorer"])
        self.lex_min = float(cfg["candidates"]["lex_min"])
        self.lex_max_df = int(cfg["candidates"]["lex_max_df"])
        self._head_col: Optional[np.ndarray] = None

    def _head_columns(self) -> np.ndarray:
        C = len(self.store)
        if self._head_col is None or self._head_col.shape[0] != C:
            col = np.full(C, -1, dtype=np.int64)
            if self.head is not None:
                col[self.head.cat_ids] = np.arange(self.head.cat_ids.shape[0])
            self._head_col = col
        return self._head_col

    def score(self, Z: np.ndarray, Zm: Optional[np.ndarray], queries: Sequence[Query]) -> List[CandidateSet]:
        """Z: (B, d) image embeddings; Zm: (B, d) metadata-text embeddings (zero rows = no text)."""
        store, memory = self.store, self.memory
        B = Z.shape[0]
        txt = Z @ store.protos.T
        meta = Zm @ store.protos.T if Zm is not None else None
        meta_present = (np.abs(Zm).sum(axis=1) > 0) if Zm is not None else np.zeros(B, bool)
        probs = self.head.probs(Z) if self.head is not None else None
        sims, idx = memory.search(Z, self.k)
        head_col = self._head_columns()
        log_prior = np.log1p(store.train_freq).astype(np.float32)

        out: List[CandidateSet] = []
        for i in range(B):
            q = queries[i]
            valid = idx[i] >= 0
            rows = idx[i][valid]
            nsims = sims[i][valid]
            votes = knn_vote(nsims.tolist(), [memory.labels[r] for r in rows], self.tau)
            n_voters = max(1, rows.shape[0])

            geo: Dict[int, float] = {}
            if q.has_geo:
                near = memory.geo_neighbours(q.lat, q.lon, self.geo_radius, self.geo_max)
                geo = geo_vote([d for _, d in near], [memory.labels[r] for r, _ in near], self.geo_sigma)

            lex = store.lexical_candidates(q.tokens, self.lex_max_df, self.lex_min, self.top_n)

            cands = set(votes) | set(geo) | set(lex)
            cands.update(_topn(txt[i], self.top_n).tolist())
            if meta is not None and meta_present[i]:
                cands.update(_topn(meta[i], self.top_n).tolist())
            if probs is not None:
                cands.update(self.head.cat_ids[_topn(probs[i], self.top_n)].tolist())
            cand = np.asarray(sorted(cands), dtype=np.int64)

            X = np.zeros((cand.shape[0], len(FEATURES)), np.float32)
            for j, c in enumerate(cand.tolist()):
                v = votes.get(c)
                if v is not None:
                    X[j, FI["knn"]], X[j, FI["knn_maxsim"]], X[j, FI["knn_frac"]] = v[0], v[1], v[2] / n_voters
                X[j, FI["lex"]] = lexical_overlap(store.tokens[c], q.tokens)
                X[j, FI["year_match"]], X[j, FI["year_mismatch"]] = year_features(store.years[c], q.year)
                X[j, FI["geo"]] = geo.get(c, 0.0)
            if probs is not None:
                cols = head_col[cand]
                inh = cols >= 0
                X[inh, FI["cls"]] = probs[i, cols[inh]]
                X[:, FI["in_head"]] = inh.astype(np.float32)
            t = txt[i, cand]
            X[:, FI["txt"]] = t
            X[:, FI["txt_rel"]] = t - txt[i].max()
            if meta is not None and meta_present[i]:
                mrow = meta[i]
                X[:, FI["meta"]] = mrow[cand]
                X[:, FI["meta_rel"]] = mrow[cand] - mrow.max()
            X[:, FI["log_prior"]] = log_prior[cand]
            out.append(CandidateSet(cand, X, list(zip(rows.tolist(), nsims.tolist()))))
        return out


def rank(cs: CandidateSet, scores: np.ndarray) -> List[Tuple[int, float]]:
    """Sort candidates by score (ties broken by image-text similarity)."""
    order = np.lexsort((-cs.X[:, FI["txt"]], -scores))
    return [(int(cs.cat_ids[j]), float(scores[j])) for j in order]


def single_feature_scores(cs: CandidateSet, feature: str) -> np.ndarray:
    return cs.X[:, FI[feature]].copy()


def ann_kwargs(cfg: Dict) -> Dict:
    k = cfg["knn"]
    return {"hnsw_m": k["hnsw_m"], "ivf_nlist": k["ivf_nlist"], "pq_m": k["pq_m"],
            "nprobe": k["nprobe"], "ef_search": k["ef_search"]}


def load_graph(cfg: Dict) -> CategoryGraph:
    P = Paths(cfg)
    parents = read_json(P.raw_parents) if os.path.exists(P.raw_parents) else {}
    return CategoryGraph(parents, max_depth=int(cfg["hierarchy"]["max_depth"]))


class Suggester:
    """End-to-end: image (+ optional metadata) -> ranked, explained category suggestions."""

    def __init__(self, cfg: Dict, memory: str = "serve", load_encoder: bool = True, encoder=None):
        self.cfg = cfg
        self.paths = P = Paths(cfg)
        self.store = CategoryStore.load(P.store_dir)
        mem_dir = P.memory_serve if memory == "serve" else P.memory_eval
        self.memory = LabelMemory.load(mem_dir, cfg["knn"]["index"], **ann_kwargs(cfg))
        self.head = LinearHead.load(P.head) if os.path.exists(P.head) else None
        fusion_path = P.fusion(cfg["fusion"]["model"])
        if not os.path.exists(fusion_path):
            raise FileNotFoundError("%s missing - run scripts/s07_fit_fusion.py first" % fusion_path)
        self.fusion = FusionModel.load(fusion_path)
        self.graph = load_graph(cfg)
        self.scorer = Scorer(self.store, self.memory, self.head, cfg)
        self.encoder = encoder
        if self.encoder is None and load_encoder:
            from .encoder import Encoder

            e = cfg["encoder"]
            self.encoder = Encoder(e["model"], e["pretrained"], e["device"], e["batch_size"], 0)
        self.lock = threading.RLock()

    # -- inference ----------------------------------------------------------
    def rank_embeddings(self, Z: np.ndarray, Zm: np.ndarray, queries: Sequence[Query],
                        top_k: int, use_hierarchy: Optional[bool] = None):
        """Score precomputed embeddings. Returns list of (ranked [(name, score)], CandidateSet)."""
        use_hierarchy = self.cfg["hierarchy"]["enabled"] if use_hierarchy is None else use_hierarchy
        results = []
        for cs in self.scorer.score(Z, Zm, queries):
            ranked_ids = rank(cs, self.fusion.predict(cs.X))
            ranked = [(self.store.names[c], s) for c, s in ranked_ids]
            if use_hierarchy:
                ranked, _ = rerank_with_hierarchy(ranked, self.graph, top_k,
                                                  float(self.cfg["hierarchy"]["min_relative_score"]))
            results.append((ranked, cs))
        return results

    def suggest(self, image, title: str = "", description: str = "", date: str = "",
                lat: Optional[float] = None, lon: Optional[float] = None,
                top_k: Optional[int] = None, cache_query: bool = False) -> Dict:
        if self.encoder is None:
            raise RuntimeError("Suggester was created without an encoder")
        top_k = int(top_k or self.cfg["serve"]["top_k"])
        min_score = float(self.cfg["serve"]["min_score"])
        t0 = time.perf_counter()
        with self.lock:
            q = Query.from_metadata(title, description, date, lat, lon)
            z = self.encoder.encode_pil([image])
            zm = self.encoder.encode_texts([q.text])
            t1 = time.perf_counter()
            (ranked, cs), = self.rank_embeddings(z, zm, [q], top_k)
            col = {int(c): j for j, c in enumerate(cs.cat_ids.tolist())}
            suggestions = []
            for name, score in ranked[:top_k]:
                if score < min_score:
                    continue
                cid = self.store.id_of[name]
                row = cs.X[col[cid]]
                sig = {f: float(row[FI[f]]) for f in FEATURES}
                suggestions.append({
                    "category": name,
                    "score": round(float(score), 4),
                    "type": self.store.types[cid],
                    "why": describe_signals(sig, self.scorer.k),
                    "signals": {k: round(v, 4) for k, v in sig.items()},
                    "top_contributions": [(f, round(v, 3)) for f, v in self.fusion.contributions(row)[:3]],
                })
            neighbours = []
            for r, s in cs.neighbours[:8]:
                neighbours.append({
                    "title": self.memory.titles[r],
                    "thumb_url": self.memory.thumbs[r],
                    "path": self.memory.paths[r],
                    "similarity": round(float(s), 4),
                    "categories": [self.store.names[c] for c in self.memory.labels[r][:6]],
                })
            qid = hashlib.sha1(z.tobytes() + q.text.encode("utf-8")).hexdigest()[:16]
            if cache_query:
                self._cache_query(qid, z[0], q, title)
        t2 = time.perf_counter()
        return {
            "query_id": qid,
            "suggestions": suggestions,
            "neighbours": neighbours,
            "timing_ms": {"encode": round((t1 - t0) * 1000, 1), "score": round((t2 - t1) * 1000, 1),
                          "total": round((t2 - t0) * 1000, 1)},
            "note": "Suggestions for human review only; nothing is edited automatically.",
        }

    # -- continual learning -------------------------------------------------
    def _cache_query(self, qid: str, z: np.ndarray, q: Query, title: str) -> None:
        d = ensure_dir(self.paths.query_cache)
        np.save(os.path.join(d, qid + ".npy"), z.astype(np.float32))
        with open(os.path.join(d, qid + ".json"), "w", encoding="utf-8") as f:
            json.dump({"title": title, "title_clean": q.title, "description": q.description,
                       "year": q.year, "lat": q.lat, "lon": q.lon}, f, ensure_ascii=False)

    def add_category(self, name: str) -> int:
        """New category: usable by the text and lexical scorers immediately (no retraining)."""
        from .textutil import clean_category_name

        name = clean_category_name(name)
        if not name:
            raise ValueError("empty category name")
        with self.lock:
            if name in self.store.id_of:
                return self.store.id_of[name]
            if self.encoder is None:
                raise RuntimeError("an encoder is needed to embed a new category name")
            proto = self.encoder.category_prototypes([name])[0]
            return self.store.add(name, proto)

    def add_labelled_image(self, z: np.ndarray, categories: Sequence[str], title: str = "",
                           thumb: str = "", path: str = "", lat: Optional[float] = None,
                           lon: Optional[float] = None) -> int:
        """Append an accepted image to the k-NN memory (no retraining). Returns its row."""
        with self.lock:
            ids = [self.add_category(c) for c in categories]
            self.memory.add(z.reshape(1, -1).astype(np.float32), [ids], [title], [thumb], [path], [lat], [lon])
            return len(self.memory) - 1

    def add_labels_to_row(self, row: int, categories: Sequence[str]) -> int:
        """Add further accepted categories to an existing memory row. Returns how many were new."""
        with self.lock:
            existing = set(self.memory.labels[row])
            new = [cid for cid in (self.add_category(c) for c in categories) if cid not in existing]
            if new:
                self.memory.labels[row] = sorted(existing | set(new))
            return len(new)

    def save(self) -> None:
        with self.lock:
            self.store.save(self.paths.store_dir)
            self.memory.save(self.paths.memory_serve)
