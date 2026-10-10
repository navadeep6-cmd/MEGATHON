"""Leakage-safe train/val/test splitting and near-duplicate detection.

* Temporal split: train on earlier uploads, test on later ones.
* temporal_grouped (default): every uploader lands in exactly one split
  (ordered by the uploader's median upload time) to stop style leakage.
* Near-duplicates across splits are removed by perceptual hash (dHash) here
  and by embedding cosine similarity after encoding (see scripts/s04_embed.py).
"""

from __future__ import annotations

from typing import Dict, List, Sequence

SPLITS = ("train", "val", "test")


def temporal_split(
    records: Sequence[dict],
    val_frac: float = 0.1,
    test_frac: float = 0.1,
    mode: str = "temporal_grouped",
) -> Dict[str, str]:
    """Return {record id: split}. Records need 'id', 'timestamp' (ISO 8601) and 'uploader'."""
    if not 0 <= val_frac < 1 or not 0 <= test_frac < 1 or val_frac + test_frac >= 1:
        raise ValueError("val_frac and test_frac must be in [0, 1) and sum to < 1")
    n = len(records)
    if n == 0:
        return {}
    train_end = n * (1.0 - val_frac - test_frac)
    val_end = n * (1.0 - test_frac)

    if mode == "temporal":
        ordered = sorted(records, key=lambda r: (r["timestamp"], str(r["id"])))
        out = {}
        for i, rec in enumerate(ordered):
            pos = i + 0.5
            out[str(rec["id"])] = "train" if pos < train_end else ("val" if pos < val_end else "test")
        return out

    if mode != "temporal_grouped":
        raise ValueError("unknown split mode: %r" % mode)

    groups: Dict[str, List[dict]] = {}
    for rec in records:
        key = rec.get("uploader") or ("__anon__%s" % rec["id"])
        groups.setdefault(key, []).append(rec)

    def median_ts(items: List[dict]) -> str:
        stamps = sorted(r["timestamp"] for r in items)
        return stamps[(len(stamps) - 1) // 2]

    ordered_groups = sorted(groups.items(), key=lambda kv: (median_ts(kv[1]), kv[0]))
    out = {}
    cum = 0
    for _, items in ordered_groups:
        mid = cum + len(items) / 2.0
        split = "train" if mid < train_end else ("val" if mid < val_end else "test")
        for rec in items:
            out[str(rec["id"])] = split
        cum += len(items)
    return out


def hamming64(a: int, b: int) -> int:
    """Hamming distance between two 64-bit hashes."""
    return bin((a ^ b) & 0xFFFFFFFFFFFFFFFF).count("1")


def near_duplicate_mask(query: Sequence[int], reference: Sequence[int], max_distance: int) -> List[bool]:
    """Pure-Python reference implementation: True where a query hash is within
    max_distance bits of any reference hash. O(|query| * |reference|)."""
    out = []
    for q in query:
        out.append(any(hamming64(q, r) <= max_distance for r in reference))
    return out


def near_duplicate_mask_np(query: Sequence[int], reference: Sequence[int], max_distance: int, chunk: int = 128):
    """Vectorised version of near_duplicate_mask using a byte popcount table."""
    import numpy as np

    q = np.asarray([int(x) & 0xFFFFFFFFFFFFFFFF for x in query], dtype=np.uint64)
    r = np.asarray([int(x) & 0xFFFFFFFFFFFFFFFF for x in reference], dtype=np.uint64)
    out = np.zeros(len(q), dtype=bool)
    if len(q) == 0 or len(r) == 0:
        return out
    table = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)
    for start in range(0, len(q), chunk):
        x = np.bitwise_xor(q[start:start + chunk, None], r[None, :])  # (c, R) uint64
        bits = table[x.view(np.uint8)].reshape(x.shape[0], x.shape[1], 8).sum(axis=2, dtype=np.int32)
        out[start:start + chunk] = (bits <= max_distance).any(axis=1)
    return out
