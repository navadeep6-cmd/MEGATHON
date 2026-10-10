"""Per-image scorer primitives (standard library only).

These are the label-voting parts of the component scorers; the matrix
products (image/text similarities, head logits) live in suggester.py.
"""

from __future__ import annotations

import math
from typing import Dict, FrozenSet, Iterable, Optional, Sequence, Tuple


def knn_vote(
    sims: Sequence[float],
    neighbour_labels: Sequence[Iterable[int]],
    tau: float,
) -> Dict[int, Tuple[float, float, int]]:
    """Similarity-weighted k-NN label vote.

    s_knn(c|x) = sum_i exp(z_x.z_i / tau) [c in Y_i] / sum_i exp(z_x.z_i / tau)
    Returns {category: (s_knn, max similarity among voters, number of voters)}.
    The max similarity is subtracted before exponentiating for stability
    (it cancels in the ratio).
    """
    if len(sims) == 0:
        return {}
    smax = max(sims)
    weights = [math.exp((s - smax) / tau) for s in sims]
    den = sum(weights)
    acc: Dict[int, list] = {}
    for w, s, labels in zip(weights, sims, neighbour_labels):
        for c in set(labels):
            entry = acc.get(c)
            if entry is None:
                acc[c] = [w, s, 1]
            else:
                entry[0] += w
                entry[1] = max(entry[1], s)
                entry[2] += 1
    return {c: (v[0] / den, v[1], v[2]) for c, v in acc.items()}


def geo_vote(
    distances_km: Sequence[float],
    neighbour_labels: Sequence[Iterable[int]],
    sigma_km: float,
) -> Dict[int, float]:
    """Distance-weighted vote of nearby geotagged labelled images (weights exp(-d/sigma))."""
    if len(distances_km) == 0:
        return {}
    weights = [math.exp(-max(0.0, d) / sigma_km) for d in distances_km]
    den = sum(weights)
    if den <= 0:
        return {}
    acc: Dict[int, float] = {}
    for w, labels in zip(weights, neighbour_labels):
        for c in set(labels):
            acc[c] = acc.get(c, 0.0) + w
    return {c: v / den for c, v in acc.items()}


def lexical_overlap(category_tokens: FrozenSet[str], query_tokens: FrozenSet[str]) -> float:
    """Fraction of a category's content tokens present in the file title/description."""
    if not category_tokens:
        return 0.0
    return len(category_tokens & query_tokens) / float(len(category_tokens))


def year_features(category_year: Optional[int], query_year: Optional[int]) -> Tuple[float, float]:
    """(match, mismatch) indicators for dated categories such as '2019 in Paris'."""
    if category_year is None or query_year is None:
        return 0.0, 0.0
    return (1.0, 0.0) if category_year == query_year else (0.0, 1.0)


def describe_signals(sig: Dict[str, float], k_neighbours: int) -> str:
    """Human-readable explanation of why a category was suggested."""
    parts = []
    voters = int(round(sig.get("knn_frac", 0.0) * k_neighbours))
    if voters > 0:
        parts.append("%d of %d most similar images use it (best match %.2f)"
                     % (voters, k_neighbours, sig.get("knn_maxsim", 0.0)))
    if sig.get("lex", 0.0) >= 0.5:
        parts.append("title/description mentions it")
    if sig.get("geo", 0.0) > 0.0:
        parts.append("used by nearby geotagged photos")
    if sig.get("year_match", 0.0) > 0:
        parts.append("date matches")
    if sig.get("in_head", 0.0) > 0 and sig.get("cls", 0.0) >= 0.3:
        parts.append("classifier confidence %.2f" % sig["cls"])
    if sig.get("txt_rel", -1.0) >= -0.01:
        parts.append("strong image-to-name match")
    if sig.get("year_mismatch", 0.0) > 0:
        parts.append("note: date differs from category year")
    return "; ".join(parts) if parts else "weak combined evidence"
