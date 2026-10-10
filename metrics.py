"""Ranking metrics for extreme multi-label suggestion (standard library only).

Sample-wise (per image, user-facing quality):
    P@k = |Yhat_k ∩ Y| / k,  R@k = |Yhat_k ∩ Y| / |Y|,  F1@k = 2PR / (P + R),
    AP  = mean over true labels of precision at that label's rank
          (true labels never retrieved contribute 0).
Macro (per category, tail behaviour):
    AP_c over images ranked by s(c | x); P/R/F1@k_c from top-k predictions.
Hierarchy-aware precision: a predicted ancestor/descendant of a true label
counts as partially correct.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Set


def precision_recall_at_k(ranked: Sequence, truth: Set, k: int):
    if not truth:
        return 0.0, 0.0
    hits = sum(1 for c in ranked[:k] if c in truth)
    return hits / float(k), hits / float(len(truth))


def f1(p: float, r: float) -> float:
    return 0.0 if p + r == 0 else 2.0 * p * r / (p + r)


def average_precision(ranked: Sequence, truth: Set) -> float:
    if not truth:
        return 0.0
    hits = 0
    total = 0.0
    for rank, c in enumerate(ranked, start=1):
        if c in truth:
            hits += 1
            total += hits / float(rank)
    return total / float(len(truth))


def sample_metrics(rankings: Sequence[Sequence], truths: Sequence[Set], ks: Iterable[int]) -> Dict[str, float]:
    """Mean over images of P@k, R@k, F1@k and AP (images with empty truth skipped)."""
    ks = list(ks)
    sums: Dict[str, float] = defaultdict(float)
    n = 0
    for ranked, truth in zip(rankings, truths):
        if not truth:
            continue
        n += 1
        for k in ks:
            p, r = precision_recall_at_k(ranked, truth, k)
            sums["P@%d" % k] += p
            sums["R@%d" % k] += r
            sums["F1@%d" % k] += f1(p, r)
        sums["mAP"] += average_precision(ranked, truth)
    out = {key: (val / n if n else 0.0) for key, val in sums.items()}
    out["n_images"] = n
    return out


def macro_metrics(
    rankings: Sequence[Sequence],
    scores: Sequence[Sequence[float]],
    truths: Sequence[Set],
    ks: Iterable[int],
    categories: Optional[Iterable] = None,
) -> Dict[str, float]:
    """Per-category metrics averaged over categories with >= 1 positive.

    rankings[i] / scores[i]: candidate categories for image i and their scores
    (same order). Categories absent from an image's ranking are treated as
    not retrieved for that image.
    categories: optional subset of categories to average over (e.g. a band).
    """
    ks = list(ks)
    positives: Dict[object, int] = defaultdict(int)
    for truth in truths:
        for c in truth:
            positives[c] += 1
    subset = set(positives) if categories is None else set(categories) & set(positives)
    if not subset:
        return {"n_categories": 0}

    per_cat_scores: Dict[object, List] = defaultdict(list)
    tp = {k: defaultdict(int) for k in ks}
    fp = {k: defaultdict(int) for k in ks}
    for ranked, sc, truth in zip(rankings, scores, truths):
        for pos, (c, s) in enumerate(zip(ranked, sc)):
            if c in subset:
                per_cat_scores[c].append((s, c in truth))
            for k in ks:
                if pos < k and c in subset:
                    if c in truth:
                        tp[k][c] += 1
                    else:
                        fp[k][c] += 1

    out: Dict[str, float] = {}
    ap_sum = 0.0
    for c in subset:
        items = sorted(per_cat_scores.get(c, []), key=lambda t: -t[0])
        hits = 0
        total = 0.0
        for rank, (_, is_pos) in enumerate(items, start=1):
            if is_pos:
                hits += 1
                total += hits / float(rank)
        ap_sum += total / positives[c]
    out["macro_mAP"] = ap_sum / len(subset)
    for k in ks:
        p_sum = r_sum = f_sum = 0.0
        for c in subset:
            t, f_ = tp[k][c], fp[k][c]
            p = t / float(t + f_) if t + f_ else 0.0
            r = t / float(positives[c])
            p_sum += p
            r_sum += r
            f_sum += f1(p, r)
        out["macro_P@%d" % k] = p_sum / len(subset)
        out["macro_R@%d" % k] = r_sum / len(subset)
        out["macro_F1@%d" % k] = f_sum / len(subset)
    out["n_categories"] = len(subset)
    return out


def hierarchical_precision_at_k(
    rankings: Sequence[Sequence],
    truths: Sequence[Set],
    k: int,
    related: Callable[[object, object], bool],
    partial_credit: float = 0.5,
) -> float:
    """Precision@k where a prediction related (ancestor/descendant) to a true
    label earns partial_credit instead of 0."""
    total = 0.0
    n = 0
    for ranked, truth in zip(rankings, truths):
        if not truth:
            continue
        n += 1
        credit = 0.0
        for c in ranked[:k]:
            if c in truth:
                credit += 1.0
            elif any(related(c, t) for t in truth):
                credit += partial_credit
        total += credit / float(k)
    return total / n if n else 0.0


def candidate_recall(candidates: Sequence[Iterable], truths: Sequence[Set]) -> float:
    """Fraction of true labels that appear anywhere in the candidate set (upper bound)."""
    hit = tot = 0
    for cand, truth in zip(candidates, truths):
        cs = set(cand)
        tot += len(truth)
        hit += sum(1 for t in truth if t in cs)
    return hit / float(tot) if tot else 0.0


def reliability(scores: Sequence[float], outcomes: Sequence[bool], n_bins: int = 10) -> Dict[str, object]:
    """Calibration table and expected calibration error (ECE)."""
    bins = [[0, 0.0, 0] for _ in range(n_bins)]  # count, sum score, positives
    for s, y in zip(scores, outcomes):
        idx = min(n_bins - 1, max(0, int(float(s) * n_bins)))
        bins[idx][0] += 1
        bins[idx][1] += float(s)
        bins[idx][2] += 1 if y else 0
    total = sum(b[0] for b in bins)
    ece = 0.0
    table = []
    for i, (cnt, ssum, pos) in enumerate(bins):
        if cnt == 0:
            continue
        conf = ssum / cnt
        acc = pos / float(cnt)
        ece += (cnt / float(total)) * abs(conf - acc)
        table.append({"bin": "%.1f-%.1f" % (i / n_bins, (i + 1) / n_bins), "n": cnt,
                      "mean_score": round(conf, 4), "accept_rate": round(acc, 4)})
    return {"ece": ece if total else 0.0, "bins": table, "n": total}
