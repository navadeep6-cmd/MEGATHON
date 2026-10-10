#!/usr/bin/env python3
"""Step 8 - evaluate on the temporal, de-duplicated test split.

Reports, for each system (single scorers, fusion variants, fusion+hierarchy,
image-only):
  * sample-wise P@k, R@k, F1@k, mAP            (user-facing quality)
  * macro (per-category) P/R/F1@k and mAP       (tail behaviour)
  * hierarchy-aware precision@k                 (ancestor/descendant = partial credit)
Stratified for the main system by frequency band (head/torso/tail/zero-shot),
category type (object/scene/event/abstract), label script, region and
original resolution. Also writes a sample of "wrong" suggestions for human
review (strict vs reviewed score, see s12_apply_review.py).

Outputs: reports/metrics.json, reports/review_sample.csv
"""

import argparse
import csv
import logging
import os
import random
import sys
import time
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from catsuggest.config import Paths, load_config  # noqa: E402
from catsuggest.fusion import FusionModel  # noqa: E402
from catsuggest.hierarchy import rerank_with_hierarchy  # noqa: E402
from catsuggest.io_utils import read_json, setup_logging, write_json  # noqa: E402
from catsuggest.metrics import (candidate_recall, hierarchical_precision_at_k, macro_metrics,  # noqa: E402
                                sample_metrics)
from catsuggest.pipeline import build_scorer, load_split, score_in_batches, truths_for  # noqa: E402
from catsuggest.suggester import Query, load_graph, rank, single_feature_scores  # noqa: E402

log = logging.getLogger("s08")

SINGLE_SCORERS = (
    ("knn", "kNN retrieval only"),
    ("txt", "text prototype (zero-shot) only"),
    ("cls", "supervised head only"),
    ("meta", "metadata text only"),
    ("lex", "lexical title match only"),
)


def split_ranked(ranked):
    return [c for c, _ in ranked], [s for _, s in ranked]


def evaluate_system(rankings, scores, truths, ks, graph, store, partial):
    related = lambda a, b: graph.related(store.names[a], store.names[b])  # noqa: E731
    return {
        "sample": sample_metrics(rankings, truths, ks),
        "macro": macro_metrics(rankings, scores, truths, ks),
        "hier_P@k": {"hP@%d" % k: hierarchical_precision_at_k(rankings, truths, k, related, partial) for k in ks},
    }


def fused_rankings(cands, fusion, store, graph, cfg, hierarchy):
    out_r, out_s, suppressed = [], [], 0
    k = int(cfg["eval"]["top_k"])
    ratio = float(cfg["hierarchy"]["min_relative_score"])
    for cs in cands:
        ranked = rank(cs, fusion.predict(cs.X))
        if hierarchy:
            named = [(store.names[c], s) for c, s in ranked]
            named, sup = rerank_with_hierarchy(named, graph, k, ratio)
            suppressed += len(sup)
            ranked = [(store.id_of[n], s) for n, s in named]
        r, s = split_ranked(ranked)
        out_r.append(r)
        out_s.append(s)
    return out_r, out_s, suppressed / max(1, len(cands))


def subset(idx, *lists):
    return [[lst[i] for i in idx] for lst in lists]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    setup_logging()
    cfg = load_config(args.config)
    P = Paths(cfg).ensure()
    ec = cfg["eval"]
    ks = [int(k) for k in ec["ks"]]
    partial = float(ec["hier_partial_credit"])

    scorer, store, _, _ = build_scorer(cfg, memory="eval")
    graph = load_graph(cfg)
    vocab = read_json(P.vocab)
    recs, Z, Zm = load_split(P, "test")
    truths = truths_for(recs, store)
    queries = [Query.from_record(r) for r in recs]
    log.info("test images: %d", len(recs))

    t0 = time.perf_counter()
    cands = score_in_batches(scorer, Z, Zm, queries)
    score_s = time.perf_counter() - t0

    results = {"n_test_images": len(recs), "n_categories": len(store),
               "candidate_recall": candidate_recall([cs.cat_ids.tolist() for cs in cands], truths),
               "mean_candidates_per_image": float(np.mean([len(cs.cat_ids) for cs in cands])),
               "scoring_ms_per_image_excl_encoder": 1000.0 * score_s / max(1, len(recs)),
               "systems": {}}

    for feat, name in SINGLE_SCORERS:
        if feat == "cls" and scorer.head is None:
            continue
        r_s = [split_ranked(rank(cs, single_feature_scores(cs, feat))) for cs in cands]
        rankings, scores = [a for a, _ in r_s], [b for _, b in r_s]
        results["systems"][name] = evaluate_system(rankings, scores, truths, ks, graph, store, partial)
        log.info("%-34s P@1 %.3f  R@10 %.3f  mAP %.3f", name, results["systems"][name]["sample"]["P@1"],
                 results["systems"][name]["sample"].get("R@10", 0.0), results["systems"][name]["sample"]["mAP"])

    main_kind = cfg["fusion"]["model"]
    main_name = main_fusion = None
    main_rankings = main_scores = None
    for kind in ("logreg", "gbdt"):
        path = P.fusion(kind)
        if not os.path.exists(path):
            continue
        fusion = FusionModel.load(path)
        for hierarchy in (False, True):
            name = "fusion (%s)%s" % (kind, " + hierarchy" if hierarchy else "")
            rankings, scores, sup = fused_rankings(cands, fusion, store, graph, cfg, hierarchy)
            res = evaluate_system(rankings, scores, truths, ks, graph, store, partial)
            res["suppressed_ancestors_per_image"] = sup
            results["systems"][name] = res
            log.info("%-34s P@1 %.3f  R@10 %.3f  mAP %.3f", name, res["sample"]["P@1"],
                     res["sample"].get("R@10", 0.0), res["sample"]["mAP"])
            if kind == main_kind and hierarchy == bool(cfg["hierarchy"]["enabled"]):
                main_name, main_rankings, main_scores = name, rankings, scores
                main_fusion = fusion
    if main_name is None:
        raise SystemExit("fusion model %r not found - run s07_fit_fusion.py" % main_kind)
    results["main_system"] = main_name

    # image only (no title/description/date/geotag) with the main fusion model
    empty = [Query() for _ in recs]
    cands_img = score_in_batches(scorer, Z, np.zeros_like(Zm), empty)
    r_img, s_img, _ = fused_rankings(cands_img, main_fusion, store, graph, cfg, bool(cfg["hierarchy"]["enabled"]))
    results["systems"][main_name + " - image only"] = evaluate_system(r_img, s_img, truths, ks, graph, store, partial)

    # stratification of the main system
    meta = {store.id_of[c["name"]]: c for c in vocab if c["name"] in store.id_of}
    for key, field in (("by_band", "band"), ("by_type", "type"), ("by_script", "script")):
        groups = defaultdict(list)
        for cid, c in meta.items():
            groups[c[field]].append(cid)
        results[key] = {g: macro_metrics(main_rankings, main_scores, truths, ks, categories=ids)
                        for g, ids in sorted(groups.items())}
    for key, field in (("by_region", "region"), ("by_resolution", "res_band"), ("by_seed_type", "seed_type")):
        groups = defaultdict(list)
        for i, r in enumerate(recs):
            groups[str(r.get(field))].append(i)
        results[key] = {}
        for g, idx in sorted(groups.items()):
            rk, tr = subset(idx, main_rankings, truths)
            results[key][g] = sample_metrics(rk, tr, ks)
    with_meta = [i for i, q in enumerate(queries) if q.text]
    without_meta = [i for i, q in enumerate(queries) if not q.text]
    results["by_metadata_present"] = {
        "has title/description": sample_metrics(*subset(with_meta, main_rankings, truths), ks),
        "no title/description": sample_metrics(*subset(without_meta, main_rankings, truths), ks),
    }
    write_json(P.report("metrics.json"), results)
    log.info("wrote %s", P.report("metrics.json"))

    # sample of "wrong" suggestions for human review (false-negative inflation)
    rng = random.Random(int(ec["seed"]))
    k_rev = int(ec["review_k"])
    wrong = []
    for i, (rk, sc, truth) in enumerate(zip(main_rankings, main_scores, truths)):
        for pos, (c, s) in enumerate(zip(rk[:k_rev], sc[:k_rev])):
            if c not in truth:
                wrong.append((i, pos + 1, c, s))
    rng.shuffle(wrong)
    with open(P.report("review_sample.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["file", "page_url", "thumb_url", "suggested_category", "rank", "score",
                    "editor_categories", "reviewer_says_correct"])
        for i, pos, c, s in wrong[:int(ec["review_sample_size"])]:
            r = recs[i]
            w.writerow([r["title"], r.get("page_url", ""), r["thumb_url"], store.names[c], pos, "%.4f" % s,
                        " | ".join(r["labels"]), ""])
    log.info("wrote %s (fill the last column with y/n)", P.report("review_sample.csv"))


if __name__ == "__main__":
    main()
