#!/usr/bin/env python3
"""Step 9 - robustness to image quality.

Re-encodes a fixed random subset of test images under synthetic corruptions
(Gaussian blur, JPEG compression, low resolution, cropping) and evaluates the
main system with the same metadata. Clean results on the same subset are the
reference.

Output: reports/robustness.json
"""

import argparse
import logging
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from catsuggest.config import Paths, load_config  # noqa: E402
from catsuggest.encoder import Encoder  # noqa: E402
from catsuggest.fusion import FusionModel  # noqa: E402
from catsuggest.hierarchy import rerank_with_hierarchy  # noqa: E402
from catsuggest.images import CORRUPTIONS, Corruption  # noqa: E402
from catsuggest.io_utils import setup_logging, write_json  # noqa: E402
from catsuggest.metrics import sample_metrics  # noqa: E402
from catsuggest.pipeline import build_scorer, load_split, score_in_batches, truths_for  # noqa: E402
from catsuggest.suggester import Query, load_graph, rank  # noqa: E402

log = logging.getLogger("s09")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    setup_logging()
    cfg = load_config(args.config)
    P = Paths(cfg).ensure()
    ks = [int(k) for k in cfg["eval"]["ks"]]
    top_k = int(cfg["eval"]["top_k"])

    scorer, store, _, _ = build_scorer(cfg, memory="eval")
    graph = load_graph(cfg)
    fusion = FusionModel.load(P.fusion(cfg["fusion"]["model"]))
    recs, _, Zm = load_split(P, "test")
    rng = random.Random(int(cfg["eval"]["seed"]))
    idx = sorted(rng.sample(range(len(recs)), min(len(recs), int(cfg["eval"]["robustness_subset"]))))
    recs = [recs[i] for i in idx]
    Zm = Zm[idx]
    truths = truths_for(recs, store)
    queries = [Query.from_record(r) for r in recs]
    paths = [P.image_path(r["id"]) for r in recs]

    ec = cfg["encoder"]
    enc = Encoder(ec["model"], ec["pretrained"], ec["device"], ec["batch_size"], ec["num_workers"])
    out = {"n_images": len(recs), "corruptions": {}}
    for name in CORRUPTIONS:
        Z, ok = enc.encode_paths(paths, transform=Corruption(name), seed=7, progress=False)
        cands = score_in_batches(scorer, Z, Zm, queries, progress=False)
        rankings = []
        for cs in cands:
            named = [(store.names[c], s) for c, s in rank(cs, fusion.predict(cs.X))]
            if cfg["hierarchy"]["enabled"]:
                named, _ = rerank_with_hierarchy(named, graph, top_k, float(cfg["hierarchy"]["min_relative_score"]))
            rankings.append([store.id_of[n] for n, _ in named])
        m = sample_metrics(rankings, truths, ks)
        m["decode_failures"] = int((~ok).sum())
        out["corruptions"][name] = m
        log.info("%-12s P@1 %.3f  P@5 %.3f  R@10 %.3f  mAP %.3f", name, m["P@1"], m.get("P@5", 0.0),
                 m.get("R@10", 0.0), m["mAP"])
    write_json(P.report("robustness.json"), out)


if __name__ == "__main__":
    main()
