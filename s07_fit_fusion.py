#!/usr/bin/env python3
"""Step 7 - fit the calibrated stacking model on the validation split.

For every validation image, candidates = union of each scorer's top-N.
Rows: (image, candidate) feature vectors; label 1 if the editor assigned it.
Candidates that are *ancestors* of a true label are left out of training:
editors file under the most specific category, so an unlabelled ancestor is
unknown, not negative.

Outputs: artifacts/fusion/fusion_{logreg,gbdt}.joblib, reports/fusion.json
"""

import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from catsuggest.config import Paths, load_config  # noqa: E402
from catsuggest.fusion import FEATURES, FusionModel  # noqa: E402
from catsuggest.io_utils import setup_logging, write_json  # noqa: E402
from catsuggest.metrics import candidate_recall  # noqa: E402
from catsuggest.pipeline import build_scorer, load_split, score_in_batches, truths_for  # noqa: E402
from catsuggest.suggester import Query, load_graph  # noqa: E402

log = logging.getLogger("s07")


def training_rows(cands, truths, store, graph):
    X, y = [], []
    for cs, truth in zip(cands, truths):
        if not truth:
            continue
        unknown = graph.ancestors_of_set(store.names[t] for t in truth)
        keep = []
        labels = []
        for j, c in enumerate(cs.cat_ids.tolist()):
            if c in truth:
                keep.append(j)
                labels.append(1)
            elif store.names[c] not in unknown:
                keep.append(j)
                labels.append(0)
        if keep:
            X.append(cs.X[keep])
            y.extend(labels)
    return np.concatenate(X, axis=0), np.asarray(y, np.int64)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    setup_logging()
    cfg = load_config(args.config)
    P = Paths(cfg).ensure()

    scorer, store, _, _ = build_scorer(cfg, memory="eval")
    graph = load_graph(cfg)
    recs, Z, Zm = load_split(P, "val")
    truths = truths_for(recs, store)
    cands = score_in_batches(scorer, Z, Zm, [Query.from_record(r) for r in recs])
    cand_recall = candidate_recall([cs.cat_ids.tolist() for cs in cands], truths)
    X, y = training_rows(cands, truths, store, graph)
    log.info("fusion training rows %d (positives %d), candidate recall %.3f", len(y), int(y.sum()), cand_recall)

    report = {"rows": int(len(y)), "positives": int(y.sum()), "val_images": len(recs),
              "mean_candidates_per_image": float(np.mean([len(cs.cat_ids) for cs in cands])),
              "candidate_recall_val": cand_recall, "features": list(FEATURES), "models": {}}
    for kind in ("logreg", "gbdt"):
        model = FusionModel.fit(X, y, kind=kind, C=float(cfg["fusion"]["C"]), seed=int(cfg["eval"]["seed"]))
        model.save(P.fusion(kind))
        report["models"][kind] = {"path": os.path.relpath(P.fusion(kind), P.artifacts),
                                  "standardised_coefficients": model.coefficients()}
        log.info("saved %s", P.fusion(kind))
    write_json(P.report("fusion.json"), report)


if __name__ == "__main__":
    main()
