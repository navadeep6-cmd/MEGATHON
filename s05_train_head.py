#!/usr/bin/env python3
"""Step 5 - train the supervised multi-label head (asymmetric loss, text-initialised).

Covers categories with >= labels.head_min_train_count training images.
Training data = clean + degraded training embeddings (+ accepted feedback);
editor rejections from the feedback loop are used as reliable negatives.
Model selection by validation mAP.

Outputs: artifacts/head.npz, reports/head_training.json
"""

import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from catsuggest.config import Paths, load_config  # noqa: E402
from catsuggest.head import LinearHead, train_head  # noqa: E402
from catsuggest.io_utils import iter_jsonl, read_json, setup_logging, write_json  # noqa: E402
from catsuggest.pipeline import load_split  # noqa: E402

log = logging.getLogger("s05")


def feedback_rows(P, col_of):
    """(embeddings, positive cols, negative cols) from applied editor feedback, if any."""
    if not os.path.exists(P.feedback_labels):
        return None
    by_query = {}
    for row in iter_jsonl(P.feedback_labels):
        entry = by_query.setdefault(row["query_id"], {"pos": set(), "neg": set()})
        col = col_of.get(row["category"])
        if col is not None:
            entry["neg" if not row["accepted"] else "pos"].add(col)
    Z, pos, neg = [], [], []
    for qid, entry in by_query.items():
        path = os.path.join(P.query_cache, qid + ".npy")
        if os.path.exists(path) and (entry["pos"] or entry["neg"]):
            Z.append(np.load(path))
            pos.append(sorted(entry["pos"]))
            neg.append(sorted(entry["neg"] - entry["pos"]))
    if not Z:
        return None
    return np.stack(Z).astype(np.float32), pos, neg


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    setup_logging()
    cfg = load_config(args.config)
    P = Paths(cfg).ensure()

    vocab = read_json(P.vocab)
    min_count = int(cfg["labels"]["head_min_train_count"])
    head_names = [c["name"] for c in vocab if c["train_count"] >= min_count]
    if not head_names:
        log.warning("no category has >= %d training images; skipping the supervised head", min_count)
        if os.path.exists(P.head):
            os.remove(P.head)
        return
    name_to_id = {c["name"]: i for i, c in enumerate(vocab)}
    col_of = {n: j for j, n in enumerate(head_names)}
    cat_ids = np.asarray([name_to_id[n] for n in head_names], np.int64)
    protos = np.load(P.protos)
    T_head = protos[cat_ids]

    tr, Ztr, _ = load_split(P, "train")
    va, Zva, _ = load_split(P, "val")
    Ytr = [[col_of[c] for c in r["labels"] if c in col_of] for r in tr]
    Yva = [[col_of[c] for c in r["labels"] if c in col_of] for r in va]

    # degraded training views share the labels of the clean image
    ids_all = read_json(P.emb_ids("train"))
    pos_of = {rid: i for i, rid in enumerate(ids_all)}
    rows = [pos_of[r["id"]] for r in tr]
    X_parts, Y = [Ztr], list(Ytr)
    for v in range(int(cfg["encoder"]["train_aug_views"])):
        path = P.emb("img_aug%d" % v, "train")
        if os.path.exists(path):
            X_parts.append(np.load(path)[rows])
            Y.extend(Ytr)
    neg = [[] for _ in Y]
    fb = feedback_rows(P, col_of)
    if fb is not None:
        Zf, pos_f, neg_f = fb
        X_parts.append(Zf)
        Y.extend(pos_f)
        neg.extend(neg_f)
        log.info("feedback: %d images (%d hard negatives)", len(pos_f), sum(len(n) for n in neg_f))
    X = np.concatenate(X_parts, axis=0)
    log.info("head: %d categories, %d training rows, %d val rows", len(head_names), X.shape[0], Zva.shape[0])

    W, b, history = train_head(X, Y, Zva, Yva, T_head, cfg["head"],
                               neg_train=neg if fb is not None else None, seed=int(cfg["eval"]["seed"]))
    LinearHead(W, b, cat_ids).save(P.head)
    write_json(P.report("head_training.json"), {"categories": len(head_names), "rows": int(X.shape[0]),
                                                "history": history})
    log.info("saved %s", P.head)


if __name__ == "__main__":
    main()
