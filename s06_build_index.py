#!/usr/bin/env python3
"""Step 6 - build the category store and the k-NN label memories.

* artifacts/categories : names, text prototypes, training frequencies, types
* artifacts/memory_eval : ANN index of TRAIN images only (used for fitting and
  evaluation, so val/test never see their own labels)
* artifacts/memory_serve: ANN index of all labelled images (used by the demo/API)
"""

import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from catsuggest.config import Paths, load_config  # noqa: E402
from catsuggest.feedback import serving_writer_pid  # noqa: E402
from catsuggest.io_utils import read_json, setup_logging  # noqa: E402
from catsuggest.pipeline import load_split  # noqa: E402
from catsuggest.store import CategoryStore, LabelMemory  # noqa: E402
from catsuggest.suggester import ann_kwargs  # noqa: E402

log = logging.getLogger("s06")


def build_memory(cfg, P, store, splits, out_dir):
    mem = None
    for split in splits:
        recs, Z, _ = load_split(P, split, drop_dedup=False)
        if mem is None:
            mem = LabelMemory(Z.shape[1], cfg["knn"]["index"], **ann_kwargs(cfg))
        mem.add(Z, [[store.id_of[c] for c in r["labels"]] for r in recs],
                [r["title"] for r in recs], [r["thumb_url"] for r in recs],
                [P.image_path(r["id"]) for r in recs], [r["lat"] for r in recs], [r["lon"] for r in recs])
    mem.save(out_dir)
    log.info("%s: %d images, index %s (%s), %.1f MB", os.path.basename(out_dir), len(mem),
             mem.ann.kind, mem.ann.backend, mem.ann.nbytes() / 1e6)
    return mem


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    setup_logging()
    cfg = load_config(args.config)
    P = Paths(cfg).ensure()
    owner = serving_writer_pid(P.serve_lock)
    if owner is not None:
        raise SystemExit("The demo/API (process %d) is using the serving index - stop it before rebuilding." % owner)

    vocab = read_json(P.vocab)
    store = CategoryStore([c["name"] for c in vocab], np.load(P.protos),
                          [c["train_count"] for c in vocab], [c["type"] for c in vocab])
    store.save(P.store_dir)
    log.info("category store: %d categories", len(store))
    build_memory(cfg, P, store, ["train"], P.memory_eval)
    build_memory(cfg, P, store, ["train", "val", "test"], P.memory_serve)
    # The serving index was rebuilt from scratch, so the record of which feedback
    # was already applied (and to which rows) is void. Both files are derived from
    # feedback.jsonl; s13 (run next by run_all.sh) re-applies every logged event.
    for derived in (P.feedback_state, P.feedback_labels):
        if os.path.exists(derived):
            os.remove(derived)


if __name__ == "__main__":
    main()
