#!/usr/bin/env python3
"""Step 4 - encode images, metadata text and category names with the open encoder.

* image embeddings for every split (+ degraded training views for robustness);
* text-tower embeddings of title + description (metadata scorer);
* category text prototypes t_c (prompt ensemble);
* embedding near-duplicates (cosine >= threshold to any training image) are
  flagged for removal from val/test.

Outputs: artifacts/embeddings/*.npy, artifacts/encoder.json
"""

import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from catsuggest.config import Paths, load_config  # noqa: E402
from catsuggest.encoder import Encoder  # noqa: E402
from catsuggest.images import Corruption  # noqa: E402
from catsuggest.io_utils import read_json, read_jsonl, setup_logging, write_json  # noqa: E402
from catsuggest.split import SPLITS  # noqa: E402
from catsuggest.textutil import build_meta_text  # noqa: E402

log = logging.getLogger("s04")


def max_cosine_to(reference: np.ndarray, queries: np.ndarray, chunk: int = 1024) -> np.ndarray:
    out = np.full(queries.shape[0], -1.0, np.float32)
    if reference.shape[0] == 0:
        return out
    for s in range(0, queries.shape[0], chunk):
        out[s:s + chunk] = (queries[s:s + chunk] @ reference.T).max(axis=1)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    setup_logging()
    cfg = load_config(args.config)
    P = Paths(cfg).ensure()
    ec = cfg["encoder"]
    enc = Encoder(ec["model"], ec["pretrained"], ec["device"], ec["batch_size"], ec["num_workers"])
    records = read_jsonl(P.dataset)

    for split in SPLITS:
        recs = [r for r in records if r["split"] == split]
        ids = [r["id"] for r in recs]
        paths = [P.image_path(i) for i in ids]
        log.info("%s: encoding %d images", split, len(paths))
        Z, ok = enc.encode_paths(paths, seed=0)
        np.save(P.emb("img", split), Z)
        np.save(P.emb("ok", split), ok)
        texts = [build_meta_text(r["title_clean"], r["description"]) for r in recs]
        np.save(P.emb("meta", split), enc.encode_texts(texts))
        write_json(P.emb_ids(split), ids)
        if (~ok).any():
            log.warning("%s: %d images failed to decode and will be ignored", split, int((~ok).sum()))
        if split == "train":
            for v in range(int(ec["train_aug_views"])):
                log.info("train: degraded view %d", v)
                Za, _ = enc.encode_paths(paths, transform=Corruption(augment=True), seed=1000 + v)
                np.save(P.emb("img_aug%d" % v, split), Za)

    # category prototypes, aligned with categories.json order
    vocab = read_json(P.vocab)
    names = [c["name"] for c in vocab]
    log.info("encoding %d category names", len(names))
    np.save(P.protos, enc.category_prototypes(names))

    # embedding near-duplicates of training images in val/test
    thr = float(cfg["split"]["embed_dup_threshold"])
    Ztr = np.load(P.emb("img", "train"))[np.load(P.emb("ok", "train"))]
    dropped = {}
    for split in ("val", "test"):
        Zs = np.load(P.emb("img", split))
        ids = read_json(P.emb_ids(split))
        sims = max_cosine_to(Ztr, Zs)
        dropped[split] = [ids[i] for i in np.nonzero(sims >= thr)[0].tolist()]
        log.info("%s: %d embedding near-duplicates (cos >= %.2f)", split, len(dropped[split]), thr)
    write_json(P.dedup_dropped, dropped)
    write_json(P.encoder_info, {"model": ec["model"], "pretrained": ec["pretrained"], "dim": enc.dim,
                                "device": enc.device})


if __name__ == "__main__":
    main()
