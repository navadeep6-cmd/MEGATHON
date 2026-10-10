#!/usr/bin/env python3
"""Step 3 - clean labels, split by time and uploader, remove near-duplicates.

* keeps images under open licences that downloaded correctly;
* drops maintenance/tracking/provenance and soft-redirected categories,
  resolves hard redirects, removes categories of individual people;
* temporal(-grouped) split; perceptual-hash (dHash) near-duplicates of
  training images are removed from val/test (and val duplicates from test);
* writes the vocabulary (all labels + parent-category distractors) with
  frequency band, semantic type and script.

Outputs: data/processed/dataset.jsonl, data/processed/categories.json,
         reports/data_stats.json
"""

import argparse
import logging
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from catsuggest import labels as L  # noqa: E402
from catsuggest.config import Paths, load_config  # noqa: E402
from catsuggest.geo import region_of  # noqa: E402
from catsuggest.images import dhash, load_rgb  # noqa: E402
from catsuggest.io_utils import read_json, read_jsonl, setup_logging, write_json, write_jsonl  # noqa: E402
from catsuggest.split import SPLITS, near_duplicate_mask_np, temporal_split  # noqa: E402
from catsuggest.textutil import clean_file_title, meta_year, name_script, strip_html  # noqa: E402

log = logging.getLogger("s03")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    setup_logging()
    cfg = load_config(args.config)
    P = Paths(cfg).ensure()
    lc, sc = cfg["labels"], cfg["split"]

    raw = read_jsonl(P.raw_files)
    parents = read_json(P.raw_parents) if os.path.exists(P.raw_parents) else {}
    redirect_info = read_json(P.raw_redirects) if os.path.exists(P.raw_redirects) else {}
    hard = redirect_info.get("hard", {})
    soft = set(redirect_info.get("soft", []))
    person = set()
    if lc.get("exclude_person_categories", True):
        # categories of individual people, and their direct subcategories ("<Name> in 2019")
        person = {c for c, ps in parents.items() if L.is_person_category(ps)}
        person |= {c for c, ps in parents.items() if any(p in person for p in ps)}
    log.info("raw records %d; person categories excluded %d; soft redirects %d", len(raw), len(person), len(soft))

    stats = Counter()
    records = []
    from tqdm import tqdm
    for r in tqdm(raw, desc="clean + hash"):
        path = P.image_path(r["pageid"])
        if not os.path.exists(path):
            stats["no_image"] += 1
            continue
        if not L.license_ok(r.get("license")):
            stats["licence_not_open"] += 1
            continue
        cats = L.clean_labels(r.get("categories", []), redirects=hard, drop=soft | person,
                              extra_patterns=lc.get("extra_drop_patterns", []))
        if not cats:
            stats["no_content_labels"] += 1
            continue
        try:
            h = dhash(load_rgb(path))
        except Exception:
            stats["bad_image"] += 1
            continue
        title_clean = clean_file_title(r.get("title", ""))
        records.append({
            "id": str(r["pageid"]),
            "title": r.get("title", ""),
            "title_clean": title_clean,
            "description": strip_html(r.get("description", ""))[:1000],
            "year": meta_year(r.get("date_original")),
            "lat": r.get("lat"),
            "lon": r.get("lon"),
            "region": region_of(r.get("lat"), r.get("lon")),
            "width": r.get("width"),
            "height": r.get("height"),
            "res_band": L.resolution_band(r.get("width"), r.get("height")),
            "timestamp": r.get("timestamp", ""),
            "uploader": r.get("uploader"),
            "license": r.get("license"),
            "thumb_url": r.get("thumb_url", ""),
            "page_url": r.get("page_url", ""),
            "path": os.path.relpath(path, P.data),
            "seed_type": r.get("seed_type"),
            "labels": cats,
            "dhash": str(h),
        })
    log.info("usable records: %d (%s)", len(records), dict(stats))
    if not records:
        raise SystemExit("no usable records - check steps 1 and 2")

    # split (uploader-grouped temporal; falls back to strict temporal if a split would be empty)
    split_mode = sc["mode"]
    assign = temporal_split(records, float(sc["val_frac"]), float(sc["test_frac"]), split_mode)
    counts = Counter(assign.values())
    if split_mode == "temporal_grouped" and any(counts.get(s, 0) == 0 for s in SPLITS):
        log.warning("uploader-grouped split left an empty split %s (one uploader dominates); "
                    "falling back to the strict temporal split", dict(counts))
        split_mode = "temporal"
        assign = temporal_split(records, float(sc["val_frac"]), float(sc["test_frac"]), split_mode)
    for rec in records:
        rec["split"] = assign[rec["id"]]

    # perceptual-hash near-duplicates: val/test vs train, test vs val
    def hashes(split):
        return [int(r["dhash"]) for r in records if r["split"] == split]

    removed = Counter()
    for split, refs in (("val", ["train"]), ("test", ["train", "val"])):
        ref = [h for s in refs for h in hashes(s)]
        items = [r for r in records if r["split"] == split]
        mask = near_duplicate_mask_np([int(r["dhash"]) for r in items], ref, int(sc["dhash_max_distance"]))
        for rec, dup in zip(items, mask):
            if dup:
                rec["split"] = "removed_duplicate"
                removed[split] += 1
    log.info("near-duplicates removed: %s", dict(removed))
    records = [r for r in records if r["split"] in SPLITS]

    # vocabulary and frequencies
    total = Counter(c for r in records for c in r["labels"])
    label_vocab = {c for c, n in total.items() if n >= int(lc["min_label_count"])}
    for r in records:
        r["labels"] = [c for c in r["labels"] if c in label_vocab]
    records = [r for r in records if r["labels"]]
    for split in ("val", "test"):
        if not any(r["split"] == split for r in records):
            raise SystemExit("the %s split is empty - crawl more files or use split.mode: temporal" % split)
    # Distractors: real Commons categories that no image in the sample carries
    # (all crawled parent/grand-parent categories). The model must rank the right
    # category among these too, so the evaluation is not a closed world of the
    # dataset's own labels.
    distractors = set()
    if lc.get("add_parent_distractors", True):
        extra = lc.get("extra_drop_patterns", [])
        for child, ps in parents.items():
            for c in [child] + list(ps):
                if c not in label_vocab and c not in soft and c not in person and not L.is_maintenance(c, extra):
                    distractors.add(c)
    vocab = sorted(label_vocab | distractors)
    train_freq = Counter(c for r in records if r["split"] == "train" for c in r["labels"])
    categories = [{
        "name": c,
        "train_count": int(train_freq.get(c, 0)),
        "total_count": int(total.get(c, 0)),
        "distractor": c in distractors,
        "type": L.category_type(c),
        "band": L.frequency_band(train_freq.get(c, 0), int(lc["tail_max"]), int(lc["torso_max"])),
        "script": name_script(c),
    } for c in vocab]

    write_jsonl(P.dataset, records)
    write_json(P.vocab, categories)
    split_counts = Counter(r["split"] for r in records)
    label_card = [len(r["labels"]) for r in records]
    data_stats = {
        "records": len(records),
        "splits": dict(split_counts),
        "categories": len(categories),
        "label_categories": len(label_vocab),
        "distractor_categories": len(distractors),
        "labels_per_image_mean": sum(label_card) / float(len(label_card)),
        "bands": dict(Counter(c["band"] for c in categories)),
        "types": dict(Counter(c["type"] for c in categories)),
        "scripts": dict(Counter(c["script"] for c in categories)),
        "regions": dict(Counter(r["region"] for r in records)),
        "resolution_bands": dict(Counter(r["res_band"] for r in records)),
        "filtered": dict(stats),
        "near_duplicates_removed": dict(removed),
        "split_mode": split_mode,
        "time_range": {s: [min((r["timestamp"] for r in records if r["split"] == s), default=""),
                           max((r["timestamp"] for r in records if r["split"] == s), default="")]
                       for s in SPLITS},
    }
    write_json(P.report("data_stats.json"), data_stats)
    log.info("dataset: %s; vocabulary %d categories (%d labels + %d distractors)", dict(split_counts),
             len(categories), len(label_vocab), len(distractors))


if __name__ == "__main__":
    main()
