#!/usr/bin/env python3
"""Step 1 - crawl Wikimedia Commons (read-only API).

* Breadth-first walk below seed categories (objects, scenes, events, abstract
  concepts and regions) collecting files.
* Fetches each file's non-hidden categories, thumbnail URL, original size,
  uploader, upload time, licence, description, EXIF date and geotag.
* Fetches parent categories (for the hierarchy) and category redirects.

Outputs: data/raw/files.jsonl, data/raw/category_parents.json,
         data/raw/category_redirects.json
"""

import argparse
import logging
import os
import sys
from collections import deque

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from catsuggest.commons_api import CommonsClient, page_to_record  # noqa: E402
from catsuggest.config import Paths, load_config  # noqa: E402
from catsuggest.io_utils import read_json, setup_logging, write_json, write_jsonl  # noqa: E402
from catsuggest.labels import clean_labels, is_maintenance  # noqa: E402
from catsuggest.textutil import clean_category_name  # noqa: E402

log = logging.getLogger("s01")


def round_robin_seeds(seeds):
    """Interleave seed types so a truncated run (--max-seeds) still covers all types."""
    queues = [(t, list(names)) for t, names in seeds.items()]
    out = []
    while any(names for _, names in queues):
        for t, names in queues:
            if names:
                out.append((t, names.pop(0)))
    return out


def crawl(client, cc, max_seeds, max_files_per_seed):
    members = {}
    seeds = round_robin_seeds(cc["seeds"])
    if max_seeds:
        seeds = seeds[:max_seeds]
    for n_seed, (seed_type, seed) in enumerate(seeds, 1):
        collected = 0
        queue = deque([(clean_category_name(seed), 0)])
        seen = {clean_category_name(seed)}
        while queue and collected < max_files_per_seed:
            cat, depth = queue.popleft()
            for m in client.category_members(cat, "file", int(cc["files_per_category"])):
                pid = m.get("pageid")
                if pid is None or str(pid) in members:
                    continue
                members[str(pid)] = {"title": m.get("title", ""), "seed": seed, "seed_type": seed_type}
                collected += 1
                if collected >= max_files_per_seed:
                    break
            if depth < int(cc["max_depth"]) and collected < max_files_per_seed:
                for sub in client.category_members(cat, "subcat", int(cc["subcats_per_category"])):
                    name = clean_category_name(sub.get("title", ""))
                    if name and name not in seen and not is_maintenance(name):
                        seen.add(name)
                        queue.append((name, depth + 1))
        level = logging.WARNING if collected == 0 else logging.INFO
        log.log(level, "[%d/%d] seed %-30s (%s): %d files", n_seed, len(seeds), seed, seed_type, collected)
    return members


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--max-seeds", type=int, default=0, help="limit number of seeds (quick runs)")
    ap.add_argument("--max-files-per-seed", type=int, default=0)
    ap.add_argument("--refresh", action="store_true", help="re-crawl even if a crawl cache exists")
    args = ap.parse_args()
    setup_logging()
    cfg = load_config(args.config)
    P = Paths(cfg).ensure()
    cc = cfg["commons"]
    client = CommonsClient(cc["api_url"], cc.get("contact", ""), float(cc["request_sleep"]))

    # 1) crawl seed categories (cache re-used only if it was made with the same settings,
    #    so a full run after a --quick run really crawls the full dataset)
    cap = args.max_files_per_seed or int(cc["max_files_per_seed"])
    params = {"max_seeds": args.max_seeds, "max_files_per_seed": cap, "seeds": cc["seeds"],
              "max_depth": cc["max_depth"], "files_per_category": cc["files_per_category"],
              "subcats_per_category": cc["subcats_per_category"]}
    cache = read_json(P.crawl_members) if os.path.exists(P.crawl_members) else None
    if cache is not None and not args.refresh and cache.get("params") == params:
        members = cache["members"]
        log.info("re-using crawl cache with %d files (use --refresh to re-crawl)", len(members))
    else:
        members = crawl(client, cc, args.max_seeds, cap)
        write_json(P.crawl_members, {"params": params, "members": members})
    log.info("files to describe: %d", len(members))

    # 2) file metadata in batches of 50
    pageids = sorted(int(p) for p in members)
    records = []
    batch = 500
    for start in range(0, len(pageids), batch):
        info = client.file_info(pageids[start:start + batch], int(cc["thumb_width"]))
        for pid, page in info.items():
            rec = page_to_record(page)
            if rec is None:
                continue
            m = members.get(str(pid), {})
            rec["seed"], rec["seed_type"] = m.get("seed"), m.get("seed_type")
            records.append(rec)
        log.info("described %d / %d files", min(start + batch, len(pageids)), len(pageids))
    write_jsonl(P.raw_files, records)
    log.info("wrote %d image records to %s", len(records), P.raw_files)

    # 3) category graph and redirects for the (cleaned) label set
    labels = set()
    for rec in records:
        labels.update(clean_labels(rec["categories"]))
    log.info("distinct content categories: %d", len(labels))
    parents_all, redirects = {}, {}
    frontier = set(labels)
    for level in range(int(cc["parent_fetch_depth"])):
        todo = sorted(c for c in frontier if c not in parents_all)
        if not todo:
            break
        log.info("fetching parents, level %d: %d categories", level + 1, len(todo))
        parents, redir = client.category_parents(todo)
        parents_all.update(parents)
        redirects.update(redir)
        frontier = set()
        for ps in parents.values():
            frontier.update(p for p in ps if not is_maintenance(p))
    soft = client.soft_redirects(sorted(labels))
    write_json(P.raw_parents, parents_all)
    write_json(P.raw_redirects, {"hard": redirects, "soft": soft})
    log.info("parents for %d categories, %d hard redirects, %d soft redirects",
             len(parents_all), len(redirects), len(soft))


if __name__ == "__main__":
    main()
