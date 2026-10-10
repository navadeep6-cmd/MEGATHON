#!/usr/bin/env python3
"""Step 2 - download thumbnails (standard 330px step) as RGB JPEG.

Polite: descriptive User-Agent, few parallel workers, back-off on 429/5xx.
Existing files are skipped, so the step can be resumed.
Output: data/images/<pageid>.jpg
"""

import argparse
import logging
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import requests  # noqa: E402

from catsuggest.commons_api import user_agent  # noqa: E402
from catsuggest.config import Paths, load_config  # noqa: E402
from catsuggest.images import to_rgb_jpeg_bytes  # noqa: E402
from catsuggest.io_utils import read_jsonl, setup_logging  # noqa: E402

log = logging.getLogger("s02")
_local = threading.local()


def _session(ua: str) -> requests.Session:
    s = getattr(_local, "session", None)
    if s is None:
        s = requests.Session()
        s.headers["User-Agent"] = ua
        _local.session = s
    return s


def fetch(url: str, dest: str, ua: str, retries: int = 5) -> bool:
    delay = 2.0
    for _ in range(retries):
        try:
            r = _session(ua).get(url, timeout=60)
        except requests.RequestException:
            time.sleep(delay)
            delay *= 2
            continue
        if r.status_code in (429, 500, 502, 503, 504):
            time.sleep(float(r.headers.get("Retry-After", delay)))
            delay *= 2
            continue
        if r.status_code != 200:
            return False
        try:
            data = to_rgb_jpeg_bytes(r.content)
        except Exception:
            return False
        tmp = dest + ".part"
        with open(tmp, "wb") as f:
            f.write(data)
        os.replace(tmp, dest)
        return True
    return False


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    setup_logging()
    cfg = load_config(args.config)
    P = Paths(cfg).ensure()
    ua = user_agent(cfg["commons"].get("contact", ""))
    records = read_jsonl(P.raw_files)
    todo = [(r["thumb_url"], P.image_path(r["pageid"])) for r in records
            if r.get("thumb_url") and not os.path.exists(P.image_path(r["pageid"]))]
    log.info("%d records, %d thumbnails to download", len(records), len(todo))
    ok = failed = 0
    from tqdm import tqdm
    with ThreadPoolExecutor(max_workers=int(cfg["commons"]["download_workers"])) as pool:
        futures = [pool.submit(fetch, url, dest, ua) for url, dest in todo]
        for fut in tqdm(as_completed(futures), total=len(futures), desc="download"):
            if fut.result():
                ok += 1
            else:
                failed += 1
    log.info("downloaded %d, failed %d", ok, failed)


if __name__ == "__main__":
    main()
