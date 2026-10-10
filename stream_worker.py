#!/usr/bin/env python3
"""Live demo of the asynchronous serving path (read-only).

upload event -> queue -> encoder -> ANN lookup -> fusion -> stored suggestions

Listens to the public Wikimedia EventStreams feed for new Commons uploads,
fetches each file's thumbnail and metadata, and appends suggestions to
data/live/suggestions.jsonl. It never edits Commons; suggestions are for
human review. Uploads are never blocked: work happens after the fact.
"""

import argparse
import json
import logging
import os
import queue
import sys
import threading
import time
from io import BytesIO

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import requests  # noqa: E402
from PIL import Image  # noqa: E402

from catsuggest.commons_api import CommonsClient, page_to_record, user_agent  # noqa: E402
from catsuggest.config import Paths, load_config  # noqa: E402
from catsuggest.io_utils import append_jsonl, setup_logging  # noqa: E402
from catsuggest.suggester import Suggester  # noqa: E402

log = logging.getLogger("stream")
STREAM_URL = "https://stream.wikimedia.org/v2/stream/recentchange"


def listen(q: "queue.Queue", ua: str, stop: threading.Event) -> None:
    """Server-sent events reader: puts File: titles of new Commons uploads on the queue."""
    while not stop.is_set():
        try:
            with requests.get(STREAM_URL, stream=True, timeout=(10, 120),
                              headers={"User-Agent": ua, "Accept": "text/event-stream"}) as r:
                r.raise_for_status()
                for line in r.iter_lines(decode_unicode=True):
                    if stop.is_set():
                        return
                    if not line or not line.startswith("data:"):
                        continue
                    try:
                        ev = json.loads(line[5:].strip())
                    except ValueError:
                        continue
                    if ev.get("wiki") == "commonswiki" and ev.get("type") == "log" \
                            and ev.get("log_type") == "upload" and ev.get("log_action") == "upload":
                        q.put(ev.get("title"))
        except requests.RequestException as exc:
            log.warning("stream interrupted (%s); reconnecting", exc)
            time.sleep(5)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--max-events", type=int, default=50)
    args = ap.parse_args()
    setup_logging()
    cfg = load_config(args.config)
    P = Paths(cfg).ensure()
    cc = cfg["commons"]
    ua = user_agent(cc.get("contact", ""))
    client = CommonsClient(cc["api_url"], cc.get("contact", ""), float(cc["request_sleep"]))
    sug = Suggester(cfg, memory="serve")
    session = requests.Session()
    session.headers["User-Agent"] = ua

    q: "queue.Queue" = queue.Queue(maxsize=1000)
    stop = threading.Event()
    threading.Thread(target=listen, args=(q, ua, stop), daemon=True).start()
    done = 0
    while done < args.max_events:
        title = q.get()
        time.sleep(5)  # give the file page a moment to settle
        try:
            info = client.file_info_by_title([title], int(cc["thumb_width"]))
            page = next(iter(info.values()), None)
            rec = page_to_record(page) if page else None
            if rec is None or not rec["thumb_url"]:
                continue
            img = Image.open(BytesIO(session.get(rec["thumb_url"], timeout=60).content)).convert("RGB")
            res = sug.suggest(img, title=rec["title"], description=rec["description"],
                              date=rec["date_original"], lat=rec["lat"], lon=rec["lon"])
        except Exception as exc:  # keep the worker alive on any single bad upload
            log.warning("skipping %s: %s", title, exc)
            continue
        append_jsonl(P.live_suggestions, {"file": rec["title"], "page_url": rec["page_url"],
                                          "editor_categories_so_far": rec["categories"],
                                          "suggestions": [(s["category"], s["score"]) for s in res["suggestions"]],
                                          "latency_ms": res["timing_ms"]["total"]})
        done += 1
        log.info("%s -> %s", rec["title"], ", ".join(s["category"] for s in res["suggestions"][:5]))
    stop.set()


if __name__ == "__main__":
    main()
