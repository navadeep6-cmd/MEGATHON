#!/usr/bin/env python3
"""Continual learning - apply editor feedback without retraining the encoder.

Reads new accept/reject events from data/feedback/feedback.jsonl:
  * accepted categories: the image embedding (cached at suggestion time) is
    appended to the serving k-NN memory; unknown category names are added to
    the category store (text prototype computed on the fly);
  * every event is appended to data/feedback/feedback_labels.jsonl, which the
    next scheduled head retrain (s05) uses - rejects as reliable hard negatives.

While the API or demo is running it owns the serving index and applies
feedback itself (demo: on every submit; API: POST /feedback/apply), so this
script refuses to run then. Use it offline, e.g. from cron between restarts.
Retrain policy: weekly, or when s14_monitor.py flags drift: s05 -> s07 -> s08.
"""

import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from catsuggest.config import Paths, load_config  # noqa: E402
from catsuggest.feedback import apply_feedback, claim_serving_writer, serving_writer_pid  # noqa: E402
from catsuggest.io_utils import setup_logging  # noqa: E402
from catsuggest.suggester import Suggester  # noqa: E402

log = logging.getLogger("s13")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    setup_logging()
    cfg = load_config(args.config)
    P = Paths(cfg).ensure()
    owner = serving_writer_pid(P.serve_lock)
    if owner is not None:
        raise SystemExit("The API/demo (process %d) owns the serving index and applies feedback itself. "
                         "Use its 'apply' button / POST /feedback/apply, or stop it first." % owner)
    if not os.path.exists(P.feedback_log):
        log.info("no feedback yet (%s)", P.feedback_log)
        return
    claim_serving_writer(P.serve_lock)
    sug = Suggester(cfg, memory="serve", load_encoder=True)
    stats = apply_feedback(sug, P)
    log.info("feedback applied: %s; serving memory now %d images, %d categories",
             stats, len(sug.memory), len(sug.store))


if __name__ == "__main__":
    main()
