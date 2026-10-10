#!/usr/bin/env python3
"""Step 12 (optional, needs a human) - strict vs reviewed score.

Stored Commons labels are incomplete, so some "wrong" suggestions are in fact
correct. Fill the last column of reports/review_sample.csv with y / n, then
run this script. With f = fraction of sampled wrong top-k suggestions judged
correct, the reviewed precision is estimated as

    P_reviewed@k = P_strict@k + (1 - P_strict@k) * f        (k <= review_k)

Output: reports/review.json
"""

import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from catsuggest.config import Paths, load_config  # noqa: E402
from catsuggest.io_utils import read_json, write_json  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    P = Paths(cfg)
    yes = no = 0
    with open(P.report("review_sample.csv"), encoding="utf-8") as f:
        for row in csv.DictReader(f):
            v = (row.get("reviewer_says_correct") or "").strip().lower()
            if v in ("y", "yes", "1", "true"):
                yes += 1
            elif v in ("n", "no", "0", "false"):
                no += 1
    if yes + no == 0:
        raise SystemExit("no reviewed rows - fill the reviewer_says_correct column with y/n first")
    frac = yes / float(yes + no)
    metrics = read_json(P.report("metrics.json"))
    sample = metrics["systems"][metrics["main_system"]]["sample"]
    k_rev = int(cfg["eval"]["review_k"])
    adjusted = {}
    for k in cfg["eval"]["ks"]:
        if int(k) > k_rev:
            continue
        p = sample["P@%d" % k]
        adjusted["P@%d" % k] = {"strict": p, "reviewed": p + (1 - p) * frac}
    write_json(P.report("review.json"), {"reviewed": yes + no, "fraction_actually_correct": frac,
                                         "review_k": k_rev, "adjusted": adjusted})
    print("reviewed %d rows; %.1f%% of 'wrong' suggestions were correct" % (yes + no, 100 * frac))


if __name__ == "__main__":
    main()
