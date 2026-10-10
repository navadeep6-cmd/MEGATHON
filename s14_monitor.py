#!/usr/bin/env python3
"""Monitoring - acceptance rate and score calibration from editor feedback.

Per category type (object / scene / event / abstract) it reports how often
suggestions are accepted and how well the fused score matches the
acceptance rate (reliability table, expected calibration error). A drift
flag is raised when acceptance falls below monitor.min_accept_rate or ECE
exceeds monitor.max_ece - the trigger for a retrain (s05 -> s07 -> s08).

Output: reports/monitoring.json
"""

import argparse
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from catsuggest.config import Paths, load_config  # noqa: E402
from catsuggest.io_utils import read_jsonl, write_json  # noqa: E402
from catsuggest.metrics import reliability  # noqa: E402


def summarise(events, min_accept, max_ece):
    groups = defaultdict(list)
    for ev in events:
        groups["all"].append(ev)
        groups["type:" + (ev.get("type") or "unknown")].append(ev)
    out = {}
    for name, evs in sorted(groups.items()):
        outcomes = [bool(e["accepted"]) for e in evs]
        rate = sum(outcomes) / float(len(outcomes))
        # calibration only over events that carry the score shown to the editor
        scored = [e for e in evs if e.get("score") is not None]
        rel = reliability([float(e["score"]) for e in scored], [bool(e["accepted"]) for e in scored])
        out[name] = {"events": len(evs), "scored_events": len(scored), "accept_rate": rate,
                     "ece": rel["ece"] if scored else None, "reliability": rel["bins"],
                     "drift": rate < min_accept or (bool(scored) and rel["ece"] > max_ece)}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    P = Paths(cfg).ensure()
    if not os.path.exists(P.feedback_log):
        raise SystemExit("no feedback log yet: %s" % P.feedback_log)
    events = read_jsonl(P.feedback_log)
    if not events:
        raise SystemExit("feedback log is empty")
    mc = cfg["monitor"]
    report = summarise(events, float(mc["min_accept_rate"]), float(mc["max_ece"]))
    report_all = report["all"]
    write_json(P.report("monitoring.json"), report)
    ece = report_all["ece"]
    print("events %d, acceptance %.1f%%, ECE %s, retrain recommended: %s" % (
        report_all["events"], 100 * report_all["accept_rate"], "n/a" if ece is None else "%.3f" % ece,
        any(v["drift"] for v in report.values())))


if __name__ == "__main__":
    main()
