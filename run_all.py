#!/usr/bin/env python3
"""Run the whole pipeline (works the same on macOS, Linux and Windows).

    python run_all.py --quick    # smoke test: 10 seeds x 150 files (~15-30 min)
    python run_all.py            # full run: ~25k images (~1-2 h, mostly downloading)

Stops at the first failing step and prints which one failed.
"""

import argparse
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))


def run(args, label):
    print("\n=== %s ===" % label, flush=True)
    start = time.time()
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    code = subprocess.call([sys.executable] + args, cwd=ROOT, env=env)
    if code != 0:
        print("\nFAILED at: %s (exit code %d). Copy the error above and ask for help." % (label, code))
        sys.exit(code)
    print("--- %s finished in %.0f s" % (label, time.time() - start), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true", help="small smoke-test run")
    args = ap.parse_args()
    if not os.environ.get("COMMONS_CONTACT"):
        sys.exit("Please set COMMONS_CONTACT first (your GitHub profile link or project URL); "
                 "Wikimedia requires a contact in the User-Agent.")

    fetch = ["scripts/s01_fetch_commons.py"]
    if args.quick:
        fetch += ["--max-seeds", "10", "--max-files-per-seed", "150"]
    steps = [
        (["-m", "unittest", "discover", "-s", "tests"], "unit tests"),
        (fetch, "1/12 fetch from Wikimedia Commons"),
        (["scripts/s02_download_images.py"], "2/12 download thumbnails"),
        (["scripts/s03_build_dataset.py"], "3/12 clean labels and split"),
        (["scripts/s04_embed.py"], "4/12 compute embeddings"),
        (["scripts/s05_train_head.py"], "5/12 train supervised head"),
        (["scripts/s06_build_index.py"], "6/12 build k-NN index"),
        (["scripts/s07_fit_fusion.py"], "7/12 fit fusion model"),
        (["scripts/s13_apply_feedback.py"], "8/12 re-apply editor feedback"),
        (["scripts/s08_evaluate.py"], "9/12 evaluate on new images"),
        (["scripts/s09_robustness.py"], "10/12 robustness test"),
        (["scripts/s10_benchmark.py"], "11/12 speed benchmark"),
        (["scripts/s11_report.py"], "12/12 write reports/RESULTS.md"),
    ]
    for step_args, label in steps:
        run(step_args, label)
    print("\nDone. Results: reports/RESULTS.md   Demo: python app/demo.py")


if __name__ == "__main__":
    main()
