#!/usr/bin/env python3
"""Step 10 - efficiency: latency, throughput, index size and memory.

* end-to-end latency per image (decode -> encode image + metadata -> ANN ->
  scorers -> fusion -> hierarchy), p50 / p95, batch size 1;
* batched throughput of the encoder (the dominant cost);
* index size and process peak memory;
* projection to 10^8 images (flat fp16 vs product quantisation).

Output: reports/benchmark.json
"""

import argparse
import logging
import os
import platform
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from catsuggest.config import Paths, load_config  # noqa: E402
from catsuggest.images import load_rgb  # noqa: E402
from catsuggest.io_utils import read_jsonl, setup_logging, write_json  # noqa: E402
from catsuggest.suggester import Suggester  # noqa: E402

log = logging.getLogger("s10")


def peak_rss_mb():
    """Peak resident memory in MB (None on Windows, where the resource module is unavailable)."""
    try:
        import resource
    except ImportError:
        return None
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / 1e6 if sys.platform == "darwin" else peak / 1e3  # bytes on macOS, KiB on Linux


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--n", type=int, default=200, help="images for the latency test")
    args = ap.parse_args()
    setup_logging()
    cfg = load_config(args.config)
    P = Paths(cfg).ensure()

    sug = Suggester(cfg, memory="eval")
    recs = [r for r in read_jsonl(P.dataset) if r["split"] == "test"][:args.n]
    if not recs:
        raise SystemExit("no test records")

    # warm-up
    sug.suggest(load_rgb(P.image_path(recs[0]["id"])), title=recs[0]["title"])
    lat = []
    for r in recs:
        t0 = time.perf_counter()
        img = load_rgb(P.image_path(r["id"]))
        sug.suggest(img, title=r["title"], description=r["description"], lat=r.get("lat"), lon=r.get("lon"))
        lat.append((time.perf_counter() - t0) * 1000.0)
    lat = np.asarray(lat)

    images = [load_rgb(P.image_path(r["id"])) for r in recs[:min(len(recs), 128)]]
    t0 = time.perf_counter()
    sug.encoder.encode_pil(images)
    enc_tput = len(images) / (time.perf_counter() - t0)

    dim = sug.memory.dim
    n_index = len(sug.memory)
    result = {
        "machine": {"platform": platform.platform(), "processor": platform.processor(),
                    "device": sug.encoder.device, "python": platform.python_version()},
        "encoder": "%s / %s" % (cfg["encoder"]["model"], cfg["encoder"]["pretrained"]),
        "latency_ms_end_to_end": {"p50": float(np.percentile(lat, 50)), "p95": float(np.percentile(lat, 95)),
                                  "mean": float(lat.mean()), "n": int(lat.size)},
        "single_stream_images_per_s": float(1000.0 / lat.mean()),
        "encoder_batched_images_per_s": float(enc_tput),
        "index": {"vectors": n_index, "dim": dim, "kind": sug.memory.ann.kind, "backend": sug.memory.ann.backend,
                  "bytes": sug.memory.ann.nbytes()},
        "categories": len(sug.store),
        "peak_rss_mb": peak_rss_mb(),
        "projection_1e8_images": {
            "flat_fp16_GB": 1e8 * dim * 2 / 1e9,
            "pq_64B_GB": 1e8 * 64 / 1e9,
            "daily_uploads_1e5_at_measured_single_stream_hours": 1e5 / (1000.0 / lat.mean()) / 3600.0,
        },
    }
    write_json(P.report("benchmark.json"), result)
    log.info("latency p50 %.1f ms, p95 %.1f ms; encoder %.1f img/s; index %.1f MB",
             result["latency_ms_end_to_end"]["p50"], result["latency_ms_end_to_end"]["p95"],
             enc_tput, result["index"]["bytes"] / 1e6)


if __name__ == "__main__":
    main()
