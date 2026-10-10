"""Editor feedback (accept / reject events) - the continual-learning signal.

The serving index has a single writer: the running API or demo process owns
it (lock file with its PID) and applies feedback itself, so its in-memory
index and the files on disk never diverge. scripts/s13_apply_feedback.py
does the same job offline, and refuses to run while a server owns the index.
"""

from __future__ import annotations

import atexit
import datetime as _dt
import os
from collections import OrderedDict
from typing import Dict, Optional

from .io_utils import append_jsonl, ensure_dir, read_json, read_jsonl, write_json


def log_feedback(path: str, query_id: str, category: str, accepted: bool,
                 score: Optional[float] = None, category_type: Optional[str] = None,
                 source: str = "api") -> dict:
    event = {
        "ts": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "query_id": str(query_id),
        "category": str(category),
        "accepted": bool(accepted),
        "score": None if score is None else float(score),
        "type": category_type,
        "source": source,
    }
    append_jsonl(path, event)
    return event


# -- single-writer lock ---------------------------------------------------------

def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if pid == os.getpid():
        return True
    if os.name == "nt":  # os.kill(pid, 0) would send CTRL_C_EVENT on Windows
        import ctypes

        kernel32 = ctypes.windll.kernel32
        handle = kernel32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
        if not handle:
            return False
        try:
            code = ctypes.c_ulong()
            ok = kernel32.GetExitCodeProcess(handle, ctypes.byref(code))
            return bool(ok) and code.value == 259  # STILL_ACTIVE
        finally:
            kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def serving_writer_pid(lock_path: str) -> Optional[int]:
    """PID of the live process that owns the serving index, or None."""
    try:
        with open(lock_path, encoding="utf-8") as f:
            pid = int(f.read().strip() or 0)
    except (OSError, ValueError):
        return None
    return pid if pid > 0 and _pid_alive(pid) else None


def claim_serving_writer(lock_path: str) -> None:
    """Make this process the only writer of the serving index (released at exit)."""
    owner = serving_writer_pid(lock_path)
    if owner is not None and owner != os.getpid():
        raise RuntimeError("the serving index is already used by process %d (API or demo). "
                           "Stop it first, or use that process." % owner)
    ensure_dir(os.path.dirname(os.path.abspath(lock_path)))
    with open(lock_path, "w", encoding="utf-8") as f:
        f.write(str(os.getpid()))
    atexit.register(release_serving_writer, lock_path)


def release_serving_writer(lock_path: str) -> None:
    if serving_writer_pid(lock_path) == os.getpid():
        try:
            os.remove(lock_path)
        except OSError:
            pass


# -- applying feedback ------------------------------------------------------------

def apply_feedback(sug, P) -> Dict[str, int]:
    """Apply new feedback events to the serving index of Suggester `sug`.

    * accepted categories: the cached query embedding becomes a new k-NN
      memory row (or, if that query was already added, the new categories are
      added to its row); unknown names are added to the category store;
    * every event is appended to feedback_labels.jsonl for the next head
      retrain, where rejects act as reliable hard negatives.
    """
    import numpy as np

    if not os.path.exists(P.feedback_log):
        return {"new_events": 0, "images_added": 0, "rows_updated": 0, "not_cached": 0}
    state = read_json(P.feedback_state) if os.path.exists(P.feedback_state) else {}
    offset = int(state.get("offset", 0))
    rows = {str(k): int(v) for k, v in state.get("rows", {}).items()}
    events = read_jsonl(P.feedback_log)
    new = events[offset:]
    stats = {"new_events": len(new), "images_added": 0, "rows_updated": 0, "not_cached": 0}
    if not new:
        return stats

    accepted: "OrderedDict[str, set]" = OrderedDict()
    for ev in new:
        if ev.get("accepted"):
            accepted.setdefault(str(ev["query_id"]), set()).add(ev["category"])

    with sug.lock:
        for qid, cats in accepted.items():
            if qid in rows:
                if sug.add_labels_to_row(rows[qid], sorted(cats)):
                    stats["rows_updated"] += 1
                continue
            zpath = os.path.join(P.query_cache, qid + ".npy")
            if not os.path.exists(zpath):
                stats["not_cached"] += 1
                continue
            mpath = os.path.join(P.query_cache, qid + ".json")
            meta = read_json(mpath) if os.path.exists(mpath) else {}
            rows[qid] = sug.add_labelled_image(np.load(zpath), sorted(cats), title=meta.get("title", ""),
                                               lat=meta.get("lat"), lon=meta.get("lon"))
            stats["images_added"] += 1
        if stats["images_added"] or stats["rows_updated"]:
            sug.save()
    # labels for the next scheduled head retrain, written after the index update succeeded
    for ev in new:
        append_jsonl(P.feedback_labels, {"query_id": ev["query_id"], "category": ev["category"],
                                         "accepted": bool(ev.get("accepted"))})
    write_json(P.feedback_state, {"offset": len(events), "rows": rows})
    return stats
