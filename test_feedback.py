import os
import sys
import tempfile
import threading
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from catsuggest import feedback  # noqa: E402
from catsuggest.io_utils import read_jsonl  # noqa: E402

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None


class LockTests(unittest.TestCase):
    def test_claim_and_release(self):
        with tempfile.TemporaryDirectory() as tmp:
            lock = os.path.join(tmp, "serve.lock")
            self.assertIsNone(feedback.serving_writer_pid(lock))
            feedback.claim_serving_writer(lock)
            self.assertEqual(feedback.serving_writer_pid(lock), os.getpid())
            feedback.claim_serving_writer(lock)  # re-claiming by the owner is fine
            feedback.release_serving_writer(lock)
            self.assertFalse(os.path.exists(lock))

    def test_stale_lock_is_ignored(self):
        with tempfile.TemporaryDirectory() as tmp:
            lock = os.path.join(tmp, "serve.lock")
            with open(lock, "w") as f:
                f.write("999999999")  # no such process
            self.assertIsNone(feedback.serving_writer_pid(lock))
            feedback.claim_serving_writer(lock)
            self.assertEqual(feedback.serving_writer_pid(lock), os.getpid())
            feedback.release_serving_writer(lock)


class MonitorTests(unittest.TestCase):
    def test_summary_tolerates_missing_scores_and_types(self):
        from s14_monitor import summarise

        events = [
            {"accepted": True, "score": 0.9, "type": "object"},
            {"accepted": False, "score": None, "type": None},
            {"accepted": True},
        ]
        out = summarise(events, min_accept=0.3, max_ece=0.15)
        self.assertEqual(out["all"]["events"], 3)
        self.assertEqual(out["all"]["scored_events"], 1)
        self.assertAlmostEqual(out["all"]["accept_rate"], 2 / 3)
        self.assertIn("type:unknown", out)
        self.assertIsNone(out["type:unknown"]["ece"])


class _FakePaths:
    def __init__(self, root):
        self.feedback_log = os.path.join(root, "feedback.jsonl")
        self.feedback_state = os.path.join(root, "state.json")
        self.feedback_labels = os.path.join(root, "labels.jsonl")
        self.query_cache = os.path.join(root, "cache")
        os.makedirs(self.query_cache)


class _FakeSuggester:
    def __init__(self):
        self.lock = threading.RLock()
        self.rows = []
        self.saved = 0

    def add_labelled_image(self, z, categories, title="", lat=None, lon=None):
        self.rows.append(set(categories))
        return len(self.rows) - 1

    def add_labels_to_row(self, row, categories):
        new = set(categories) - self.rows[row]
        self.rows[row] |= new
        return len(new)

    def save(self):
        self.saved += 1


@unittest.skipIf(np is None, "numpy not installed")
class ApplyFeedbackTests(unittest.TestCase):
    def test_second_round_of_accepts_updates_the_same_row(self):
        with tempfile.TemporaryDirectory() as tmp:
            P = _FakePaths(tmp)
            np.save(os.path.join(P.query_cache, "q1.npy"), np.ones(4, np.float32))
            sug = _FakeSuggester()
            feedback.log_feedback(P.feedback_log, "q1", "Bridges in Paris", True, 0.8, "object")
            feedback.log_feedback(P.feedback_log, "q1", "Cats", False, 0.4, "object")
            feedback.log_feedback(P.feedback_log, "q2", "Dogs", True, 0.7, "object")  # not cached
            s1 = feedback.apply_feedback(sug, P)
            self.assertEqual((s1["new_events"], s1["images_added"], s1["not_cached"]), (3, 1, 1))
            self.assertEqual(sug.rows, [{"Bridges in Paris"}])
            # later the editor accepts one more category for the same image
            feedback.log_feedback(P.feedback_log, "q1", "Seine", True, 0.6, "scene")
            s2 = feedback.apply_feedback(sug, P)
            self.assertEqual((s2["new_events"], s2["images_added"], s2["rows_updated"]), (1, 0, 1))
            self.assertEqual(sug.rows, [{"Bridges in Paris", "Seine"}])
            self.assertEqual(feedback.apply_feedback(sug, P)["new_events"], 0)
            self.assertEqual(len(read_jsonl(P.feedback_labels)), 4)


if __name__ == "__main__":
    unittest.main()
