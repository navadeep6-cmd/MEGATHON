import math
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from catsuggest import metrics, scoring  # noqa: E402


class MetricTests(unittest.TestCase):
    def test_precision_recall_f1(self):
        p, r = metrics.precision_recall_at_k(["a", "x", "b", "y"], {"a", "b", "c"}, 2)
        self.assertAlmostEqual(p, 0.5)
        self.assertAlmostEqual(r, 1 / 3)
        self.assertAlmostEqual(metrics.f1(0.5, 1 / 3), 0.4)
        self.assertEqual(metrics.f1(0.0, 0.0), 0.0)

    def test_average_precision(self):
        # hits at ranks 1 and 3; third label never retrieved
        ap = metrics.average_precision(["a", "x", "b"], {"a", "b", "c"})
        self.assertAlmostEqual(ap, (1.0 + 2 / 3) / 3)
        self.assertEqual(metrics.average_precision([], {"a"}), 0.0)

    def test_sample_metrics(self):
        out = metrics.sample_metrics([["a", "b"], ["x", "c"], ["z"]], [{"a"}, {"c"}, set()], ks=[1, 2])
        self.assertEqual(out["n_images"], 2)
        self.assertAlmostEqual(out["P@1"], 0.5)
        self.assertAlmostEqual(out["R@2"], 1.0)
        self.assertAlmostEqual(out["P@2"], 0.5)
        self.assertAlmostEqual(out["mAP"], (1.0 + 0.5) / 2)

    def test_macro_metrics(self):
        rankings = [["a", "b"], ["b", "a"], ["c"]]
        scores = [[0.9, 0.1], [0.8, 0.3], [0.7]]
        truths = [{"a"}, {"a"}, {"b"}]
        out = metrics.macro_metrics(rankings, scores, truths, ks=[1])
        # category a: images ranked by score 0.9 (pos), 0.3 (pos) -> AP 1.0
        # category b: scores 0.8 (neg), 0.1 (neg); its positive (image 3) never retrieved -> AP 0
        self.assertAlmostEqual(out["macro_mAP"], 0.5)
        # top-1: a predicted for img0 (tp); b predicted for img1 (fp); a: P=1,R=0.5; b: P=0,R=0
        self.assertAlmostEqual(out["macro_P@1"], 0.5)
        self.assertAlmostEqual(out["macro_R@1"], 0.25)
        self.assertEqual(out["n_categories"], 2)
        sub = metrics.macro_metrics(rankings, scores, truths, ks=[1], categories=["a"])
        self.assertAlmostEqual(sub["macro_mAP"], 1.0)

    def test_hierarchical_precision(self):
        related = lambda a, b: {a, b} == {"Bridges", "Pont Neuf"}  # noqa: E731
        hp = metrics.hierarchical_precision_at_k([["Bridges", "Seine"]], [{"Pont Neuf"}], 2, related, 0.5)
        self.assertAlmostEqual(hp, 0.25)

    def test_candidate_recall(self):
        self.assertAlmostEqual(metrics.candidate_recall([["a", "b"], ["c"]], [{"a", "z"}, {"c"}]), 2 / 3)

    def test_reliability(self):
        out = metrics.reliability([0.95, 0.95, 0.05, 0.05], [True, True, False, False], n_bins=10)
        self.assertAlmostEqual(out["ece"], 0.05)
        self.assertEqual(out["n"], 4)


class ScoringTests(unittest.TestCase):
    def test_knn_vote_matches_formula(self):
        sims = [0.9, 0.8, 0.5]
        labels = [[1, 2], [2], [3, 3]]
        tau = 0.1
        out = scoring.knn_vote(sims, labels, tau)
        w = [math.exp(s / tau) for s in sims]
        den = sum(w)
        self.assertAlmostEqual(out[1][0], w[0] / den)
        self.assertAlmostEqual(out[2][0], (w[0] + w[1]) / den)
        self.assertAlmostEqual(out[3][0], w[2] / den)  # duplicate label counted once
        self.assertEqual(out[2][2], 2)
        self.assertAlmostEqual(out[2][1], 0.9)
        self.assertEqual(scoring.knn_vote([], [], 0.1), {})

    def test_geo_vote(self):
        out = scoring.geo_vote([0.0, 10.0], [[1], [2]], sigma_km=10.0)
        self.assertAlmostEqual(out[1], 1 / (1 + math.exp(-1)))
        self.assertAlmostEqual(out[1] + out[2], 1.0)

    def test_lexical_and_year(self):
        self.assertAlmostEqual(scoring.lexical_overlap(frozenset({"bridge", "pari"}), frozenset({"pari"})), 0.5)
        self.assertEqual(scoring.lexical_overlap(frozenset(), frozenset({"x"})), 0.0)
        self.assertEqual(scoring.year_features(2019, 2019), (1.0, 0.0))
        self.assertEqual(scoring.year_features(2019, 2018), (0.0, 1.0))
        self.assertEqual(scoring.year_features(None, 2018), (0.0, 0.0))

    def test_describe(self):
        text = scoring.describe_signals({"knn_frac": 0.1, "knn_maxsim": 0.87, "lex": 1.0}, 50)
        self.assertIn("5 of 50", text)
        self.assertIn("title", text)
        self.assertEqual(scoring.describe_signals({}, 50), "weak combined evidence")


if __name__ == "__main__":
    unittest.main()
