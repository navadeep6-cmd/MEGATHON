import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from catsuggest import geo, split  # noqa: E402
from catsuggest.hierarchy import CategoryGraph, rerank_with_hierarchy  # noqa: E402


def _graph():
    # Pont Neuf -> Bridges in Paris -> Bridges in France -> Bridges ; cycle Bridges <-> Crossings
    return CategoryGraph({
        "Pont Neuf": ["Bridges in Paris", "Île de la Cité"],
        "Bridges in Paris": ["Bridges in France", "Paris"],
        "Bridges in France": ["Bridges"],
        "Bridges": ["Crossings"],
        "Crossings": ["Bridges"],
    }, max_depth=3)


class HierarchyTests(unittest.TestCase):
    def test_ancestors_depth_and_cycles(self):
        g = _graph()
        self.assertEqual(g.ancestors("Pont Neuf"),
                         frozenset({"Bridges in Paris", "Île de la Cité", "Bridges in France", "Paris", "Bridges"}))
        self.assertEqual(g.ancestors("Bridges"), frozenset({"Crossings"}))
        self.assertTrue(g.related("Bridges in France", "Pont Neuf"))
        self.assertTrue(g.related("Pont Neuf", "Bridges in France"))
        self.assertFalse(g.related("Paris", "Bridges"))
        self.assertEqual(g.ancestors("Unknown"), frozenset())

    def test_ancestor_after_descendant_is_suppressed(self):
        g = _graph()
        ranked = [("Pont Neuf", 0.9), ("Bridges in Paris", 0.8), ("Seine", 0.5)]
        full, suppressed = rerank_with_hierarchy(ranked, g, k=2)
        self.assertEqual([c for c, _ in full[:2]], ["Pont Neuf", "Seine"])
        self.assertEqual(suppressed, [("Bridges in Paris", 0.8)])
        self.assertEqual(len(full), 3)

    def test_strong_descendant_replaces_ancestor(self):
        g = _graph()
        ranked = [("Bridges in Paris", 0.8), ("Pont Neuf", 0.7), ("Seine", 0.5)]
        full, suppressed = rerank_with_hierarchy(ranked, g, k=2, min_relative_score=0.5)
        self.assertEqual([c for c, _ in full[:2]], ["Pont Neuf", "Seine"])
        self.assertIn(("Bridges in Paris", 0.8), suppressed)

    def test_weak_descendant_keeps_ancestor(self):
        g = _graph()
        ranked = [("Bridges in Paris", 0.8), ("Pont Neuf", 0.2)]
        full, suppressed = rerank_with_hierarchy(ranked, g, k=2, min_relative_score=0.5)
        self.assertEqual([c for c, _ in full[:2]], ["Bridges in Paris", "Pont Neuf"])
        self.assertEqual(suppressed, [])

    def test_lone_ancestor_kept(self):
        g = _graph()
        full, _ = rerank_with_hierarchy([("Bridges in France", 0.6), ("Seine", 0.4)], g, k=5)
        self.assertEqual([c for c, _ in full], ["Bridges in France", "Seine"])


class SplitTests(unittest.TestCase):
    def _records(self):
        recs = []
        for i in range(100):
            recs.append({"id": i, "timestamp": "2024-01-%02dT00:00:%02dZ" % (1 + i // 10, i % 60),
                         "uploader": "user%d" % (i // 5)})
        return recs

    def test_temporal(self):
        out = split.temporal_split(self._records(), 0.1, 0.1, mode="temporal")
        counts = {s: sum(1 for v in out.values() if v == s) for s in split.SPLITS}
        self.assertEqual(counts, {"train": 80, "val": 10, "test": 10})
        self.assertEqual(out["0"], "train")
        self.assertEqual(out["99"], "test")

    def test_grouped_keeps_uploaders_together(self):
        recs = self._records()
        out = split.temporal_split(recs, 0.1, 0.1, mode="temporal_grouped")
        by_user = {}
        for r in recs:
            by_user.setdefault(r["uploader"], set()).add(out[str(r["id"])])
        self.assertTrue(all(len(s) == 1 for s in by_user.values()))
        for s in split.SPLITS:
            self.assertGreater(sum(1 for v in out.values() if v == s), 0)

    def test_missing_uploader(self):
        recs = [{"id": i, "timestamp": "2024-01-01T00:00:%02dZ" % i, "uploader": None} for i in range(10)]
        out = split.temporal_split(recs, 0.2, 0.2)
        self.assertEqual(len(out), 10)

    def test_bad_fractions(self):
        with self.assertRaises(ValueError):
            split.temporal_split(self._records(), 0.6, 0.5)

    def test_hamming(self):
        self.assertEqual(split.hamming64(0, 0), 0)
        self.assertEqual(split.hamming64(0b1011, 0b0001), 2)
        self.assertEqual(split.hamming64(0, 0xFFFFFFFFFFFFFFFF), 64)
        mask = split.near_duplicate_mask([0b1111, 0xFF00FF00FF00FF00], [0b1110, 0], max_distance=1)
        self.assertEqual(mask, [True, False])


class GeoTests(unittest.TestCase):
    def test_haversine(self):
        d = geo.haversine_km(48.8566, 2.3522, 51.5074, -0.1278)  # Paris - London
        self.assertAlmostEqual(d, 343.5, delta=2.0)
        self.assertAlmostEqual(geo.haversine_km(10, 10, 10, 10), 0.0)

    def test_regions(self):
        cases = {
            (48.86, 2.35): "Europe", (55.75, 37.6): "Europe", (37.39, -6.0): "Europe",
            (36.8, 10.2): "Africa", (30.0, 31.2): "Africa", (6.5, 3.4): "Africa", (-1.3, 36.8): "Africa",
            (31.8, 35.2): "West Asia", (35.7, 51.4): "West Asia", (25.2, 55.3): "West Asia",
            (28.6, 77.2): "Asia", (35.7, 139.7): "Asia", (-6.2, 106.8): "Asia", (55.0, 83.0): "Asia",
            (-33.9, 151.2): "Oceania", (21.3, -157.8): "Oceania", (-36.8, 174.8): "Oceania",
            (40.7, -74.0): "North America", (19.4, -99.1): "North America",
            (-23.5, -46.6): "Latin America", (4.7, -74.0): "Latin America",
            (-75.0, 0.0): "Antarctica", (None, None): "no geotag", (float("nan"), 3.0): "no geotag",
        }
        for (lat, lon), expected in cases.items():
            self.assertEqual(geo.region_of(lat, lon), expected, (lat, lon))


if __name__ == "__main__":
    unittest.main()
