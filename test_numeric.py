"""Tests that need numpy / scikit-learn / torch / Pillow. Skipped when they are missing."""

import os
import random
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None
try:
    import sklearn  # noqa: F401
except ImportError:  # pragma: no cover
    sklearn = None
try:
    import torch
except ImportError:  # pragma: no cover
    torch = None
try:
    from PIL import Image
except ImportError:  # pragma: no cover
    Image = None

CFG = {
    "knn": {"k": 5, "tau": 0.05, "index": "flat", "hnsw_m": 16, "ivf_nlist": 16, "pq_m": 8, "nprobe": 4,
            "ef_search": 32},
    "geo": {"radius_km": 30, "sigma_km": 5, "max_neighbours": 10},
    "candidates": {"per_scorer": 5, "lex_min": 0.5, "lex_max_df": 100},
}


def _unit(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


@unittest.skipIf(np is None, "numpy not installed")
class StoreScorerTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        self.d = 16
        self.names = ["Bridges in Paris", "Diwali 2019 in Delhi", "Cats", "Beaches of Goa"]
        self.protos = _unit(rng.normal(size=(4, self.d))).astype(np.float32)

    def _store_memory(self):
        from catsuggest.store import CategoryStore, LabelMemory

        store = CategoryStore(self.names, self.protos, [3, 2, 1, 0])
        mem = LabelMemory(self.d, "flat")
        # memory images = noisy copies of category prototypes
        rng = np.random.default_rng(1)
        Z = _unit(self.protos[[0, 0, 1, 2]] + 0.05 * rng.normal(size=(4, self.d))).astype(np.float32)
        mem.add(Z, [[0], [0], [1], [2]], ["a", "b", "c", "d"], lats=[48.85, 48.86, None, None],
                lons=[2.35, 2.34, None, None])
        return store, mem

    def test_ann_numpy_and_faiss_agree_with_bruteforce(self):
        from catsuggest.ann import AnnIndex

        rng = np.random.default_rng(2)
        X = _unit(rng.normal(size=(50, self.d))).astype(np.float32)
        q = X[:3]
        idx = AnnIndex(self.d, "flat")
        idx.add(X)
        sims, ids = idx.search(q, 5)
        self.assertEqual(ids[:, 0].tolist(), [0, 1, 2])
        brute = np.argsort(-(q @ X.T), axis=1)[:, :5]
        self.assertEqual(ids.tolist(), brute.tolist())
        sims, ids = idx.search(q, 500)  # k larger than the index
        self.assertEqual(ids.shape, (3, 50))
        with tempfile.TemporaryDirectory() as tmp:
            idx.save(tmp)
            again = type(idx).load(tmp, self.d, "flat")
            self.assertEqual(again.search(q, 5)[1].tolist(), brute.tolist())

    def test_store_add_and_lexical(self):
        store, _ = self._store_memory()
        cid = store.add("Lighthouses in Kerala", self.protos[0] * 3)
        self.assertEqual(cid, 4)
        self.assertEqual(store.add("Lighthouses in Kerala", self.protos[0]), 4)
        self.assertAlmostEqual(float(np.linalg.norm(store.protos[4])), 1.0, places=5)
        from catsuggest.textutil import content_tokens
        lex = store.lexical_candidates(content_tokens("Pont Neuf bridge Paris at dusk"), 100, 0.5, 10)
        self.assertIn(0, lex)
        self.assertAlmostEqual(lex[0], 1.0)

    def test_memory_roundtrip_and_geo(self):
        from catsuggest.store import LabelMemory

        _, mem = self._store_memory()
        near = mem.geo_neighbours(48.857, 2.352, 30, 10)
        self.assertEqual(sorted(r for r, _ in near), [0, 1])
        with tempfile.TemporaryDirectory() as tmp:
            mem.save(tmp)
            again = LabelMemory.load(tmp, "flat")
            self.assertEqual(again.labels, mem.labels)
            self.assertEqual(len(again), 4)
            self.assertEqual(len(again.geo_neighbours(48.857, 2.352, 30, 10)), 2)

    def test_scorer_features(self):
        from catsuggest.fusion import FEATURES
        from catsuggest.suggester import Query, Scorer, rank, single_feature_scores

        store, mem = self._store_memory()
        scorer = Scorer(store, mem, None, CFG)
        z = self.protos[[0]]
        q = Query.from_metadata("File:Pont Neuf bridge in Paris.jpg", "", "2018-05-04", 48.857, 2.352)
        cs = scorer.score(z, np.zeros_like(z), [q])[0]
        self.assertEqual(cs.X.shape, (len(cs.cat_ids), len(FEATURES)))
        self.assertIn(0, cs.cat_ids.tolist())
        top = rank(cs, single_feature_scores(cs, "knn"))[0]
        self.assertEqual(top[0], 0)
        row = cs.X[cs.cat_ids.tolist().index(0)]
        fi = {f: i for i, f in enumerate(FEATURES)}
        self.assertGreater(row[fi["knn"]], 0.5)
        self.assertAlmostEqual(float(row[fi["geo"]]), 1.0, places=5)
        self.assertAlmostEqual(float(row[fi["lex"]]), 1.0, places=5)
        self.assertEqual(float(row[fi["meta"]]), 0.0)  # no metadata embedding supplied
        diwali = cs.cat_ids.tolist().index(1) if 1 in cs.cat_ids.tolist() else None
        if diwali is not None:
            self.assertEqual(float(cs.X[diwali, fi["year_mismatch"]]), 1.0)

    def test_head_numpy_and_map(self):
        from catsuggest.head import LinearHead, sample_map_dense

        head = LinearHead(np.eye(2, self.d, dtype=np.float32), np.zeros(2, np.float32), np.array([0, 2]))
        p = head.probs(np.ones((3, self.d), np.float32))
        self.assertTrue(((p > 0) & (p < 1)).all())
        scores = np.array([[0.9, 0.1, 0.5], [0.1, 0.9, 0.5]])
        Y = np.array([[1, 0, 0], [1, 0, 0]])
        self.assertAlmostEqual(sample_map_dense(scores, Y), (1.0 + 1 / 3) / 2)

    def test_dhash_numpy_matches_python(self):
        from catsuggest.split import near_duplicate_mask, near_duplicate_mask_np

        rng = random.Random(0)
        q = [rng.getrandbits(64) for _ in range(20)]
        r = [rng.getrandbits(64) for _ in range(30)] + [q[3] ^ 0b111]
        self.assertEqual(near_duplicate_mask_np(q, r, 4).tolist(), near_duplicate_mask(q, r, 4))


@unittest.skipIf(np is None or sklearn is None, "numpy/scikit-learn not installed")
class FusionTests(unittest.TestCase):
    def test_fit_predict_save_load(self):
        from catsuggest.fusion import FEATURES, FusionModel

        rng = np.random.default_rng(0)
        X = rng.normal(size=(400, len(FEATURES))).astype(np.float32)
        y = (X[:, 0] + 0.5 * X[:, 5] > 0).astype(int)
        for kind in ("logreg", "gbdt"):
            m = FusionModel.fit(X, y, kind=kind)
            p = m.predict(X)
            self.assertEqual(p.shape, (400,))
            self.assertGreater(float(((p > 0.5) == y).mean()), 0.8)
            with tempfile.TemporaryDirectory() as tmp:
                path = os.path.join(tmp, "f.joblib")
                m.save(path)
                self.assertTrue(np.allclose(FusionModel.load(path).predict(X), p))
        m = FusionModel.fit(X, y, kind="logreg")
        self.assertGreater(m.coefficients()["knn"], 0)
        self.assertEqual(len(m.contributions(X[0])), len(FEATURES))


@unittest.skipIf(np is None or torch is None, "numpy/torch not installed")
class HeadTrainingTests(unittest.TestCase):
    def test_asl_known_negatives_penalised_more(self):
        from catsuggest.head import asl_loss

        logits = torch.tensor([[2.0, 0.2]])
        y = torch.tensor([[0.0, 0.0]])
        plain = asl_loss(logits, y)
        known = asl_loss(logits, y, known_neg=torch.tensor([[1.0, 1.0]]))
        self.assertGreater(float(known), float(plain))
        pos = asl_loss(torch.tensor([[5.0]]), torch.tensor([[1.0]]))
        self.assertLess(float(pos), 0.01)

    def test_train_head_learns(self):
        from catsuggest.head import train_head

        rng = np.random.default_rng(0)
        d, H = 16, 3
        T = _unit(rng.normal(size=(H, d))).astype(np.float32)
        labels = rng.integers(0, H, size=300)
        X = _unit(T[labels] + 0.3 * rng.normal(size=(300, d))).astype(np.float32)
        Y = [[int(c)] for c in labels]
        cfg = {"lr": 0.01, "weight_decay": 0.0, "batch_size": 64, "gamma_pos": 0.0, "gamma_neg": 4.0,
               "margin": 0.05, "epochs": 5, "patience": 3, "text_init_scale": 20.0, "init_bias_offset": 3.0}
        W, b, hist = train_head(X[:200], Y[:200], X[200:], Y[200:], T, cfg)
        self.assertEqual(W.shape, (H, d))
        self.assertGreater(max(h["val_mAP"] for h in hist), 0.8)


@unittest.skipIf(Image is None, "Pillow not installed")
class ImageTests(unittest.TestCase):
    def test_dhash_and_corruptions(self):
        from catsuggest.images import CORRUPTIONS, Corruption, dhash, random_degrade, to_rgb_jpeg_bytes
        import io

        img = Image.new("RGB", (120, 80))
        for x in range(120):
            for y in range(80):
                img.putpixel((x, y), (x * 2, y * 3, (x + y) % 255))
        h1 = dhash(img)
        h2 = dhash(img.resize((240, 160)))
        self.assertLessEqual(bin(h1 ^ h2).count("1"), 4)
        for name in CORRUPTIONS:
            out = Corruption(name)(img, random.Random(0))
            self.assertEqual(out.mode, "RGB")
        self.assertEqual(random_degrade(img, random.Random(1)).mode, "RGB")
        buf = io.BytesIO()
        Image.new("RGBA", (10, 10), (255, 0, 0, 0)).save(buf, format="PNG")
        jpeg = to_rgb_jpeg_bytes(buf.getvalue())
        self.assertEqual(Image.open(io.BytesIO(jpeg)).mode, "RGB")


if __name__ == "__main__":
    unittest.main()
