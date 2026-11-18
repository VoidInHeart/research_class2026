"""Targeted regressions for open-set gating and execution supervision."""
import tempfile
import unittest

import joblib
import numpy as np

from intentexec.routing import EmbeddingRouter, Recognizer, Router


def row(text, intent):
    return {"text": text, "intent": intent}


class RoutingTests(unittest.TestCase):
    def setUp(self):
        self.known = ["reserve", "find"]
        self.fit = [row("book dinner", "reserve"), row("book a table", "reserve"),
                    row("search menu", "find"), row("search restaurant", "find")]
        self.cal = [row("book dinner", "reserve"), row("search menu", "find"),
                    row("flights seats", "train-unknown")]
        self.rec = Recognizer().fit(self.fit, self.known).calibrate(self.cal)

    def test_oos_uses_absolute_scores_and_preserves_known_order(self):
        weights, rejected = self.rec.predict(["book dinner", "search menu", "totally absent"])
        np.testing.assert_equal(weights.argmax(axis=1)[:2], [0, 1])
        np.testing.assert_allclose(weights[2], [0.5, 0.5])
        np.testing.assert_allclose(weights.sum(axis=1), 1)
        np.testing.assert_equal(rejected, [False, False, True])
        self.assertEqual(self.rec.predict([])[0].shape, (0, 2))

    def test_matrix_uses_tail_intent_mass_and_multi_correctness(self):
        labels = ["a"] * 4 + ["b"] * 4
        train_w = np.eye(2)[[0] * 4 + [1] * 4]
        # Both-correct and both-fail requests are valid observations.
        y = [[1, 1], [1, 0], [0, 0], [1, 0]] + [[0, 1]] * 4
        w = [[.99, .01], [.51, .49]]
        model = Router("matrix").fit(train_w, labels, ["a", "b"], y)
        np.testing.assert_equal(model.choose(w), [0, 1])
        hard = Router("lookup").fit(train_w, labels, ["a", "b"], y)
        np.testing.assert_equal(hard.choose(w), [0, 0])
        self.assertEqual(model.choose(np.empty((0, 2))).shape, (0,))

    def test_constant_and_single_executor_do_not_require_a_unique_winner(self):
        w = [[1, 0], [0, 1], [1, 0], [0, 1]]
        labels = ["a", "b"] * 2
        for y in ([[0, 0]] * 4, [[1, 1]] * 4, [[1], [0], [1], [0]]):
            model = Router("mlp").fit(w, labels, ["a", "b"], y)
            self.assertEqual(model.scores(w).shape, (4, len(y[0])))
            np.testing.assert_equal(model.choose(w), [0] * 4)

    def test_supervision_rejects_unknown_and_wrong_width(self):
        with self.assertRaises(ValueError):
            Router("fixed").fit([[1, 0]], ["unknown"], self.known, [[1, 0]])
        with self.assertRaises(ValueError):
            Router("fixed").fit([[1, 1]], ["reserve"], self.known, [[1, 0]])
        with self.assertRaises(ValueError):
            self.rec.calibrate(self.fit)
        with self.assertRaises(ValueError):
            Recognizer().fit(self.fit, ["find"])

    def test_shared_features_and_serialization(self):
        texts = [r["text"] for r in self.fit]
        f = self.rec.encode(texts)
        success = np.array([[1, 0], [1, 0], [0, 1], [0, 1]])
        raw = EmbeddingRouter().fit(f, success)
        np.testing.assert_equal(raw.choose(f), [0, 0, 1, 1])
        scores, gate = self.rec.predict_encoded(f)
        router = Router("matrix").fit(scores, [r["intent"] for r in self.fit], self.known, success)
        with tempfile.TemporaryDirectory() as d:
            path = d + "/model.joblib"
            joblib.dump((self.rec, router, raw), path)
            recog2, router2, raw2 = joblib.load(path)
            scores2, gate2 = recog2.predict(texts)
            np.testing.assert_equal(gate2, gate)
            np.testing.assert_allclose(scores2, scores)
            np.testing.assert_equal(router2.choose(scores2), router.choose(scores))
            np.testing.assert_equal(raw2.choose(f), raw.choose(f))


if __name__ == "__main__":
    unittest.main()

