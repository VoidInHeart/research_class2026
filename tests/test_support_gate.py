from io import BytesIO
import unittest
from unittest.mock import patch

import joblib
import numpy as np

from intentexec.support_gate import SIGNALS, SupportGate


class SupportGateTests(unittest.TestCase):
    def setUp(self):
        self.query = "USER: bus schedule"
        fit_texts = ["missingfirstword\nA\nB\nC\nD\n" + self.query, "bus stop", "bus fare",
                     "arrival time", "listen to rock"]
        rows = [{"text": t, "intent": "bus" if i < 4 else "music"} for i, t in enumerate(fit_texts)]
        fit = np.array([[1, 0], [.8, .6], [.6, .8], [-1, 0], [0, -1]], dtype=np.float32)
        self.reference = ["bus fare", "listen to rock", "noise", self.query]
        ref_rows = [{"intent": "bus", "text": t} for t in self.reference]
        self.ref_vectors = np.array([[1, 0], [0, -1], [-1, 0], [.6, .8]], dtype=np.float32)
        self.gate = SupportGate(rows, fit, ["bus", "music"], ref_rows, self.ref_vectors)

    def test_raw_and_reference_units(self):
        result = self.gate.components([self.query, "newword"], self.ref_vectors[[0, 1]])
        np.testing.assert_allclose(result["centroid"], [-1 / np.sqrt(2), -1], atol=1e-6)
        np.testing.assert_allclose(result["semantic"], [-.8, -1], atol=1e-6)
        self.assertEqual(result["lexical"][1], 0)
        self.assertNotIn("missingfirstword", self.gate.vectorizer.vocabulary_)
        ref = self.gate.components(self.reference, self.ref_vectors)
        for name in SIGNALS:
            q1, median, q3 = np.quantile(ref[name], [.25, .5, .75])
            self.assertAlmostEqual(self.gate.center[name], median)
            scale = q3 - q1 if q3 - q1 >= 1e-6 else 1
            self.assertAlmostEqual(self.gate.scale[name], scale)
            np.testing.assert_allclose(result["z_" + name], (result[name] - median) / scale)

    def test_serve_single_batch_and_serialized_state(self):
        weights = [.5, .25, .25]
        buffer = BytesIO()
        joblib.dump(self.gate, buffer)
        buffer.seek(0)
        loaded = joblib.load(buffer)
        batch = loaded.score(self.reference, self.ref_vectors, weights)
        ref = self.gate.components(self.reference, self.ref_vectors)
        expected = sum(w * ref["z_" + name] for name, w in zip(SIGNALS, weights))
        np.testing.assert_allclose(batch, expected, atol=1e-6)
        for i in range(len(self.reference)):
            single = loaded.score(self.reference[i:i+1], self.ref_vectors[i:i+1], weights)
            np.testing.assert_allclose(single, batch[i:i+1], atol=1e-6)
        with patch.object(loaded.vectorizer, "transform", side_effect=AssertionError("unused lexical channel")):
            np.testing.assert_allclose(loaded.score(self.reference, self.ref_vectors, [1, 0, 0]), ref["z_centroid"])
        with self.assertRaises(ValueError):
            loaded.score(self.reference, self.ref_vectors, [.5, -.5, 1])
        with self.assertRaises(ValueError):
            loaded.score(self.reference[:1], np.zeros((1, 2)), weights)
        self.assertEqual(loaded.score([], self.ref_vectors[:0], weights).shape, (0,))

    def test_constant_reference_and_short_window_lexical_identity(self):
        short = "\n".join(["A", "B", "C", "D", self.query])
        vectors = self.ref_vectors[:1]
        row = [{"intent": "bus", "text": short}]
        gate = SupportGate(row, vectors, ["bus"], row, vectors)
        scores = gate.components(["ignored-preamble\n" + short], vectors)
        np.testing.assert_allclose(scores["lexical"], [-1], atol=1e-6)
        for name in SIGNALS:
            self.assertEqual(gate.scale[name], 1)
            np.testing.assert_allclose(scores["z_" + name], [0], atol=1e-6)
