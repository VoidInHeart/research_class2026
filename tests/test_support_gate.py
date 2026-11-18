from io import BytesIO
import unittest
from unittest.mock import Mock, patch

import joblib
import numpy as np

from intentexec.routing import Recognizer
from intentexec.experiment import fit_support
from intentexec.support_gate import SIGNALS, SupportGate, known_budget_cutoff


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

    def test_known_error_budget_and_conservative_ties(self):
        scores = np.arange(377)
        cut = known_budget_cutoff(scores)
        self.assertEqual(np.count_nonzero(scores > cut), 29)
        tied = [4, 3, 3, 1]
        self.assertEqual(known_budget_cutoff(tied, .5), 3.)
        self.assertEqual(known_budget_cutoff(tied, 0), 4.)
        self.assertTrue(np.all(np.asarray(tied) > known_budget_cutoff(tied, 1)))
        for data, rate in [([], .1), ([1, np.nan], .2), ([1], -1)]:
            with self.assertRaises(ValueError):
                known_budget_cutoff(data, rate)

    def test_recognizer_reuses_intent_scores_and_requires_text_only_for_lexical_weights(self):
        rec = Recognizer(encoder="minilm")
        labels = ["bus", "music", "bus", "bus"]
        known = ["bus", "music"]
        rows = [{"text": t, "intent": label} for t, label in zip(self.reference, labels)]
        rec.fit_centroids(self.ref_vectors, labels, known)
        original = np.vstack([self.ref_vectors, self.ref_vectors[-1:]])
        rec.calibrate_scores(rec._cosines(original), rows + [{"intent": "unseen"}])
        old_weights, _ = rec.predict_encoded(self.ref_vectors)
        rec.calibrate_support(self.gate, [0, 1, 0], rows, self.ref_vectors)
        weights, _ = rec.predict_encoded(self.ref_vectors)
        np.testing.assert_array_equal(old_weights, weights)
        with self.assertRaisesRegex(ValueError, "known-only"):
            rec.calibrate_support(self.gate, [0, 1, 0], rows + [{"intent": "unseen"}], original)
        scores = rec.calibrate_support(self.gate, [.25, .25, .5], rows, self.ref_vectors)
        with self.assertRaisesRegex(ValueError, "input text"):
            rec.predict_encoded(self.ref_vectors)
        buf = BytesIO()
        joblib.dump(rec, buf)
        buf.seek(0)
        loaded = joblib.load(buf)
        with patch.object(loaded, "encode", return_value=self.ref_vectors):
            weights, rejected = loaded.predict(iter(self.reference))
        np.testing.assert_array_equal(old_weights, weights)
        np.testing.assert_array_equal(rejected, scores > loaded.support_cutoff)

    def test_adaptive_training_reads_only_known_calibration_requests(self):
        fit = [{"intent": "bus", "text": "bus timetable"}, {"intent": "music", "text": "rock song"}]
        router = [{"intent": "bus", "text": "bus schedule"}, {"intent": "music", "text": "rock music"}]
        cal = [{"intent": "bus", "text": "bus time"}, {"intent": "held out", "text": "hidden request"}]
        vectors = {r["text"]: self.ref_vectors[i] for rows in (fit, router, cal[:1]) for i, r in enumerate(rows)}
        rec = Recognizer(encoder="minilm")
        catalog = Mock()
        catalog.read_bytes.return_value = b"{}"
        active = {"d": {"bus", "music"}}
        decision = {"candidate": "semantic_mix", "weights": [.25, .5, .25]}
        with (patch.object(rec, "encode", side_effect=lambda ts: np.stack([vectors[t] for t in ts])),
              patch("intentexec.window_splits._dialogue_labels", return_value=active),
              patch("intentexec.support_selection.select_weights", return_value=decision) as select):
            _, info, chosen = fit_support(rec, {"fit": fit, "known": ["bus", "music"], "cal": cal},
                                          router, None, catalog)
        self.assertEqual(select.call_args.args[0], fit)
        self.assertEqual(select.call_args.args[3], router)
        self.assertIs(select.call_args.kwargs["dialogue_intents"], active)
        self.assertEqual(chosen, decision)
        self.assertEqual(info["selected_candidate"], "semantic_mix")
        self.assertEqual(info["known_calibration_rows"], 1)
        self.assertEqual(info["known_calibration_rejected"], 0)

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
