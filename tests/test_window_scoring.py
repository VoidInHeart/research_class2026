import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import joblib
import numpy as np

from intentexec.experiment import infer_one
from intentexec.routing import Recognizer
from intentexec.window_scoring import WindowFeatures, _model_fingerprint, calibrate, weights, window_text


class WindowScoringTests(unittest.TestCase):
    def setUp(self):
        self.fit = [dict(id="f1", text="oldest clue\nu1\ns2\nu3\ns4\nUSER: book a room", intent="book"),
                    dict(id="f2", text="USER: search trains", intent="find")]
        self.cal = [dict(id="c1", text="USER: book a room", intent="book"),
                    dict(id="c2", text="unrelated", intent="unknown")]
        self.known = ["find", "book"]
        self.rows = self.fit + self.cal

    def test_fit_and_calibration_match_recognizer(self):
        data = WindowFeatures(self.rows, "tfidf", None, None)
        for view in ("short", "long"):
            actual = data.score(view, self.fit, {"cal": self.cal, "empty": []}, self.known)
            window = "recent5" if view == "short" else "original"
            rec = Recognizer(window=window).fit(self.fit, self.known).calibrate(self.cal)
            expected_w, expected_gate = rec.predict([r["text"] for r in self.cal])
            threshold, temperature = calibrate(actual["cal"], self.cal, self.known)
            self.assertEqual((threshold, temperature), (rec.threshold, rec.temperature))
            np.testing.assert_allclose(weights(actual["cal"], temperature), expected_w)
            np.testing.assert_equal(actual["cal"].max(axis=1) < threshold, expected_gate)
            np.testing.assert_equal(actual["cal"][1], [0., 0.])
            self.assertEqual(actual["empty"].shape, (0, 2))

    def test_last_five_utterances(self):
        text = "\n".join(f"{i}: utterance" for i in range(7))
        self.assertEqual(window_text(text, "long"), text)
        self.assertEqual(window_text(text, "short").splitlines(), text.splitlines()[-5:])

    def test_default_uses_recent_five_and_executor_receives_full_dialogue(self):
        rec = Recognizer().fit(self.fit, self.known).calibrate(self.cal)
        text = self.fit[0]["text"]
        self.assertNotIn("oldest", rec.vectorizer.vocabulary_)
        np.testing.assert_array_equal(rec.encode([text]).toarray(),
                                      rec.encode([window_text(text, "short")]).toarray())
        router, pool = Mock(), Mock()
        router.choose.return_value = [0]
        bundle = dict(recognizer=rec, experts=["search"], base="base", known=self.known,
                      routers={"matrix": router})
        infer_one(bundle, pool, "matrix_no_gate", text)
        pool.execute.assert_called_once_with("search", text)

    def test_legacy_saved_recognizer_keeps_original_window(self):
        rec = Recognizer(window="original").fit(self.fit, self.known).calibrate(self.cal)
        text = self.fit[0]["text"]
        full = rec.encode([text]).toarray()
        short = rec.encode([window_text(text, "short")]).toarray()
        self.assertFalse(np.array_equal(full, short))
        del rec.window
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "legacy.joblib"
            joblib.dump(rec, path)
            restored = joblib.load(path)
        self.assertEqual(restored.window, "original")
        np.testing.assert_array_equal(restored.encode([text]).toarray(), full)

    def test_embedding_cache_is_content_and_model_specific(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            model = root / "tiny"
            model.mkdir()
            (model / "config.json").write_text("{}")
            model_file = model / "model.safetensors"
            model_file.write_bytes(b"model A")
            cache = root / "cache"
            features = np.eye(4, dtype=np.float32)
            with patch("intentexec.window_scoring.version", return_value="test"):
                first = WindowFeatures(self.rows, "minilm", model, cache)
                with patch.object(first.rec, "encode", return_value=features) as encode:
                    first.score("long", self.fit, {"cal": self.cal}, self.known)
                    first.score("long", self.fit, {"cal": self.cal}, self.known)
                    self.assertEqual(encode.call_count, 1)
                reopened = WindowFeatures(reversed(self.rows), "minilm", model, cache)
                with patch.object(reopened.rec, "encode", side_effect=AssertionError):
                    reopened.score("long", self.fit, {"cal": self.cal}, self.known)
                changed = [dict(r) for r in self.rows]
                changed[-1]["text"] += " updated"
                other = WindowFeatures(changed, "minilm", model, cache)
                with patch.object(other.rec, "encode", return_value=features) as encode:
                    other.score("long", self.fit, {"cal": changed[-2:]}, self.known)
                    encode.assert_called_once()
                model_file.write_bytes(b"model B")
                self.assertNotEqual(first.model_fingerprint, _model_fingerprint(model))
