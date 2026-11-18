import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from intentexec.routing import Recognizer
from intentexec.window_scoring import WindowFeatures, _model_fingerprint, calibrate, weights, window_text


class WindowScoringTests(unittest.TestCase):
    def setUp(self):
        self.fit = [dict(id="f1", text="USER: book a room", intent="book"),
                    dict(id="f2", text="USER: search trains", intent="find")]
        self.cal = [dict(id="c1", text="USER: book a room", intent="book"),
                    dict(id="c2", text="unrelated", intent="unknown")]
        self.known = ["find", "book"]
        self.rows = self.fit + self.cal

    def test_fit_and_calibration_match_recognizer(self):
        data = WindowFeatures(self.rows, "tfidf", None, None)
        for view in ("short", "long"):
            actual = data.score(view, self.fit, {"cal": self.cal, "empty": []}, self.known)
            rec = Recognizer().fit(self.fit, self.known).calibrate(self.cal)
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
