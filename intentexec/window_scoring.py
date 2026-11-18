"""Windowed intent scores. Frozen MiniLM embeddings are shared across holdouts."""
import hashlib
import inspect
import json
from importlib.metadata import version
from pathlib import Path
import tempfile
from zipfile import BadZipFile

import numpy as np
from scipy.special import softmax

from .routing import Recognizer, _intent_order, _local_snapshot, recent_five


def window_text(text, view):
    if view == "long":
        return text
    if view == "short":
        return recent_five(text)
    raise ValueError(f"Unknown window: {view}")


def _model_fingerprint(model_path):
    snapshot = _local_snapshot(model_path)
    digest = hashlib.sha256()
    for path in sorted(snapshot.rglob("*")):
        if path.is_file() and path.suffix.lower() in {".json", ".txt", ".model", ".safetensors", ".bin"}:
            digest.update(path.relative_to(snapshot).as_posix().encode() + b"\0")
            with path.open("rb") as handle:
                digest.update(hashlib.file_digest(handle, "sha256").digest())
    # A changed truncation policy, pooling implementation or runtime invalidates old embeddings.
    for value in (inspect.getsource(Recognizer.encode), inspect.getsource(Recognizer._load_minilm),
                  version("torch"), version("transformers"), version("tokenizers")):
        digest.update(value.encode() + b"\0")
    return digest.hexdigest()


def calibrate(cal_scores, cal_rows, known):
    rec = Recognizer()
    rec.known = _intent_order(known)
    rec.calibrate_scores(cal_scores, cal_rows)
    return rec.threshold, rec.temperature


def weights(scores, temperature):
    return softmax(np.asarray(scores, dtype=np.float64) / temperature, axis=1)


class WindowFeatures:
    """The vocabulary and centroids use only this fold's fit rows. Inputs keep stable IDs."""

    def __init__(self, rows, encoder, model_path, cache_dir):
        self.rec = Recognizer(encoder, model_path, window="original")
        rows = sorted(rows, key=lambda row: row["id"])
        self.index = {row["id"]: i for i, row in enumerate(rows)}
        if not rows or len(self.index) != len(rows):
            raise ValueError("Window features require nonempty rows with distinct IDs.")
        self.texts = [row["text"] for row in rows]
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        self._encoded = {}
        self.model_fingerprint = _model_fingerprint(model_path) if encoder == "minilm" else None

    def _embedding_matrix(self, view):
        if view in self._encoded:
            return self._encoded[view]
        texts = [window_text(text, view) for text in self.texts]
        digest = hashlib.sha256(json.dumps(
            [self.model_fingerprint, view, list(self.index), texts],
            ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()
        path = self.cache_dir / f"minilm-{view}-{digest}.npz" if self.cache_dir else None
        if path is not None and path.is_file():
            try:
                with np.load(path, allow_pickle=False) as saved:
                    features = saved["features"]
                    if (str(saved["digest"]) == digest and features.ndim == 2
                            and features.shape[0] == len(texts) and features.shape[1] > 0
                            and features.dtype == np.float32 and np.isfinite(features).all()):
                        self._encoded[view] = features
                        return features
            except (OSError, ValueError, EOFError, KeyError, BadZipFile):
                pass
        features = self.rec.encode(texts)
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            temp = tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npz", delete=False)
            try:
                with temp:
                    np.savez(temp, features=features, digest=digest)
                Path(temp.name).replace(path)
            finally:
                Path(temp.name).unlink(missing_ok=True)
        self._encoded[view] = features
        return features

    def score(self, view, fit, phases, known):
        fit = list(fit)
        fit_indices = [self.index[row["id"]] for row in fit]
        phase_indices = {part: [self.index[row["id"]] for row in rows] for part, rows in phases.items()}
        if self.rec.encoder == "tfidf":
            rec = Recognizer("tfidf", window="original")
            rec.fit([{"text": window_text(self.texts[i], view), "intent": row["intent"]}
                     for i, row in zip(fit_indices, fit)], known)
            return {part: rec._cosines(rec.encode([window_text(self.texts[i], view) for i in indices]))
                    for part, indices in phase_indices.items()}
        features = self._embedding_matrix(view)
        self.rec.fit_centroids(features[fit_indices], [row["intent"] for row in fit], known)
        return {part: self.rec._cosines(features[indices]) for part, indices in phase_indices.items()}
