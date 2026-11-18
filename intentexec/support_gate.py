"""Novelty of the recent-five representation relative to known training examples."""
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from .routing import Recognizer, _texts, recent_five


SIGNALS = ("centroid", "semantic", "lexical")
KNOWN_FRR = 29 / 377


def known_budget_cutoff(scores, rate=KNOWN_FRR):
    """Fit a score-only cutoff on known calibration requests; reject strictly above it.

    Round n*rate to the nearest integer (half up). Ties straddling that budget
    are accepted together, so future predictions never require request IDs.
    """
    values = np.asarray(scores, dtype=np.float64)
    if (values.ndim != 1 or not len(values) or not np.isfinite(values).all()
            or not np.isfinite(rate) or not 0 <= rate <= 1):
        raise ValueError("Expected finite known calibration scores and a rejection rate in [0, 1]")
    k = int(np.floor(len(values) * rate + .5))
    if k == len(values):
        return float(np.nextafter(values.min(), -np.inf))
    index = len(values) - 1 - k
    return float(np.partition(values, index)[index])


def _embeddings(values, n, width=None):
    x = np.asarray(values)
    if (x.ndim != 2 or x.shape[0] != n or x.shape[1] == 0
            or (width is not None and x.shape[1] != width)
            or not np.allclose(np.linalg.norm(x, axis=1), 1, rtol=0, atol=1e-4)):
        raise ValueError("Expected aligned, finite, L2-normalized embedding vectors")
    return x


class SupportGate:
    """Known examples define support; an independent known partition sets score units.

    Input vectors are frozen, L2-normalized recent-five embeddings in row order.
    A larger score means less support. The operating threshold is selected separately.
    """

    def __init__(self, fit_rows, fit_vectors, known, reference_rows, reference_vectors):
        fit_rows, reference_rows = list(fit_rows), list(reference_rows)
        fit = _embeddings(fit_vectors, len(fit_rows))
        self.width = fit.shape[1]
        labels = np.asarray([r["intent"] for r in fit_rows])
        self.recognizer = Recognizer().fit_centroids(fit, labels, known)
        if not reference_rows or any(r["intent"] not in self.recognizer.known for r in reference_rows):
            raise ValueError("Score normalization requires nonempty, known-only reference data")
        self.by_class = [fit[labels == label] for label in self.recognizer.known]
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True,
                                          token_pattern=r"(?u)\b\w+\b", dtype=np.float32)
        texts = _texts([r["text"] for r in fit_rows])
        self.lexical_index = self.vectorizer.fit_transform([recent_five(t) for t in texts]).T.tocsc()
        reference = self._raw([r["text"] for r in reference_rows], reference_vectors, SIGNALS)
        self.center, self.scale = {}, {}
        for name, values in reference.items():
            q1, median, q3 = np.quantile(values, [.25, .50, .75])
            self.center[name] = float(median)
            iqr = float(q3 - q1)
            self.scale[name] = iqr if iqr >= 1e-6 else 1.0

    def _raw(self, texts, vectors, active):
        if texts is None:
            if "lexical" in active:
                raise ValueError("Lexical support requires the public input text")
            n = len(vectors)
        else:
            texts = _texts(texts)
            n = len(texts)
        vectors = _embeddings(vectors, n, self.width)
        scores = {name: np.empty(n, dtype=np.float64) for name in active}
        # Preserve the main recognizer's exact batched cosine calculation.
        if "centroid" in active:
            scores["centroid"] = -self.recognizer._cosines(vectors).max(axis=1)
        # Bound temporary query-by-training-example matrices on held-out shards.
        for start in range(0, n, 256):
            part = slice(start, start + 256)
            batch = vectors[part]
            if "semantic" in active:
                strongest = np.full(len(batch), -np.inf)
                for examples in self.by_class:
                    k = min(3, len(examples))
                    similarity = batch @ examples.T
                    similarity.partition(-k, axis=1)
                    strongest = np.maximum(strongest, similarity[:, -k:].mean(axis=1))
                scores["semantic"][part] = -strongest
            if "lexical" in active:
                words = self.vectorizer.transform([recent_five(t) for t in texts[part]])
                scores["lexical"][part] = -(words @ self.lexical_index).max(axis=1).toarray().ravel()
        return scores

    def components(self, texts, vectors):
        raw = self._raw(texts, vectors, SIGNALS)
        scaled = {"z_" + n: (raw[n] - self.center[n]) / self.scale[n] for n in SIGNALS}
        return {**raw, **scaled}

    def score(self, texts, vectors, weights):
        """Combine standardized scores in centroid, semantic, lexical order."""
        w = np.asarray(weights, dtype=np.float64)
        if (w.shape != (3,) or not np.isfinite(w).all() or np.any(w < 0)
                or not np.isclose(w.sum(), 1.0, rtol=0, atol=1e-8)):
            raise ValueError("Expected three nonnegative weights that sum to one")
        active = {n: weight for n, weight in zip(SIGNALS, w) if weight > 0}
        raw = self._raw(texts, vectors, active)
        return sum(weight * (raw[n] - self.center[n]) / self.scale[n] for n, weight in active.items())
