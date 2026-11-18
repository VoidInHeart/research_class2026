"""Open-set intent recognition followed by an intent-score-only expert router.

Score columns follow the supplied known-intent list. Expert columns follow the
supplied execution outcome matrix. Final inference requires neither gold
intent labels nor outcomes from the request being routed.
"""
from pathlib import Path
from collections.abc import Mapping

import numpy as np
from scipy.special import softmax
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import normalize


def _intent_order(known):
    labels = list(known)
    if not labels or len(set(labels)) != len(labels):
        raise ValueError("known must be a nonempty, ordered list of distinct intent labels")
    if isinstance(known, (set, frozenset)) or any(not isinstance(s, str) or not s for s in labels):
        raise ValueError("known must preserve a stable order of nonempty string labels")
    return labels


def _texts(items):
    if isinstance(items, (str, bytes, Mapping)):
        raise TypeError("expected a sequence of request texts")
    values = list(items)
    if any(not isinstance(x, str) or not x.strip() for x in values):
        raise ValueError("each request text must be a nonempty string")
    return values


def _local_snapshot(path):
    if path is None:
        raise ValueError("encoder=minilm requires a local model_path")
    location = Path(path).expanduser().resolve()
    if (location / "config.json").is_file():
        return location
    ref = location / "refs" / "main"
    if ref.is_file():
        revision = ref.read_text(encoding="utf-8").strip()
        if revision and Path(revision).name == revision:
            candidate = location / "snapshots" / revision
            if (candidate / "config.json").is_file():
                return candidate
    # An offline copy may omit refs/main. A single snapshot is unambiguous.
    snapshots = list((location / "snapshots").glob("*/config.json"))
    if len(snapshots) == 1:
        return snapshots[0].parent
    raise FileNotFoundError(f"no local model config at {location}; pass a downloaded snapshot directory")


class Recognizer:
    """Known-class centroid cosine scores, with independently calibrated OOS gate."""

    def __init__(self, encoder="tfidf", model_path=None):
        if encoder not in ("tfidf", "minilm"):
            raise ValueError("encoder must be tfidf or minilm")
        self.encoder = encoder
        self.model_path = str(Path(model_path).expanduser().resolve()) if model_path is not None else None
        self.threshold = None
        self.temperature = None
        self._minilm = None

    def _load_minilm(self):
        if self._minilm is None:
            from transformers import AutoModel, AutoTokenizer

            snapshot = _local_snapshot(self.model_path)
            tokenizer = AutoTokenizer.from_pretrained(str(snapshot), local_files_only=True,
                                                       trust_remote_code=False)
            tokenizer.truncation_side = "left"  # preserve the latest user turn
            model = AutoModel.from_pretrained(str(snapshot), local_files_only=True,
                                              trust_remote_code=False).eval()
            self._minilm = (tokenizer, model)
        return self._minilm

    def __getstate__(self):
        # A serialized recognizer reloads the frozen local encoder on demand.
        state = self.__dict__.copy()
        state["_minilm"] = None
        return state

    def encode(self, texts):
        """Return normalized raw text features, reusable by an embedding baseline."""
        values = _texts(texts)
        if self.encoder == "tfidf":
            if not hasattr(self, "vectorizer"):
                raise RuntimeError("fit the recognizer before encoding with TF-IDF")
            if not values:
                from scipy.sparse import csr_matrix
                return csr_matrix((0, len(self.vectorizer.vocabulary_)), dtype=np.float32)
            return self.vectorizer.transform(values)
        import torch

        tokenizer, model = self._load_minilm()
        if not values:
            return np.empty((0, model.config.hidden_size), dtype=np.float32)
        batches = []
        with torch.inference_mode():
            for start in range(0, len(values), 32):
                tokens = tokenizer(values[start:start + 32], padding=True, truncation=True,
                                   max_length=256, return_tensors="pt")
                output = model(**tokens).last_hidden_state
                mask = tokens["attention_mask"].unsqueeze(-1).to(output.dtype)
                pooled = (output * mask).sum(1) / mask.sum(1).clamp_min(1)
                batches.append(torch.nn.functional.normalize(pooled, dim=1).cpu().numpy())
        return np.concatenate(batches)

    def fit(self, rows, known):
        rows = list(rows)
        self.known = _intent_order(known)
        labels = np.asarray([r["intent"] for r in rows])
        if set(labels) != set(self.known):
            raise ValueError("fit rows must contain all and only the declared known intents")
        texts = _texts([r["text"] for r in rows])
        if self.encoder == "tfidf":
            self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True,
                                               token_pattern=r"(?u)\b\w+\b", dtype=np.float32)
            self.vectorizer.fit(texts)
        features = self.encode(texts)
        means = np.vstack([np.asarray(features[labels == label].mean(axis=0)).ravel()
                           for label in self.known])
        self.centroids = normalize(means, axis=1)
        self.threshold = self.temperature = None
        return self

    def _cosines(self, features):
        if not hasattr(self, "centroids"):
            raise RuntimeError("fit the recognizer before scoring")
        if features.ndim != 2 or features.shape[1] != self.centroids.shape[1]:
            raise ValueError("unexpected encoded feature shape")
        return np.asarray(features @ self.centroids.T, dtype=np.float64)

    def calibrate(self, rows):
        """Maximize known acceptance + OOS rejection with equal group weight.

        Ties prefer greater known acceptance. Temperature separately minimizes
        known-only intent NLL on a prespecified grid; it never defines the gate.
        """
        rows = list(rows)
        labels = np.asarray([r["intent"] for r in rows], dtype=str)
        in_scope = np.isin(labels, self.known)
        if not in_scope.any() or in_scope.all():
            raise ValueError("calibration requires both known and held-out unknown requests")
        cosines = self._cosines(self.encode([r["text"] for r in rows]))
        maxima = cosines.max(axis=1)
        thresholds = np.r_[np.unique(maxima), np.nextafter(maxima.max(), np.inf)]
        counts_known = np.searchsorted(np.sort(maxima[in_scope]), thresholds, side="left")
        counts_ood = np.searchsorted(np.sort(maxima[~in_scope]), thresholds, side="left")
        n_known, n_ood = int(in_scope.sum()), int((~in_scope).sum())
        # Exact integer ranking also makes ties independent of rounding.
        utility = (n_known - counts_known) * n_ood + counts_ood * n_known
        self.threshold = float(thresholds[utility.argmax()])
        targets = np.asarray([self.known.index(label) for label in labels[in_scope]])
        grid = (0.02, 0.05, 0.1, 0.2, 0.5, 1.0)
        def nll(temperature):
            probability = softmax(cosines[in_scope] / temperature, axis=1)
            return -np.log(probability[np.arange(n_known), targets].clip(1e-15, 1)).mean()
        self.temperature = min(grid, key=nll)
        return self

    def predict_encoded(self, features):
        if self.threshold is None:
            raise RuntimeError("calibrate on development requests before predicting")
        scores = self._cosines(features)
        return softmax(scores / self.temperature, axis=1), scores.max(axis=1) < self.threshold

    def predict(self, texts):
        return self.predict_encoded(self.encode(texts))


def _weights(values, width, *, require_rows):
    weights = np.asarray(values, dtype=np.float64)
    if (weights.ndim != 2 or weights.shape[1] != width or
            (require_rows and not len(weights)) or not np.isfinite(weights).all() or
            (weights < 0).any() or not np.allclose(weights.sum(axis=1), 1, rtol=1e-5, atol=1e-5)):
        raise ValueError("expected normalized nonnegative intent weights shaped (requests, K)")
    return weights


def _outcomes(matrix, n):
    y = np.asarray(matrix, dtype=np.float64)
    if y.ndim != 2 or y.shape[0] != n or not n or not y.shape[1] or not np.isin(y, [0, 1]).all():
        raise ValueError("expected a binary success matrix shaped (requests, experts)")
    return y


class Router:
    """Predict an executor index from the full known-intent weight vector."""

    def __init__(self, method, seed=1118):
        if method not in ("fixed", "lookup", "matrix", "mlp"):
            raise ValueError("method must be fixed, lookup, matrix or mlp")
        self.method, self.seed = method, seed

    def fit(self, weights, intents, known, success_matrix):
        self.known = _intent_order(known)
        w = _weights(weights, len(self.known), require_rows=True)
        labels = np.asarray(list(intents))
        if labels.shape != (len(w),) or not np.isin(labels, self.known).all():
            raise ValueError("router supervision must provide one known intent per request")
        y = _outcomes(success_matrix, len(w))
        self.global_success = y.mean(axis=0)
        # Two virtual observations per intent shrink sparse or missing classes.
        groups = [y[labels == label] for label in self.known]
        self.ability = np.stack([(g.sum(axis=0) + 2 * self.global_success) / (len(g) + 2)
                                 for g in groups])
        if self.method == "mlp":
            self._fit_mlp(w, y)
        return self

    def _fit_mlp(self, x, y):
        from sklearn.neural_network import MLPClassifier

        # sklearn treats a single label column differently; fit independent
        # multi-label outputs together, with a second constant column if M=1.
        probabilities = (y.sum(axis=0) + 1) / (len(y) + 2)
        self.constant = np.all(y == y[0], axis=0)
        self.constant_values = probabilities[self.constant]
        self.classifier = None
        if self.constant.all():
            return
        targets = y if y.shape[1] > 1 else np.column_stack([y, np.zeros(len(y))])
        self.classifier = MLPClassifier(hidden_layer_sizes=(24,), activation="relu",
                                        solver="adam", learning_rate_init=0.008, alpha=0.01,
                                        max_iter=250, tol=1e-4, random_state=self.seed)
        self.classifier.fit(x, targets.astype(int))

    def scores(self, weights):
        if not hasattr(self, "ability"):
            raise RuntimeError("fit the router before selecting an expert")
        w = _weights(weights, len(self.known), require_rows=False)
        shape = (len(w), len(self.global_success))
        if not len(w):
            return np.empty(shape)
        if self.method == "fixed":
            return np.broadcast_to(self.global_success, shape)
        if self.method == "lookup":
            return self.ability[w.argmax(axis=1)]
        if self.method == "matrix":
            return w @ self.ability
        if self.classifier is None:
            return np.broadcast_to(self.constant_values, shape)
        values = self.classifier.predict_proba(w)[:, :shape[1]]
        values[:, self.constant] = self.constant_values
        return values

    def choose(self, weights):
        return self.scores(weights).argmax(axis=1)


class EmbeddingRouter:
    """Text-feature control: independent logistic success scores per expert."""

    def __init__(self, seed=1118):
        self.seed = seed

    def fit(self, features, success_matrix):
        y = _outcomes(success_matrix, features.shape[0])
        self.models = []
        self.width = features.shape[1]
        for targets in y.T:
            if np.all(targets == targets[0]):
                self.models.append(float((targets.sum() + 1) / (len(targets) + 2)))
            else:
                model = LogisticRegression(C=5.0, max_iter=250, random_state=self.seed)
                self.models.append(model.fit(features, targets))
        return self

    def choose(self, features):
        if not hasattr(self, "models"):
            raise RuntimeError("fit the embedding router before scoring")
        if features.ndim != 2 or features.shape[1] != self.width:
            raise ValueError("unexpected text feature width")
        if not features.shape[0]:
            return np.empty(0, dtype=int)
        columns = [np.full(features.shape[0], model) if isinstance(model, float)
                   else model.predict_proba(features)[:, 1] for model in self.models]
        return np.column_stack(columns).argmax(axis=1)

