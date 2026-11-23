"""Predict service variants and slot presence from the public dialogue prefix."""
import argparse
from collections import Counter
import io
import json
from pathlib import Path

import numpy as np
from scipy.special import expit
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import LogisticRegression

from .routing import _texts, recent_five
from .window_frozen import digest
from .window_splits import build_folds


def hashed_prefix(texts):
    vectorizer = HashingVectorizer(n_features=4096, ngram_range=(1, 2), alternate_sign=False)
    return vectorizer.transform([recent_five(t) for t in _texts(texts)])


class SchemaCues:
    def __init__(self, services, slots, coef, intercept):
        self.services, self.slots = list(services), list(slots)
        self.coef, self.intercept = coef, intercept

    def transform(self, texts):
        probabilities = expit(hashed_prefix(texts) @ self.coef.T + self.intercept)
        service_scores = probabilities[:, :len(self.services)]
        service_scores /= service_scores.sum(axis=1, keepdims=True)
        return probabilities


def load_schema_cues(path):
    with np.load(path, allow_pickle=False) as data:
        return SchemaCues(data["services"].tolist(), data["slots"].tolist(), data["coef"], data["intercept"])


def fit(rows):
    services = sorted({r["gold"]["service"] for r in rows})
    counts = Counter(s for r in rows for s in r["gold"]["arguments"])
    slots = sorted(s for s, count in counts.items() if 10 <= count <= len(rows) - 10)
    labels = np.array([[r["gold"]["service"] == service for service in services]
                       + [s in r["gold"]["arguments"] for s in slots] for r in rows], dtype=int)
    features = hashed_prefix([r["text"] for r in rows])
    coefs, intercepts = [], []
    for y in labels.T:
        estimator = LogisticRegression(solver="liblinear", C=4, random_state=1118, max_iter=200)
        estimator.fit(features, y)
        coefs.append(estimator.coef_[0])
        intercepts.append(estimator.intercept_[0])
    return SchemaCues(services, slots, np.stack(coefs), np.array(intercepts))


def heldout_audit(model, rows, fit_rows):
    scores = model.transform([r["text"] for r in rows])
    n_services = len(model.services)
    services = [r["gold"]["service"] for r in rows]
    major = Counter(r["gold"]["service"] for r in fit_rows).most_common(1)[0][0]
    slot_gold = np.asarray([[s in r["gold"]["arguments"] for s in model.slots] for r in rows])
    predicted = scores[:, n_services:] >= 0.5
    tp = int(np.count_nonzero(slot_gold & predicted))
    return {"n": len(rows), "training_majority_service_correct": services.count(major),
            "predicted_service_correct": int((np.asarray(model.services)[scores[:, :n_services].argmax(1)] == services).sum()),
            "supported_slot_micro_f1_at_half": 2 * tp / (slot_gold.sum() + predicted.sum())}


def prepare(out):
    root, run = Path("data/sgd"), Path("artifacts/verified-minilm")
    study = Path("configs/support_validation_folds.json")
    lock = json.loads(Path("reports/support_validation/development_lock.json").read_bytes())
    if digest(root / "manifest.json") != lock["original_corpus_sha256"] or digest(study) != lock["holdout_config_sha256"]:
        raise ValueError("Pilot data or original fold definition changed")
    fold = build_folds(root, Path("configs/intent_catalog.json"), run, study)["original_domains"]
    frozen = lock["folds"]["original_domains"]
    rows = fold["fit"]
    if fold["known"] != frozen["known"] or len(rows) != frozen["counts"]["fit"]:
        raise ValueError("Unexpected known fit partition")
    model = fit(rows)
    binary = io.BytesIO()
    np.savez_compressed(binary, services=model.services, slots=model.slots, coef=model.coef, intercept=model.intercept)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "schema_cues.npz"
    if path.exists() and path.read_bytes() != binary.getvalue():
        raise FileExistsError("Choose another directory for a different fitted feature model")
    path.write_bytes(binary.getvalue())
    audit = {"data": "pilot/original_domains/fit", "rows": len(rows), "dialogues": len({r["dialogue_id"] for r in rows}),
             "service_count": len(model.services), "slot_count": len(model.slots), "model_sha256": digest(path),
             "split_sha256": digest(run / "split.json"), "known": fold["known"],
             "independent_pilot_router_rows": heldout_audit(model, fold["router"], rows)}
    (out / "schema_fit.json").write_bytes((json.dumps(audit, indent=2) + "\n").encode())
    print(json.dumps(audit, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("reports/within_intent"))
    prepare(parser.parse_args().out)
