"""Fit expected exact-call utilities of the current pool on known route supervision."""
import numpy as np

from .executors import BASE, EXPERTS
from .routing import Router, _weights


EXECUTORS = (*EXPERTS, BASE)
BASE_INDEX = EXECUTORS.index(BASE)
SELECTION_ORDER = ("always_base", "conservative", "matrix4", "matrix3")


def fit_tables(weights, intents, known, outcomes):
    """Columns are the three experts followed by base, with one paired result per request."""
    known = list(known)
    y = np.asarray(outcomes, dtype=np.float64)
    if y.ndim != 2 or y.shape[1] != len(EXECUTORS):
        raise ValueError("Expected three expert outcomes followed by the paired base outcome")
    labels = np.asarray(list(intents))
    model = Router("matrix").fit(weights, labels, known, y)
    advantage = []
    for label in known:
        group = y[labels == label]
        delta = group[:, :BASE_INDEX] - group[:, BASE_INDEX, None]
        # Heuristic penalty for scarce discordant pairs, in success-probability units.
        # With no observed evidence in an intent, every expert gets a negative margin.
        wins_minus_losses = delta.sum(axis=0)
        discordant = np.abs(delta).sum(axis=0)
        advantage.append((wins_minus_losses - np.sqrt(discordant + 1)) / (len(group) + 2))
    fixed = int(model.global_success[:BASE_INDEX].argmax())
    fixed_all = fixed if model.global_success[fixed] > model.global_success[BASE_INDEX] else BASE_INDEX
    return {"known": known, "fixed_expert": fixed, "fixed_all": fixed_all,
            "ability": model.ability.tolist(), "advantage": np.asarray(advantage).tolist()}


def choose_all(fold_lock, weights):
    """Infer using intent weights only. The caller implements the independent OOS gate."""
    w = _weights(weights, len(fold_lock["known"]), require_rows=False)
    utility = w @ np.asarray(fold_lock["ability"])
    three = utility[:, :BASE_INDEX].argmax(axis=1)
    rows = np.arange(len(w))
    four = np.where(utility[rows, three] > utility[:, BASE_INDEX], three, BASE_INDEX)
    margin = w @ np.asarray(fold_lock["advantage"])
    specialist = margin.argmax(axis=1)
    conservative = np.where(margin[rows, specialist] > 0, specialist, BASE_INDEX)
    choices = {"fixed_expert": np.full(len(w), fold_lock["fixed_expert"], dtype=int),
               "fixed_all": np.full(len(w), fold_lock["fixed_all"], dtype=int),
               "always_base": np.full(len(w), BASE_INDEX, dtype=int),
               "matrix3": three, "matrix4": four, "conservative": conservative}
    if "selected" in fold_lock:
        choices["selected"] = choices[fold_lock["selected"]]
    return choices
