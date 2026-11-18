"""Matched known-rejection operating points and paired, frozen-gate evaluation."""
from hashlib import sha256

import numpy as np


def _scores_and_ids(scores, ids):
    values = np.asarray(scores, dtype=np.float64)
    ids = list(ids)
    if (values.ndim != 1 or len(values) != len(ids) or not np.isfinite(values).all()
            or not all(isinstance(uid, str) for uid in ids) or len(set(ids)) != len(ids)):
        raise ValueError("Expected finite 1-D scores and aligned unique string request IDs")
    return values, ids


def _binary(values, n):
    array = np.asarray(values)
    if array.shape != (n,) or not np.isin(array, (False, True)).all():
        raise ValueError("Expected an aligned 1-D binary vector")
    return array.astype(bool, copy=False)


def _priority(uid):
    # The secondary ID key also orders the hypothetical event of a hash collision.
    return sha256(uid.encode("utf-8")).hexdigest(), uid


def threshold(scores, ids, known_mask, k):
    """Choose a frozen cutoff rejecting exactly k examples among the supplied known rows.

    Larger scores reject. Equal scores only need a request-ID tie break if the
    requested budget divides a block of equal known scores. At k=0 reject only
    above the largest known score. At k=n reject down through the smallest one.
    """
    scores, ids = _scores_and_ids(scores, ids)
    known = np.flatnonzero(_binary(known_mask, len(ids)))
    n = len(known)
    if not n or isinstance(k, bool) or not isinstance(k, (int, np.integer)) or not 0 <= k <= n:
        raise ValueError("Budget must be an integer between 0 and the nonempty known count")
    ranked = sorted(known, key=lambda i: (-scores[i], _priority(ids[i])))
    i = ranked[max(0, k - 1)]
    value = float(scores[i])
    if k == 0:
        return {"score": value, "ties": "none"}
    if k == n or scores[ranked[k]] < value:
        return {"score": value, "ties": "all"}
    return {"score": value, "ties": "id_sha256", "last_id": ids[i]}


def apply_threshold(scores, ids, boundary):
    """Apply the saved cutoff using only the score and public request ID."""
    scores, ids = _scores_and_ids(scores, ids)
    value = float(boundary["score"])
    if not np.isfinite(value):
        raise ValueError("Cutoff must be finite")
    ties = boundary["ties"]
    if ties == "all":
        return scores >= value
    rejected = scores > value
    if ties == "none":
        return rejected
    if ties != "id_sha256":
        raise ValueError("Unrecognized boundary tie policy")
    last = _priority(boundary["last_id"])
    for i in np.flatnonzero(scores == value):
        rejected[i] = _priority(ids[i]) <= last
    return rejected


def _rate(num, den):
    return float(num / den) if den else None


def evaluate(mask, unknown_mask, expert_exact, expert_method, base_exact, base_method):
    """Evaluate one gate. True routes to the frozen base, false to the preselected expert."""
    n = len(mask)
    if not n:
        raise ValueError("Cannot evaluate an empty cohort")
    reject, unknown, ee, em, be, bm = (_binary(x, n) for x in
                                      (mask, unknown_mask, expert_exact, expert_method, base_exact, base_method))
    exact = np.where(reject, be, ee)
    method = np.where(reject, bm, em)
    nk, nu = int((~unknown).sum()), int(unknown.sum())
    kr, ur = int((reject & ~unknown).sum()), int((reject & unknown).sum())
    return {"n": n, "nk": nk, "nu": nu, "known_rejected": kr, "unknown_rejected": ur,
            "known_false_reject_rate": _rate(kr, nk), "unknown_recall": _rate(ur, nu),
            "accepted": n - kr - ur, "known_accepted": nk - kr, "unknown_accepted": nu - ur,
            "exact_correct": int(exact.sum()), "method_correct": int(method.sum()),
            "known_exact_correct": int((exact & ~unknown).sum()), "unknown_exact_correct": int((exact & unknown).sum()),
            "known_method_correct": int((method & ~unknown).sum()),
            "unknown_method_correct": int((method & unknown).sum()),
            "exact": float(exact.mean()), "method": float(method.mean())}


def compare_gates(old, new, unknown_mask, dialogues, expert_exact, expert_method, base_exact, base_method):
    """Frozen-threshold paired differences, candidate minus baseline, clustering by dialogue.

    A rejection gain means an extra rejected request, including for known_rejected.
    Execution gains and losses mean complete or method correctness was rescued or lost.
    CIs resample dialogues while keeping the cutoffs, executor selections and outputs fixed.
    They do not account for uncertainty in fitting gate scores or reselecting cutoffs.
    """
    n = len(old)
    if not n or len(dialogues) != n:
        raise ValueError("Expected a nonempty cohort with an aligned dialogue ID per request")
    old, new, unknown, ee, em, be, bm = (_binary(x, n) for x in
                                        (old, new, unknown_mask, expert_exact, expert_method, base_exact, base_method))
    groups, inverse = np.unique(np.asarray(dialogues), return_inverse=True)
    draws = 2000
    rng = np.random.default_rng(1118)
    multiplicity = rng.multinomial(len(groups), np.full(len(groups), 1 / len(groups)), size=draws)
    all_rows = np.ones(n, dtype=bool)
    outcomes = {
        "known_rejected": (~unknown, old, new),
        "unknown_rejected": (unknown, old, new),
        "exact": (all_rows, np.where(old, be, ee), np.where(new, be, ee)),
        "method": (all_rows, np.where(old, bm, em), np.where(new, bm, em)),
    }
    result = {"direction": "candidate_minus_baseline", "ci_condition": "frozen_scores_and_thresholds",
              "bootstrap_unit": "dialogue", "bootstrap_draws": draws, "dialogues": len(groups)}
    for key, (keep, before, after) in outcomes.items():
        delta = (after.astype(int) - before.astype(int)) * keep
        den = int(keep.sum())
        count = int(delta.sum())
        numerators = np.bincount(inverse, weights=delta, minlength=len(groups))
        denominators = np.bincount(inverse, weights=keep, minlength=len(groups))
        d = multiplicity @ denominators
        net = multiplicity @ numerators
        ratios = net[d > 0] / d[d > 0]
        gain, loss = int((delta == 1).sum()), int((delta == -1).sum())
        result[key] = {"n": den, "baseline_count": int((before & keep).sum()),
                       "candidate_count": int((after & keep).sum()), "delta_count": count,
                       "gain": gain, "loss": loss, "difference": _rate(count, den),
                       "ci95": np.quantile(ratios, [.025, .975]).tolist() if len(ratios) else None,
                       "valid_draws": len(ratios)}
    return result
