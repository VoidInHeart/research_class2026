"""Evaluate one unknown-intent holdout at a time on paired requests."""
from collections import Counter, defaultdict

import numpy as np
from sklearn.metrics import roc_auc_score


def _ratio(num, den):
    return float(num / den) if den else None


def summarize(rows, known):
    """Return recognition, gate and routed execution rates on [0,1].

    known_top1 uses every known request, including rejected ones. known_accepted_top1
    conditions on passing the gate. Macro recall averages the known intents represented
    in this test fold. gate_oracle_exact keeps the observed gate fixed and allows an
    oracle to select an expert only on accepted requests.
    """
    rows = list(rows)
    if not rows:
        raise ValueError("A holdout must contain at least one request")
    known = set(known)
    intents = defaultdict(Counter)
    totals = Counter()
    unknown_flags, oos_scores = [], []
    for row in rows:
        intent = row["intent"]
        is_known = intent in known
        reject = bool(row["gate_oos"])
        recognized = is_known and row["top1"] == intent
        correct = bool(row["correct"])
        group = intents[intent]
        group["n"] += 1
        group["accepted"] += not reject
        group["top1_correct"] += recognized
        group["exact_correct"] += correct
        totals["known"] += is_known
        totals["known_top1_correct"] += recognized
        totals["known_accepted_top1_correct"] += recognized and not reject
        totals["known_rejected"] += is_known and reject
        totals["unknown_rejected"] += not is_known and reject
        totals["exact_correct"] += correct
        totals["method_correct"] += bool(row["method_correct"])
        totals["known_exact_correct"] += correct and is_known
        totals["gate_oracle_correct"] += bool(row["base_correct"] if reject else row["expert_best_correct"])
        unknown_flags.append(not is_known)
        oos_scores.append(row["oos_score"])

    n = len(rows)
    nk = totals["known"]
    nu = n - nk
    accepted = nk - totals["known_rejected"]
    false_reject = _ratio(totals["known_rejected"], nk)
    recall = _ratio(totals["unknown_rejected"], nu)
    recalls = [group["top1_correct"] / group["n"] for key, group in intents.items() if key in known]
    return {
        "n": n, "dialogues": len({r["dialogue_id"] for r in rows}), "n_known": nk, "n_unknown": nu,
        "known_top1": _ratio(totals["known_top1_correct"], nk),
        "known_macro_recall": float(np.mean(recalls)) if recalls else None,
        "known_accepted_top1": _ratio(totals["known_accepted_top1_correct"], accepted),
        "known_false_reject": false_reject, "unknown_recall": recall,
        "gate_balanced_accuracy": (1 - false_reject + recall) / 2 if nk and nu else None,
        "oos_auroc": float(roc_auc_score(unknown_flags, oos_scores)) if nk and nu else None,
        "exact": totals["exact_correct"] / n, "method_accuracy": totals["method_correct"] / n,
        "known_exact": _ratio(totals["known_exact_correct"], nk),
        "unknown_exact": _ratio(totals["exact_correct"] - totals["known_exact_correct"], nu),
        "gate_oracle_exact": totals["gate_oracle_correct"] / n,
        "counts": dict(totals),
        "by_intent": {label: {"known": label in known, **group} for label, group in sorted(intents.items())},
    }


def _paired_index(rows):
    indexed = {}
    for row in rows:
        if row["id"] in indexed:
            raise ValueError(f"Duplicate paired request ID {row['id']}")
        indexed[row["id"]] = row
    return indexed


def _differences(counts):
    # Columns: subgroup denominators and paired candidate-minus-baseline numerators.
    n, nk, nu, top, kgate, ugate, exact = counts.T
    with np.errstate(divide="ignore", invalid="ignore"):
        return {"known_top1": top / nk, "gate_balanced_accuracy": (kgate / nk + ugate / nu) / 2,
                "exact": exact / n}


def paired(rows_a, rows_b, known, seed=1118):
    """Compare candidate B against baseline A within one fold and policy.

    For known_top1, gate_balanced_accuracy and exact, return difference (= B - A),
    dialogue-cluster percentile ci95, paired rescues/harm and n. Gate interval and
    difference give the known and unknown classes equal weight; gate rescue/harm
    counts are raw request counts, with stratum counts in gate_by_truth.
    """
    a, b = _paired_index(rows_a), _paired_index(rows_b)
    if not a or a.keys() != b.keys():
        raise ValueError("Paired variants must contain the same nonempty set of request IDs")
    known = set(known)
    counts = defaultdict(lambda: np.zeros(7, dtype=np.int64))
    changes = defaultdict(Counter)
    for uid in sorted(a):
        left, right = a[uid], b[uid]
        if any(left[k] != right[k] for k in ("dialogue_id", "intent", "role")):
            raise ValueError(f"Dialogue or truth differs for paired request {uid}")
        is_known = left["intent"] in known
        g = counts[left["dialogue_id"]]
        g[0] += 1
        g[1 if is_known else 2] += 1
        should_reject = not is_known
        gate_delta = int(bool(right["gate_oos"]) == should_reject) - int(bool(left["gate_oos"]) == should_reject)
        call_delta = int(right["correct"]) - int(left["correct"])
        gate_group = "known" if is_known else "unknown"
        deltas = {"exact": call_delta, "gate_balanced_accuracy": gate_delta, gate_group: gate_delta}
        if is_known:
            top_delta = int(right["top1"] == left["intent"]) - int(left["top1"] == left["intent"])
            g[3] += top_delta
            deltas["known_top1"] = top_delta
        g[4 if is_known else 5] += gate_delta
        g[6] += call_delta
        for metric, delta in deltas.items():
            changes[metric]["n"] += 1
            changes[metric]["rescued"] += delta > 0
            changes[metric]["harmed"] += delta < 0

    groups = np.stack([counts[did] for did in sorted(counts)])
    point = _differences(groups.sum(axis=0, keepdims=True))
    draws = 2000
    rng = np.random.default_rng(seed)
    multiplicities = rng.multinomial(len(groups), np.full(len(groups), 1 / len(groups)), size=draws)
    bootstrap = _differences(multiplicities @ groups)
    result = {"direction": "candidate_minus_baseline", "dialogues": len(groups), "bootstrap_draws": draws}
    for metric, value in point.items():
        finite = bootstrap[metric][np.isfinite(bootstrap[metric])]
        result[metric] = {"difference": float(value[0]) if np.isfinite(value[0]) else None,
                          "ci95": np.quantile(finite, [.025, .975]).tolist() if len(finite) else None,
                          "valid_draws": len(finite),
                          **{k: changes[metric][k] for k in ("rescued", "harmed", "n")}}
    result["gate_by_truth"] = {truth: {k: changes[truth][k] for k in ("rescued", "harmed", "n")}
                               for truth in ("known", "unknown")}
    return result
