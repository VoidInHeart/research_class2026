"""Compare frozen OOS score records at calibrated and retrospectively matched operating points."""
import numpy as np

from .near_gate_eval import apply_threshold, compare_gates, evaluate, threshold


METHODS = ("centroid", "semantic", "lexical", "adaptive")


def evaluate_records(rows, thresholds, legacy_raw_cut, related_known=None, rate=29 / 377):
    """Evaluate threshold migration separately from equal-budget ranking.

    thresholds contains four scalar cutoffs learned on calibration knowns, and
    legacy_raw_cut is the novelty cutoff of the original calibrated centroid gate.
    Higher scores reject, and the deployment rule strictly exceeds its cutoff.
    Matched-score ties use the same public request-ID rule as the preceding study.
    A known-only shard still estimates false rejections and execution quality;
    unknown recall and its paired interval are undefined and remain null.
    """
    rows = list(rows)
    if not rows or not 0 <= rate <= 1 or set(thresholds) != set(METHODS):
        raise ValueError("Expected a nonempty cohort, valid rejection rate and one cutoff per method")
    ids = [r["id"] for r in rows]
    dialogues = [r["dialogue_id"] for r in rows]
    unknown = np.asarray([r["unknown"] for r in rows])
    if not np.isin(unknown, (False, True)).all():
        raise ValueError("Unknown labels must be binary")
    unknown = unknown.astype(bool)
    nk, nu = int((~unknown).sum()), int(unknown.sum())
    if not nk:
        raise ValueError("No known requests to define the matched operating point")
    k = int(np.floor(nk * rate + .5))
    outcomes = [[r[key] for r in rows] for key in ("expert_exact", "expert_method", "base_exact", "base_method")]
    related = np.array([related_known is not None and r["intent"] == related_known for r in rows], dtype=bool)
    if (related & unknown).any():
        raise ValueError("The related known label occurs among unknown requests")

    def metrics(mask, boundary):
        return {**evaluate(mask, unknown, *outcomes), "boundary": boundary,
                "related_known_n": int(related.sum()), "related_known_rejected": int((mask & related).sum())}

    old_boundary = {"score": float(legacy_raw_cut), "ties": "none"}
    old = apply_threshold([r["raw_centroid"] for r in rows], ids, old_boundary)
    report = {"n": len(rows), "nk": nk, "nu": nu, "dialogues": len(set(dialogues)), "known_budget": k,
              "target_known_frr": float(rate), "related_known": related_known,
              "legacy_calibrated": metrics(old, old_boundary), "from_cal": {}, "matched": {}}
    masks = {"from_cal": {}, "matched": {}}
    for method in METHODS:
        score = np.asarray([r["scores"][method] for r in rows], dtype=np.float64)
        boundaries = {"from_cal": {"score": float(thresholds[method]), "ties": "none"},
                      "matched": threshold(score, ids, ~unknown, k)}
        for name, boundary in boundaries.items():
            mask = apply_threshold(score, ids, boundary)
            masks[name][method] = mask
            report[name][method] = metrics(mask, boundary)
        if report["matched"][method]["known_rejected"] != k:
            raise AssertionError("The matched known-rejection budget was not met")
    for name, family in masks.items():
        report[name + "_paired"] = {method: compare_gates(family["centroid"], mask, unknown, dialogues, *outcomes)
                                     for method, mask in family.items()}
    return report
