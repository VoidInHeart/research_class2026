"""Select a support score by rotating known training intents as development-only unknowns."""
from collections import defaultdict
from hashlib import sha256

import numpy as np

from .near_gate_eval import apply_threshold, threshold
from .support_gate import SupportGate
from .window_splits import _components


# The first candidate wins ties. Candidate order and screening rules are fixed before evaluation.
CANDIDATES = {
    "centroid": (1., 0., 0.), "semantic": (0., 1., 0.), "lexical": (0., 0., 1.),
    "centroid_semantic": (.5, .5, 0.), "semantic_mix": (.25, .5, .25),
    "lexical_mix": (.25, .25, .5),
}
MIN_HIDDEN_DIALOGUES = 3
MIN_ROTATIONS = 3
MIN_KNOWN_EVALUATION = 13


def _group_metadata(fit, router, dialogue_intents, seed):
    source = {}
    active = defaultdict(set)
    for phase, rows in (("fit", fit), ("router", router)):
        for row in rows:
            did, uid = row["dialogue_id"], row["id"]
            if uid in source:
                raise ValueError("Development request IDs must be unique across fit and router")
            source[uid] = phase
            active[did].add(row["intent"])
    if dialogue_intents is not None:
        for did, labels in active.items():
            if did not in dialogue_intents or not labels <= set(dialogue_intents[did]):
                raise ValueError("Incomplete dialogue intent inventory")
            labels.update(dialogue_intents[did])
    groups = _components([*fit, *router])
    group_for, group_labels, half = {}, {}, {}
    for gid, rows in groups.items():
        # A shortened template can link two otherwise disjoint source phases.
        if len({source[r["id"]] for r in rows}) > 1:
            continue
        group_labels[gid] = set().union(*(active[r["dialogue_id"]] for r in rows))
        half[gid] = sha256(f"{seed}/{gid}".encode()).digest()[0] % 2
        for row in rows:
            group_for[row["id"]] = gid
    return group_for, group_labels, half


def select_weights(fit_rows, fit_vectors, known, router_rows, router_vectors, *,
                   dialogue_intents=None, known_frr=29 / 377, seed=202611):
    """Return JSON-ready weights picked without receiving calibration/test requests.

    Each rotation hides a complete intent and all supporting fit dialogue/template
    groups. Known-only router references and evaluation requests come from opposite
    halves of the independent router data; then halves swap. Gold in evaluation
    is used only to enforce the matched known-error budget and count pseudo-OOS.
    Pass the SGD active-intent inventory to also exclude unscored mentions.
    """
    fit = list(fit_rows)
    router = list(router_rows)
    known = list(known)
    fit_v, router_v = np.asarray(fit_vectors), np.asarray(router_vectors)
    if (not known or len(set(known)) != len(known) or not 0 < known_frr < 1
            or fit_v.ndim != 2 or router_v.ndim != 2 or fit_v.shape[1] != router_v.shape[1]
            or len(fit) != len(fit_v) or len(router) != len(router_v)):
        raise ValueError("Invalid development matrices, known inventory or error budget")
    if any(r["intent"] not in known or r.get("role", "known") != "known" for r in [*fit, *router]):
        raise ValueError("Support selection accepts only known fit and router examples")
    group_for, active, halves = _group_metadata(fit, router, dialogue_intents, seed)
    fit_groups = [group_for.get(r["id"]) for r in fit]
    router_groups = [group_for.get(r["id"]) for r in router]
    rotations, skipped = {}, {}
    for hidden in sorted(known):
        training = [i for i, group in enumerate(fit_groups) if group is not None and hidden not in active[group]]
        remaining = {fit[i]["intent"] for i in training}
        supported = [label for label in known if label in remaining]
        if len(supported) < 2:
            skipped[hidden] = "fewer_than_two_supported_known_labels"
            continue
        # Rows of unsupported active intents are excluded along with their dialogue group.
        usable = [i for i, g in enumerate(router_groups) if g is not None and active[g] <= remaining | {hidden}]
        hidden_dialogues = {router[i]["dialogue_id"] for i in usable if router[i]["intent"] == hidden}
        if len(hidden_dialogues) < MIN_HIDDEN_DIALOGUES:
            skipped[hidden] = "fewer_than_three_hidden_dialogues"
            continue
        hidden_hits = dict.fromkeys(CANDIDATES, 0)
        splits = []
        for eval_half in (0, 1):
            ref = [i for i in usable if halves[router_groups[i]] != eval_half and hidden not in active[router_groups[i]]]
            ev = [i for i in usable if halves[router_groups[i]] == eval_half]
            is_known = np.array([router[i]["intent"] != hidden for i in ev], dtype=bool)
            n_known, n_hidden = int(is_known.sum()), int((~is_known).sum())
            if not n_hidden:
                continue
            if not ref or n_known < MIN_KNOWN_EVALUATION:
                splits.clear()
                break
            gate = SupportGate([fit[i] for i in training], fit_v[training], supported,
                               [router[i] for i in ref], router_v[ref])
            parts = gate.components([router[i]["text"] for i in ev], router_v[ev])
            z = np.column_stack([parts[k] for k in ("z_centroid", "z_semantic", "z_lexical")])
            ids = [router[i]["id"] for i in ev]
            k = int(np.floor(known_frr * n_known + .5))
            hits = {}
            for name, w in CANDIDATES.items():
                scores = z @ w
                cut = threshold(scores, ids, is_known, k)
                decision = apply_threshold(scores, ids, cut)
                hits[name] = int((decision & ~is_known).sum())
                hidden_hits[name] += hits[name]
            splits.append({"evaluation_half": eval_half, "reference": len(ref), "known": n_known,
                           "known_rejected": k, "hidden": n_hidden, "hidden_rejected": hits})
        if not splits:
            skipped[hidden] = "insufficient_grouped_reference_or_known_evaluation"
            continue
        total = sum(s["hidden"] for s in splits)
        rotations[hidden] = {"fit": len(training), "known": supported, "hidden_dialogues": len(hidden_dialogues),
                             "folds": splits, "recall": {k: v / total for k, v in hidden_hits.items()}}
    macro = {m: float(np.mean([s["recall"][m] for s in rotations.values()])) if rotations else None
             for m in CANDIDATES}
    enough = len(rotations) >= MIN_ROTATIONS
    choice = max(CANDIDATES, key=macro.get) if enough else "centroid"
    return {"candidate": choice, "weights": list(CANDIDATES[choice]),
            "reason": "highest_macro_pseudo_unknown_recall" if enough else "insufficient_supported_rotations",
            "macro_recall": macro, "candidate_weights": {m: list(w) for m, w in CANDIDATES.items()},
            "known_frr": known_frr, "seed": seed, "active_intents_from_unscored_turns": dialogue_intents is not None,
            "excluded_cross_phase_rows": len(fit) + len(router) - len(group_for),
            "rotations": rotations, "skipped": skipped}
