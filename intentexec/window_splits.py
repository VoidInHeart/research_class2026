"""Construct window-ablation folds around a retrieval pool with an immutable fit partition."""
from collections import Counter, defaultdict
from pathlib import Path

from .data import _index, _json, _template, load_sgd


PHASES = ("fit", "router", "cal", "test")


def _dialogue_labels(data_root, catalog):
    # A held-out intent mentioned later need not have a scorable service call.
    method_index, _ = _index(catalog)
    labels = {}
    for shard in sorted((Path(data_root) / "train").glob("dialogues_*.json")):
        for dialogue in _json(shard):
            labels[dialogue["dialogue_id"]] = {
                method_index[f["state"]["active_intent"]] for turn in dialogue["turns"]
                if turn["speaker"] == "USER" for f in turn["frames"]
                if f["state"]["active_intent"] != "NONE"}
    return labels


def _components(rows):
    """Union across dialogue, full history and last-five-turn normalized text."""
    parent = {r["dialogue_id"]: r["dialogue_id"] for r in rows}

    def find(did):
        while parent[did] != did:
            parent[did] = parent[parent[did]]
            did = parent[did]
        return did

    templates = {}
    for row in rows:
        did, text = row["dialogue_id"], row["text"]
        for view in (text, "\n".join(text.splitlines()[-5:])):
            other = templates.setdefault(_template(view), did)
            a, b = find(did), find(other)
            parent[max(a, b)] = min(a, b)
    grouped = defaultdict(list)
    for row in rows:
        grouped[find(row["dialogue_id"])].append(row)
    return grouped


def _build_fold(groups, active, source_phase, frozen_known, fit_counts, spec):
    dev, test = set(spec["dev_unknown"]), set(spec["test_unknown"])
    if not dev or not test or dev & test:
        raise ValueError("Non-empty, disjoint calibration and evaluation holdouts are required")
    intended_known = set(frozen_known) - dev - test
    known = intended_known
    # Some intents only occur in mixed dialogues with a held-out intent.
    # Use fit membership alone to find the supported known inventory. An intent
    # removed here can quarantine more dialogues, so recompute until stable.
    while True:
        supported = {row["intent"] for rows in groups.values()
                     if all(source_phase[r["id"]] == "fit" and active[r["dialogue_id"]] <= known
                            for r in rows) for row in rows}
        if supported == known:
            break
        known = supported
    scope = known | dev | test
    partitions = {phase: [] for phase in PHASES}
    excluded = Counter()

    def role(label):
        if label in dev:
            return "dev_unknown"
        return "test_unknown" if label in test else "known"

    for rows in groups.values():
        phases = {source_phase[r["id"]] for r in rows}
        labels = set().union(*(active[d] for d in {r["dialogue_id"] for r in rows}))
        has_dev, has_test = bool(labels & dev), bool(labels & test)
        # A new five-turn duplicate can straddle original partitions. Quarantine
        # the entire component, keeping the frozen expert's fit examples private.
        if len(phases) != 1:
            reason = "crosses_frozen_partitions"
        elif labels - scope:
            reason = "out_of_fold_catalog"
        elif has_dev and has_test:
            reason = "both_holdouts"
        elif (has_dev or has_test) and phases & {"fit", "router"}:
            reason = "heldout_in_source_fit_or_router"
        else:
            phase = "cal" if has_dev else "test" if has_test else next(iter(phases))
            for row in rows:
                partitions[phase].append(dict(row, role=role(row["intent"]),
                    dialogue_roles=sorted({role(label) for label in active[row["dialogue_id"]]})))
            continue
        excluded[reason] += len(rows)
    for rows in partitions.values():
        rows.sort(key=lambda row: row["id"])
    intent_counts = {p: dict(sorted(Counter(r["intent"] for r in rows).items())) for p, rows in partitions.items()}
    role_counts = {p: dict(sorted(Counter(r["role"] for r in rows).items())) for p, rows in partitions.items()}
    if set(intent_counts["fit"]) != known:
        raise ValueError(f"Declared known intents missing fit rows: {sorted(known - set(intent_counts['fit']))}")
    if any(not partitions[p] for p in PHASES) or any(role_counts[p].get(role, 0) == 0
            for p, role in (("cal", "known"), ("cal", "dev_unknown"), ("test", "known"), ("test", "test_unknown"))):
        raise ValueError("The development and evaluation sets must each contain known and unknown requests")
    actual_dev = sorted(dev & intent_counts["cal"].keys())
    actual_test = sorted(test & intent_counts["test"].keys())
    metadata = {"source_expert_fit_counts": dict(sorted(fit_counts.items())),
                "known_without_clean_fit_support": sorted(intended_known - known),
                "heldout_expert_fit_counts": {k: fit_counts.get(k, 0) for k in sorted(dev | test)},
                "requested_dev_unknown": sorted(dev), "requested_test_unknown": sorted(test),
                "counts_by_role": role_counts, "counts_by_intent": intent_counts,
                "dialogue_counts": {p: len({r["dialogue_id"] for r in rows}) for p, rows in partitions.items()},
                "excluded_rows": dict(sorted(excluded.items()))}
    return {**partitions, "known": sorted(known), "dev_unknown": actual_dev, "test_unknown": actual_test,
            "metadata": metadata}


def build_folds(data_root, catalog_path, frozen_run_dir, config_path):
    """Rotate labels for recognition without moving expert-training cases into evaluation.

    Fit and router use subsets of their original partitions. Original cal and test
    supply a common evaluation reservoir. A held-out dialogue goes to cal or test
    according to its labels; known-only dialogues keep their earlier assignment.
    """
    source = _json(Path(frozen_run_dir) / "split.json")
    catalog = _json(catalog_path)
    active = _dialogue_labels(data_root, catalog)
    indexed = {row["id"]: row for row in load_sgd(data_root, catalog_path)}
    source_phase = {}
    for phase in PHASES:
        for uid in source[phase]:
            if uid not in indexed or uid in source_phase:
                raise ValueError(f"Missing or repeated frozen source ID {uid}")
            source_phase[uid] = phase
    fit_counts = Counter(indexed[uid]["intent"] for uid in source["fit"])
    if set(fit_counts) != set(source["known"]):
        raise ValueError("Frozen known inventory and fit labels differ")
    groups = _components([indexed[uid] for uid in source_phase])
    folds = {}
    for name, spec in _json(config_path).items():
        requested = set(spec["dev_unknown"]) | set(spec["test_unknown"])
        if requested - catalog["classes"].keys():
            raise ValueError(f"Unknown holdout labels in {name}: {requested - catalog['classes'].keys()}")
        folds[name] = _build_fold(groups, active, source_phase, set(source["known"]), fit_counts, spec)
    return folds
