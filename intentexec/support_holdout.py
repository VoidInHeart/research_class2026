"""Select disjoint, previously unused SGD training shards for one locked evaluation."""
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path

from .data import _json, load_sgd
from .window_splits import _components, _dialogue_labels


def _select(fresh, prior, active, previous_dialogues, fold):
    known, cal, unknown = (set(fold[name]) for name in ("known", "dev_unknown", "test_unknown"))
    if not known or not unknown or known & unknown or cal & (known | unknown):
        raise ValueError("Use distinct, nonempty known and test-unknown inventories")
    if set(active) & set(previous_dialogues):
        raise ValueError("The new shards reuse pilot dialogue identifiers")
    prior_ids = {r["id"] for r in prior}
    fresh_ids = {r["id"] for r in fresh}
    if len(fresh_ids) != len(fresh) or prior_ids & fresh_ids:
        raise ValueError("Duplicate or reused external request ID")

    selected, excluded = [], Counter()
    unknown_exclusions, excluded_combinations = Counter(), Counter()
    scope = known | unknown
    def role(label):
        return "known" if label in known else "test_unknown"

    for component in _components([*prior, *fresh]).values():
        candidates = [r for r in component if r["id"] in fresh_ids]
        if not candidates:
            continue
        component_intents = set().union(*(active[d] for d in {r["dialogue_id"] for r in candidates}))
        reason = None
        if len(candidates) != len(component):
            # A direct or transitive shared template (full or recent-five window)
            # with any pilot partition quarantines the complete fresh component.
            reason = "connected_to_pilot_template"
        elif component_intents & cal:
            reason = "contains_calibration_unknown"
        elif component_intents - scope:
            reason = "contains_out_of_catalog_intent"
        if reason:
            excluded[reason] += len(candidates)
            removed = sum(r["intent"] in unknown for r in candidates)
            if removed:
                unknown_exclusions[reason] += removed
                if reason == "contains_out_of_catalog_intent":
                    excluded_combinations[tuple(sorted(component_intents - scope))] += removed
            continue
        for r in candidates:
            selected.append(dict(r, role=role(r["intent"]),
                                 dialogue_roles=sorted({role(label) for label in active[r["dialogue_id"]]})))
    selected.sort(key=lambda r: r["id"])
    if not selected:
        raise ValueError("No external rows satisfied the scope and isolation rules")
    audit = {"excluded_rows": dict(sorted(excluded.items())),
             "excluded_test_unknown_by_reason": dict(sorted(unknown_exclusions.items())),
             "out_of_catalog_combinations_for_excluded_test_unknown": [
                 {"outside_intents": list(labels), "requests": count}
                 for labels, count in sorted(excluded_combinations.items())]}
    return selected, audit


def fresh_rows(data_root, extra_root, catalog_path, fold, prior_rows, seed=202611):
    """Return all eligible external rows. No subsampling, tuning or model scores."""
    prior_rows = list(prior_rows)
    data_root, extra_root = Path(data_root), Path(extra_root)
    old_manifest = _json(data_root / "manifest.json")
    new_manifest = _json(extra_root / "manifest.json")
    lock = _json(Path(__file__).resolve().parents[1] / "configs" / "support_holdout_sha256.json")
    if new_manifest["sha256"] != lock:
        raise ValueError("External shard digests do not match the preselected validation release")
    if any(old_manifest[k] != new_manifest[k] for k in ("revision", "split", "repository")):
        raise ValueError("Evaluation and pilot data must come from the same pinned SGD release")
    old_files, new_files = set(old_manifest["sha256"]), set(new_manifest["sha256"])
    if old_files & new_files != {"schema.json"}:
        raise ValueError("Evaluation and pilot shards must be disjoint with one shared schema")
    if old_manifest["sha256"]["schema.json"] != new_manifest["sha256"]["schema.json"]:
        raise ValueError("Fresh data must use the exact same public service catalog")

    # Authenticate both source directories against their manifests and verify that
    # the comparison set covers every previously available scoring request.
    original = load_sgd(data_root, catalog_path)
    if (len(prior_rows) != len(original)
            or {r["id"]: r["text"] for r in prior_rows} != {r["id"]: r["text"] for r in original}):
        raise ValueError("Expected the entire pilot corpus for template quarantine")
    fresh = load_sgd(extra_root, catalog_path)
    catalog = _json(catalog_path)
    active = _dialogue_labels(extra_root, catalog)
    previous_dialogues = _dialogue_labels(data_root, catalog)
    rows, audit = _select(fresh, original, active, previous_dialogues, fold)
    counts = Counter(r["role"] for r in rows)
    role_counts = {role: counts[role] for role in ("known", "test_unknown")}
    cohort_bytes = json.dumps([r["id"] for r in rows], separators=(",", ":")).encode()
    metadata = {"source_revision": new_manifest["revision"], "repository": new_manifest["repository"],
                "split": new_manifest["split"], "shards": sorted(new_files - {"schema.json"}),
                "source_sha256": new_manifest["sha256"], "seed": seed, "selection": "all_eligible",
                "pre_filter_requests": len(fresh), "pre_filter_dialogues": len(active),
                "estimable": all(role_counts.values()),
                "pre_filter_test_unknown": sum(r["intent"] in fold["test_unknown"] for r in fresh),
                **audit, "dialogues": len({r["dialogue_id"] for r in rows}),
                "counts_by_role": role_counts, "counts_by_intent": dict(sorted(Counter(r["intent"] for r in rows).items())),
                "counts_by_service": dict(sorted(Counter(r["gold"]["service"] for r in rows).items())),
                "cohort_sha256": sha256(cohort_bytes).hexdigest()}
    return rows, metadata
