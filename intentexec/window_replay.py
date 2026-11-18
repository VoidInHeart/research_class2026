"""Verify published window-study predictions and reproduce paired metrics offline."""
import argparse
from collections import Counter, defaultdict
import gzip
from hashlib import sha256
import json
from pathlib import Path

from .window_metrics import paired, summarize


VIEWS = {"original": ["long", "long"], "recent5": ["short", "short"], "dual": ["long", "short"]}


def _check(ok, message):
    if not ok:
        raise ValueError(message)


def _same(registry, key, value):
    _check(registry.setdefault(key, value) == value, f"Conflicting frozen fields for {key}")


def replay(directory):
    directory = Path(directory)
    report = json.loads((directory / "results.json").read_bytes())
    for filename, key in (("folds.json", "folds_sha256"), ("predictions.jsonl.gz", "predictions_sha256")):
        _check(sha256((directory / filename).read_bytes()).hexdigest() == report[key], f"SHA-256 mismatch: {filename}")
    folds = json.loads((directory / "folds.json").read_bytes())
    protocol = report["protocol"]
    _check(protocol["windows_gate_and_intent"] == VIEWS, "Unexpected window protocol")
    _check(protocol["bootstrap_draws"] == 2000 and protocol["bootstrap_unit"] == "dialogue",
           "Unexpected bootstrap protocol")
    experts = protocol["frozen"]["experts"]
    _check(experts[-1] == "base", "Unexpected fallback executor")
    tests = {}
    for name, fold in folds.items():
        ids = [i for phase in ("fit", "router", "cal", "test") for i in fold["ids"][phase]]
        _check(len(ids) == len(set(ids)), f"Overlapping phase membership: {name}")
        inventories = [set(fold[k]) for k in ("known", "dev_unknown", "test_unknown")]
        _check(sum(map(len, inventories)) == len(set.union(*inventories)), f"Overlapping intent catalogs: {name}")
        tests[name] = set(fold["ids"]["test"])
    overlap = {name: {other: len(test & target) for other, target in tests.items()} for name, test in tests.items()}
    _check(overlap == report["test_overlap"], "Test overlap differs from fold manifest")

    groups, selected_outcomes, fixed_truth = defaultdict(list), {}, {}
    with gzip.open(directory / "predictions.jsonl.gz", "rt", encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            key = (row["encoder"], row["fold"], row["window"])
            encoder, name, variant = key
            _check(encoder in report["encoders"] and name in folds and variant in VIEWS,
                   f"Unexpected prediction group: {key}")
            uid = row["id"]
            _check(uid in tests[name], f"Non-test ID in {key}: {uid}")
            _check(row["top1"] in folds[name]["known"], f"Unknown top-1 prediction: {uid}")
            in_scope = row["intent"] in folds[name]["known"]
            _check(row["role"] == ("known" if in_scope else "test_unknown")
                   and (in_scope or row["intent"] in folds[name]["test_unknown"]), f"Invalid intent or role: {uid}")
            selected = row["selected"]
            _check(selected in experts and row["gate_oos"] == (selected == "base"), f"Invalid route: {uid}")
            _check(not row["correct"] or row["method_correct"], f"Exact success requires method success: {uid}")
            if selected == "base":
                _check(row["base_correct"] == row["correct"], f"Fallback outcome mismatch: {uid}")
            else:
                _check(not row["correct"] or row["expert_best_correct"], f"Expert outcome mismatch: {uid}")
            _same(selected_outcomes, (uid, selected), (row["correct"], row["method_correct"]))
            _same(fixed_truth, uid, (row["intent"], row["dialogue_id"], row["base_correct"], row["expert_best_correct"]))
            groups[key].append(row)

    for encoder, results in report["encoders"].items():
        _check(set(results) == set(folds), f"Fold list mismatch: {encoder}")
        for name, fold in folds.items():
            result = results[name]
            known = fold["known"]
            variants = {v: groups[encoder, name, v] for v in VIEWS}
            for variant, records in variants.items():
                _check([r["id"] for r in records] == fold["ids"]["test"], f"Test order mismatch: {encoder}/{name}/{variant}")
                _check(Counter(r["intent"] for r in records) == fold["metadata"]["counts_by_intent"]["test"],
                       f"Test intent counts differ: {encoder}/{name}")
                threshold = result["thresholds"][VIEWS[variant][0]]
                _check(all(r["gate_oos"] == (r["oos_score"] > -threshold) for r in records),
                       f"Gate disagrees with saved threshold: {encoder}/{name}/{variant}")
            for long, short, dual in zip(variants["original"], variants["recent5"], variants["dual"], strict=True):
                same_gate = (long["gate_oos"], long["oos_score"]) == (dual["gate_oos"], dual["oos_score"])
                same_intent = short["top1"] == dual["top1"]
                both_accepted = not short["gate_oos"] and not dual["gate_oos"]
                _check(same_gate and same_intent and (not both_accepted or short["selected"] == dual["selected"]),
                       f"Dual window disagrees with its component: {encoder}/{name}/{dual['id']}")
            metrics = {v: summarize(records, known) for v, records in variants.items()}
            _check(metrics == result["variants"], f"Variant metrics differ: {encoder}/{name}")
            contrasts = {f"{b}_minus_{a}": paired(variants[a], variants[b], known, seed=protocol["seed"])
                         for a, b in (("original", "recent5"), ("original", "dual"), ("recent5", "dual"))}
            _check(contrasts == result["paired"], f"Paired metrics differ: {encoder}/{name}")
    return len(groups), sum(len(rows) for rows in groups.values())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    args = parser.parse_args()
    groups, count = replay(args.run_dir)
    print(f"Verified {groups} encoder/fold/window groups and {count} prediction records.")


if __name__ == "__main__":
    main()
