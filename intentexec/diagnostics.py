"""Post-hoc switch and traffic-mix diagnostics on a frozen local experiment."""
import argparse
from collections import defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile

from .metrics import call_parts


def indexed_rows(path):
    rows = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row["id"] in rows:
                raise ValueError(f"Duplicate request in {path.name}: {row['id']}")
            rows[row["id"]] = row
    return rows


def switches(group, method):
    """Counts are paired on the same requests, using lookup as the comparator."""
    changed = [r for r in group if r["selected"][method] != r["selected"]["lookup"]]
    rescued = sum(r["success"][method] and not r["success"]["lookup"] for r in changed)
    harmed = sum(r["success"]["lookup"] and not r["success"][method] for r in changed)
    return {"n": len(group), "switches": len(changed), "rescued": rescued, "harmed": harmed,
            "unchanged_correctness_among_switches": len(changed) - rescued - harmed, "net": rescued - harmed}


def accuracy(group, key, name):
    return sum(r[key][name] for r in group) / len(group) if group else None


def analyze(run):
    paths = {name: run / name for name in ("public.jsonl", "decisions.jsonl", "references.jsonl", "summary.json")}
    public, decisions, references = (indexed_rows(paths[name]) for name in
                                     ("public.jsonl", "decisions.jsonl", "references.jsonl"))
    summary = json.loads(paths["summary.json"].read_bytes())
    if not public or public.keys() != decisions.keys() or public.keys() != references.keys():
        raise ValueError("Frozen decisions, public inputs, and references must contain the same unique requests")
    if len(public) != summary["evaluated"]:
        raise ValueError("Request count differs from summary")
    known = summary["known"]
    policies = sorted(summary["quality"])
    experts = summary["expert_ids"]
    cases, top1_groups = [], defaultdict(list)
    for uid in public:
        decision, reference = decisions[uid], references[uid]
        selected, calls = decision["selected"], reference["calls"]
        weights = decision["weights"]
        if len(weights) != len(known) or not all(math.isfinite(x) for x in weights):
            raise ValueError(f"Invalid intent weights at {uid}")
        if set(selected) != set(policies) or not set(experts) <= calls.keys():
            raise ValueError(f"Policy or expert mismatch at {uid}")
        by_executor = {name: call_parts(prediction, reference["gold"])[0] for name, prediction in calls.items()}
        case = {"selected": selected, "fixed": by_executor, "is_known": reference["intent"] in known,
                "gate_oos": decision["pred_oos"],
                "success": {p: by_executor[selected[p]] for p in policies}}
        cases.append(case)
        if not case["gate_oos"]:
            top1_groups[known[max(range(len(known)), key=weights.__getitem__)]].append(case)

    for policy in policies:
        if not math.isclose(accuracy(cases, "success", policy), summary["quality"][policy]["correct"]):
            raise ValueError(f"Recomputed correct-call rate disagrees with frozen summary for {policy}")
    alternatives = ("matrix", "mlp")
    paired = {p: switches(cases, p) for p in alternatives}
    by_top1 = {label: {p: switches(group, p) for p in alternatives}
               for label, group in sorted(top1_groups.items())}
    quadrants = {}
    # Gate labels describe the shared recognizer, even for the ungated policies.
    for is_known, truth in [(True, "known"), (False, "unknown")]:
        for gate, decision in [(False, "accepted"), (True, "oos")]:
            group = [r for r in cases if r["is_known"] == is_known and r["gate_oos"] == gate]
            quadrants[truth + "_" + decision] = {"n": len(group), "accuracy": {
                p: accuracy(group, "success", p) for p in policies}}
    known_cases = [r for r in cases if r["is_known"]]
    unknown_cases = [r for r in cases if not r["is_known"]]
    observed = len(unknown_cases) / len(cases)
    mixture = []
    for frac in sorted({0., .1, .25, .5, .75, 1., observed}):
        estimates = {}
        for p in policies:
            k, u = accuracy(known_cases, "success", p), accuracy(unknown_cases, "success", p)
            if frac == 0:
                estimates[p] = k
            elif frac == 1:
                estimates[p] = u
            else:
                estimates[p] = (1-frac) * k + frac * u if k is not None and u is not None else None
        mixture.append({"unknown_fraction": frac, "estimated_accuracy": estimates})
    return {"encoder": summary["config"]["encoder"], "evaluated": len(cases), "observed_unknown_fraction": observed,
            "input_sha256": {k: hashlib.sha256(v.read_bytes()).hexdigest() for k, v in paths.items()},
            "scope": "Top1 groups include accepted requests only; switch counts use lookup as paired baseline. "
                     "Quadrants describe the common gate, including for methods that bypass it. "
                     "Mixtures reweight empirical known/unknown accuracy holding their conditional distributions fixed.",
            "switches_vs_lookup": paired, "accepted_by_predicted_top1": by_top1,
            "truth_by_gate": quadrants, "oos_mix_sensitivity": mixture,
            "always_use_one_expert_full_test": {e: accuracy(cases, "fixed", e) for e in experts}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    if args.out.resolve().is_relative_to(args.run_dir.resolve()):
        parser.error("Write diagnostics outside the frozen run directory")
    report = analyze(args.run_dir)
    args.out.parent.mkdir(exist_ok=True, parents=True)
    with tempfile.NamedTemporaryFile(dir=args.out.parent, mode="w", encoding="utf-8", delete=False) as tmp:
        json.dump(report, tmp, indent=2, allow_nan=False)
        tmp.write("\n")
        temporary = tmp.name
    os.replace(temporary, args.out)
    print(f"{report['encoder']}: {report['evaluated']} evaluated, {args.out}")


if __name__ == "__main__":
    main()
