"""Compare two local executors on the routes frozen by the support-gate study."""
import argparse
from collections import defaultdict
import gzip
import json
from pathlib import Path

import joblib
import numpy as np

from .data import load_sgd
from .executors import BASE, EXPERTS, RetrievalPool
from .experiment import completed_output, save_json
from .metrics import call_parts
from .near_gate_eval import apply_threshold
from .window_frozen import digest


METRICS = ("service", "method_name", "service_method", "argument_keys", "arguments", "exact")
POLICIES = ("adaptive_from_cal", "adaptive_matched", "centroid_default", "always_matrix_expert", "always_base")
REPEATS = 2000
SEED = 1118


def read_gzip(path):
    with gzip.open(path, "rt", encoding="utf8") as stream:
        return [json.loads(line) for line in stream]


def write_gzip(path, records):
    with path.open("wb") as raw, gzip.GzipFile(filename="", fileobj=raw, mode="wb", mtime=0) as stream:
        for row in records:
            stream.write((json.dumps(row, separators=(",", ":"), ensure_ascii=False, sort_keys=True) + "\n").encode("utf8"))


def frozen_routes(directory):
    source = json.loads((directory / "results.json").read_bytes())
    lock = json.loads((directory / "development_lock.json").read_bytes())
    pred_path = directory / "predictions.jsonl.gz"
    if source["prediction_sha256"] != digest(pred_path) or source["lock_sha256"] != digest(directory / "development_lock.json"):
        raise ValueError("Frozen route files differ from the published support experiment")
    by_fold, queries = defaultdict(list), defaultdict(set)
    for row in read_gzip(pred_path):
        if row["expert"] not in EXPERTS:
            raise ValueError("Unknown frozen expert")
        by_fold[row["fold"]].append(row)
        queries[row["id"]].update([row["expert"], BASE])
    if set(by_fold) != set(source["folds"]) or sum(map(len, queries.values())) != source["unique_frozen_executions"]:
        raise ValueError("Frozen cohort differs from published execution coverage")
    return by_fold, queries, source, lock


def execute_pool(pool, old_inputs, tasks):
    records = {}
    for uid in sorted(tasks):
        row = old_inputs[uid]
        calls = {key: pool.execute(key, row["text"]) for key in sorted(tasks[uid])}
        records[uid] = {"id": uid, "dialogue_id": row["dialogue_id"], "intent": row["intent"],
                        "gold": row["gold"], "calls": calls}
    return records


def check_baseline(by_fold, old):
    for name, cohort in by_fold.items():
        if len({r["id"] for r in cohort}) != len(cohort):
            raise ValueError("Duplicate request in " + name)
        for row in cohort:
            ref = old[row["id"]]
            if ref["dialogue_id"] != row["dialogue_id"] or ref["intent"] != row["intent"]:
                raise ValueError("Reference labels differ from the frozen cohort")
            for role, key in (("expert", row["expert"]), ("base", BASE)):
                exact, method, _ = call_parts(ref["calls"][key], ref["gold"])
                if (exact, method) != (row[role + "_exact"], row[role + "_method"]):
                    raise ValueError(f"Baseline does not reproduce {name}/{row['id']}/{key}")


def score(call, gold):
    exact, method, args = call_parts(call, gold)
    if not isinstance(call, dict) or not isinstance(call.get("arguments"), dict):
        return [False] * len(METRICS)
    return [call.get("service") == gold["service"], call.get("method") == gold["method"], method,
            set(call["arguments"]) == set(gold["arguments"]), args, exact]


def rejection_masks(rows, metrics):
    ids = [r["id"] for r in rows]
    adaptive = [r["scores"]["adaptive"] for r in rows]
    centroid = [r["raw_centroid"] for r in rows]
    return np.column_stack((
        apply_threshold(adaptive, ids, metrics["from_cal"]["adaptive"]["boundary"]),
        apply_threshold(adaptive, ids, metrics["matched"]["adaptive"]["boundary"]),
        apply_threshold(centroid, ids, metrics["legacy_calibrated"]["boundary"]),
        np.zeros(len(rows), dtype=bool), np.ones(len(rows), dtype=bool),
    ))


def subgroup_metrics(before, after, dialogues):
    """Use the same dialogue bootstrap draws for every path and metric within this cohort."""
    n = len(before)
    delta = after.astype(np.int8) - before.astype(np.int8)
    old = before.sum(axis=0)
    new = after.sum(axis=0)
    gains = (delta == 1).sum(axis=0)
    losses = (delta == -1).sum(axis=0)
    unique, inverse, sizes = np.unique(dialogues, return_inverse=True, return_counts=True)
    if n:
        # A resampled dialogue contributes every one of its eligible requests.
        count = len(unique)
        changes = np.zeros((count, len(POLICIES), len(METRICS)), dtype=np.float64)
        np.add.at(changes, inverse, delta)
        rng = np.random.default_rng(SEED)
        draws = rng.multinomial(count, np.full(count, 1.0 / count), size=REPEATS).astype(np.float64)
        rates = (draws @ changes.reshape(count, -1)) / (draws @ sizes)[:, None]
        intervals = np.quantile(rates, [.025, .975], axis=0).reshape(2, len(POLICIES), len(METRICS))
    result = {"n": n, "dialogues": len(unique), "routes": {}}
    for p, policy in enumerate(POLICIES):
        result["routes"][policy] = {}
        for m, metric in enumerate(METRICS):
            result["routes"][policy][metric] = {
                "before": int(old[p, m]), "after": int(new[p, m]),
                "gain": int(gains[p, m]), "loss": int(losses[p, m]),
                "delta_count": int(new[p, m] - old[p, m]),
                "difference": float((new[p, m] - old[p, m]) / n) if n else None,
                "ci95": intervals[:, p, m].tolist() if n else None,
            }
    return result


def evaluate_ledger(records, by_fold, queries, source):
    if set(records) != set(queries):
        raise ValueError("Ledger and frozen cohort have different request IDs")
    for uid, r in records.items():
        if set(r["old"]) != queries[uid] or set(r["new"]) != queries[uid]:
            raise ValueError(f"Incomplete executor outputs for {uid}")
    check_baseline(by_fold, {u: {**r, "calls": r["old"]} for u, r in records.items()})
    scores = {(uid, k, mode): score(r[mode][k], r["gold"]) for uid, r in records.items()
              for k in queries[uid] for mode in ("old", "new")}
    report = {}
    for name, rows in by_fold.items():
        old_metrics = source["folds"][name]["metrics"]
        masks = rejection_masks(rows, old_metrics)
        unknown = np.array([r["unknown"] for r in rows])
        dialogues = np.array([r["dialogue_id"] for r in rows])
        gated = {}
        for mode in ("old", "new"):
            base = np.array([scores[r["id"], BASE, mode] for r in rows])
            expert = np.array([scores[r["id"], r["expert"], mode] for r in rows])
            gated[mode] = np.where(masks[:, :, None], base[:, None, :], expert[:, None, :])
        report[name] = {}
        for group, mask in (("all", np.ones(len(rows), bool)), ("known", ~unknown), ("unknown", unknown)):
            block = subgroup_metrics(gated["old"][mask], gated["new"][mask], dialogues[mask])
            block["routed_to_base"] = dict(zip(POLICIES, masks[mask].sum(axis=0).tolist()))
            report[name][group] = block
        # Check the actual deployed, matched, and historical decisions against all published end points.
        history = (old_metrics["from_cal"]["adaptive"], old_metrics["matched"]["adaptive"], old_metrics["legacy_calibrated"])
        for policy, previous in zip(POLICIES[:3], history, strict=True):
            for group, prefix in (("all", ""), ("known", "known_"), ("unknown", "unknown_")):
                stats = report[name][group]["routes"][policy]
                if (stats["exact"]["before"] != previous[prefix + "exact_correct"]
                        or stats["service_method"]["before"] != previous[prefix + "method_correct"]):
                    raise ValueError(f"Historical route endpoints differ in {name}/{policy}/{group}")
            for group in ("known", "unknown"):
                if report[name][group]["routed_to_base"][policy] != previous[group + "_rejected"]:
                    raise ValueError(f"Historical gate decisions differ in {name}/{policy}/{group}")
    return report


def run(args):
    by_fold, queries, source, lock = frozen_routes(args.routes)
    inputs = {f: digest(args.routes / f) for f in ("development_lock.json", "predictions.jsonl.gz", "results.json")}
    if args.stage == "replay":
        result = json.loads((args.out / "results.json").read_bytes())
        path = args.out / "calls.jsonl.gz"
        if result["frozen_inputs"] != inputs or result["ledger_sha256"] != digest(path):
            raise ValueError("Archive fingerprint mismatch")
        records = {r["id"]: r for r in read_gzip(path)}
        if evaluate_ledger(records, by_fold, queries, source) != result["folds"]:
            raise ValueError("Replayed endpoint or confidence interval differs")
        print(f"Reproduced {len(by_fold)} folds; {len(records)} unique requests.")
        return
    pool_path = args.legacy / "pool.joblib"
    if digest(pool_path) != lock["expert_pool_sha256"] or digest(args.data / "manifest.json") != source["fresh_manifest_sha256"]:
        raise ValueError("Study data or frozen pool differs")
    saved = joblib.load(pool_path)
    input_rows = {r["id"]: r for r in load_sgd(args.data, args.catalog)}
    if not set(queries) <= set(input_rows) or saved["config"]["backend"] != "retrieval":
        raise ValueError("Missing data for the local frozen routing experiment")
    construction = (saved["fit"], saved["schema"], saved["config"]["seed"])
    out = completed_output(args.out)
    if args.stage == "baseline":
        old = execute_pool(RetrievalPool(*construction), input_rows, queries)
        check_baseline(by_fold, old)
        path = out / "old_calls.jsonl.gz"
        write_gzip(path, old.values())
        save_json(out / "baseline.json", {"frozen_inputs": inputs, "old_calls_sha256": digest(path)})
    else:
        repair_path = Path(__file__).with_name("transaction_executor.py")
        repair_digest = source_hash(repair_path)
        if args.repair_sha256 and args.repair_sha256 != repair_digest:
            raise ValueError("Execution repair differs from the specified development lock")
        from .transaction_executor import RepairedRetrievalPool
        baseline = json.loads((args.baseline / "baseline.json").read_bytes())
        path = args.baseline / "old_calls.jsonl.gz"
        if baseline["frozen_inputs"] != inputs or digest(path) != baseline["old_calls_sha256"]:
            raise ValueError("Baseline cache differs from verified original calls")
        old = {r["id"]: r for r in read_gzip(path)}
        fresh = execute_pool(RepairedRetrievalPool(*construction), input_rows, queries)
        records = {}
        for uid, r in fresh.items():
            ref = old[uid]
            if any(r[k] != ref[k] for k in ("gold", "intent", "dialogue_id")):
                raise ValueError("Request metadata no longer matches the previous run")
            records[uid] = {"id": uid, "dialogue_id": r["dialogue_id"], "intent": r["intent"],
                            "gold": r["gold"], "old": ref["calls"], "new": r["calls"]}
        path = out / "calls.jsonl.gz"
        write_gzip(path, records.values())
        folds = evaluate_ledger(records, by_fold, queries, source)
        if source_hash(repair_path) != repair_digest:
            raise ValueError("Execution repair changed during evaluation")
        save_json(out / "results.json", {"frozen_inputs": inputs, "ledger_sha256": digest(path),
                  "repair_source_sha256_lf": repair_digest,
                  "expert_pool_sha256": lock["expert_pool_sha256"], "data_manifest_sha256": source["fresh_manifest_sha256"],
                  "fit_requests": len(saved["fit"]), "unique_requests": len(records),
                  "unique_dialogues": len({r["dialogue_id"] for r in records.values()}),
                  "fold_observations": sum(map(len, by_fold.values())), "calls_per_version": sum(map(len, queries.values())),
                  "primary_policy": POLICIES[0], "bootstrap_unit": "dialogue within cohort",
                  "bootstrap_draws": REPEATS, "bootstrap_seed": SEED, "folds": folds})
    out.rename(args.out)
    print(f"Verified {sum(map(len, by_fold.values()))} baseline rows; saved {args.out}")


def source_hash(path):
    from hashlib import sha256
    return sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["baseline", "evaluate", "replay"])
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--routes", type=Path, default=Path("reports/support_validation"))
    parser.add_argument("--baseline", type=Path, default=Path("artifacts/transaction-baseline"))
    parser.add_argument("--legacy", type=Path, default=Path("artifacts/verified-minilm"))
    parser.add_argument("--data", type=Path, default=Path("data/sgd-support-holdout"))
    parser.add_argument("--catalog", type=Path, default=Path("configs/intent_catalog.json"))
    parser.add_argument("--repair-sha256", help="Expected LF-normalized digest of the frozen executor source")
    run(parser.parse_args())
