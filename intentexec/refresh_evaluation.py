"""Evaluate second-layer routing against the frozen four-executor outcome ledger."""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np

from .data import load_sgd
from .executors import BASE, EXPERTS
from .experiment import completed_output, save_json
from .metrics import call_parts
from .transaction_executor import RepairedRetrievalPool
from .transaction_study import frozen_routes, read_gzip, rejection_masks, source_hash, write_gzip
from .window_frozen import digest


ORDER = (*EXPERTS, BASE)
BASE_ID = ORDER.index(BASE)
LEGACY = "legacy"
POLICY_NAMES = (LEGACY, "always_base", "fixed_expert", "fixed_all", "matrix3", "matrix4", "conservative", "selected")
BOOTSTRAPS = 2000


def build_ledger(args):
    """Run all four repaired backends and authenticate every previously published output."""
    inputs = {r["id"]: r for r in load_sgd(args.data, args.catalog)}
    previous = json.loads((args.transaction / "results.json").read_bytes())
    call_file = args.transaction / "calls.jsonl.gz"
    pool_file = args.legacy / "pool.joblib"
    if (digest(call_file) != previous["ledger_sha256"]
            or digest(pool_file) != previous["expert_pool_sha256"]
            or digest(args.data / "manifest.json") != previous["data_manifest_sha256"]
            or source_hash(Path(__file__).with_name("transaction_executor.py")) != previous["repair_source_sha256_lf"]):
        raise ValueError("Frozen execution inputs changed")
    saved = joblib.load(pool_file)
    pool = RepairedRetrievalPool(saved["fit"], saved["schema"], saved["config"]["seed"])
    encoded = []
    verified = 0
    for old in read_gzip(call_file):
        r = inputs[old["id"]]
        if any(r[k] != old[k] for k in ("gold", "dialogue_id", "intent")):
            raise ValueError("Prior row metadata changed")
        calls = {name: pool.execute(name, r["text"]) for name in ORDER}
        if any(calls[name] != value for name, value in old["new"].items()):
            raise ValueError(f"Previous calls disagree at {r['id']}")
        verified += len(old["new"])
        outcomes = [call_parts(calls[k], r["gold"]) for k in ORDER]
        encoded.append({"id": r["id"], "exact": [int(v[0]) for v in outcomes],
                        "tool": [int(v[1]) for v in outcomes]})
    if verified != previous["calls_per_version"] or len({r["id"] for r in encoded}) != previous["unique_requests"]:
        raise ValueError("Unexpected execution count")
    out = completed_output(args.out)
    ledger = out / "outcomes.jsonl.gz"
    write_gzip(ledger, encoded)
    save_json(out / "inputs.json", {"outcomes_sha256": digest(ledger), "source_calls_sha256": digest(call_file),
              "repair_source_sha256_lf": previous["repair_source_sha256_lf"],
              "verified_calls": verified, "all_calls": len(encoded) * len(ORDER), "executor_order": ORDER})
    out.rename(args.out)
    print(f"Verified {verified} prior calls; scored {len(encoded) * len(ORDER)} calls.")


def _summarize_group(outcomes, tools, choices, rejected, dialogues, policies):
    n, width = choices.shape
    if width != len(policies):
        raise ValueError("Unexpected number of policy choices")
    exact = np.take_along_axis(outcomes, choices, axis=1)
    method = np.take_along_axis(tools, choices, axis=1)
    clusters, inv, sizes = np.unique(dialogues, return_inverse=True, return_counts=True)
    old = exact[:, policies.index(LEGACY), None]
    base = outcomes[:, BASE_ID, None]
    diffs = np.stack([exact - old, exact - base], axis=-1)
    intervals = None
    if n:
        rng = np.random.default_rng(1118)
        count = len(clusters)
        per_dialogue = np.zeros((count, width, 2), dtype=np.float64)
        np.add.at(per_dialogue, inv, diffs)
        draws = rng.multinomial(count, np.full(count, 1.0 / count), size=BOOTSTRAPS).astype(np.float64)
        rates = (draws @ per_dialogue.reshape(count, -1)) / (draws @ sizes)[:, None]
        intervals = np.quantile(rates, [.025, .975], axis=0).reshape(2, width, 2)
    gate_oracle = np.where(rejected, outcomes[:, BASE_ID], outcomes.max(axis=1))
    expert_only = np.where(rejected, outcomes[:, BASE_ID], outcomes[:, :BASE_ID].max(axis=1))
    result = {"n": n, "dialogues": len(clusters), "gate_rejected": int(rejected.sum()),
              "oracle_exact": int(outcomes.max(axis=1).sum()), "gate_oracle_exact": int(gate_oracle.sum()),
              "gate_three_expert_oracle_exact": int(expert_only.sum()), "policies": {}}
    for j, name in enumerate(policies):
        selected = choices[:, j]
        values = {"exact": int(exact[:, j].sum()), "tool": int(method[:, j].sum()),
                  "accepted_to_base": int(((~rejected) & (selected == BASE_ID)).sum()),
                  "executor_counts": dict(zip(ORDER, np.bincount(selected, minlength=len(ORDER)).tolist()))}
        for i, ref in enumerate(("vs_legacy", "vs_base")):
            dx = diffs[:, j, i]
            values[ref] = {"gain": int((dx > 0).sum()), "loss": int((dx < 0).sum()), "net": int(dx.sum()),
                           "difference": float(dx.mean()) if n else None,
                           "ci95": intervals[:, j, i].tolist() if n else None}
        result["policies"][name] = values
    return result


def summarize(rows, policy_names, related_known):
    """Paired exact-call effects under a common hard OOS gate, per frozen cohort."""
    rows = list(rows)
    if not rows or len({r["id"] for r in rows}) != len(rows):
        raise ValueError("Expected a nonempty cohort with unique request IDs")
    def matrix(key, width, top):
        values = np.array([r[key] for r in rows])
        if values.shape != (len(rows), width) or not np.isin(values, np.arange(top + 1)).all():
            raise ValueError("Invalid " + key)
        return values.astype(int)
    exact = matrix("exact", len(ORDER), 1)
    tools = matrix("tool", len(ORDER), 1)
    choices = matrix("choices", len(policy_names), BASE_ID)
    rejected = np.asarray([r["rejected"] for r in rows], dtype=bool)
    if np.any(exact > tools) or (choices[rejected] != BASE_ID).any():
        raise ValueError("Inconsistent method outcome or violation of the hard gate")
    if (choices[:, policy_names.index("always_base")] != BASE_ID).any():
        raise ValueError("The always-base policy must use only the base")
    unknown = np.asarray([r["unknown"] for r in rows], dtype=bool)
    near = np.asarray([related_known is not None and r["intent"] == related_known for r in rows])
    if np.any(near & unknown):
        raise ValueError("The designated near-known label is unknown")
    dialogues = np.array([r["dialogue_id"] for r in rows])
    masks = {"all": np.ones(len(rows), bool), "known": ~unknown, "unknown": unknown, "near_known": near}
    return {k: _summarize_group(exact[m], tools[m], choices[m], rejected[m], dialogues[m], policy_names)
            for k, m in masks.items()}


def evaluate(args):
    from .refresh_routing import choose_all
    from .support_study import bank_for
    from .window_scoring import weights
    by_fold, _, support, lock = frozen_routes(args.support)
    old = json.loads((args.transaction / "results.json").read_bytes())
    ledger_lock = json.loads((args.ledger / "inputs.json").read_bytes())
    ledger_file = args.ledger / "outcomes.jsonl.gz"
    development = json.loads(args.development.read_bytes())
    if (ledger_lock["outcomes_sha256"] != digest(ledger_file)
            or ledger_lock["source_calls_sha256"] != old["ledger_sha256"] or list(ORDER) != ledger_lock["executor_order"]
            or development["executor_order"] != list(ORDER) or development["pool_sha256"] != old["expert_pool_sha256"]
            or development["support_lock_sha256"] != support["lock_sha256"]
            or development["repair_source_sha256"] != old["repair_source_sha256_lf"]
            or set(development["folds"]) != set(by_fold) or digest(args.data / "manifest.json") != old["data_manifest_sha256"]):
        raise ValueError("Execution ledger or confirmation inputs changed")
    ledger = {r["id"]: r for r in read_gzip(ledger_file)}
    embedding, bank = bank_for(load_sgd(args.data, args.catalog), args)
    if bank.model_fingerprint != lock["model"]:
        raise ValueError("Frozen MiniLM fingerprint mismatch")
    all_predictions, report = [], {}
    for name, records in by_fold.items():
        f = lock["folds"][name]
        model = args.support_models / (name + ".joblib")
        if digest(model) != f["model_sha256"] or f["known"] != development["folds"][name]["known"]:
            raise ValueError("Frozen stage-one or original matrix weights changed")
        gate, original_router = joblib.load(model)
        vec = embedding[[bank.index[r["id"]] for r in records]]
        cosines = gate.recognizer._cosines(vec)
        w = weights(cosines, f["temperature"])
        legacy = original_router.choose(w)
        if ([EXPERTS[i] for i in legacy] != [r["expert"] for r in records]
                or [f["known"][i] for i in np.argmax(w, axis=1)] != [r["top1"] for r in records]):
            raise ValueError("Recognition or original choices differ at " + name)
        proposed = {**choose_all(development["folds"][name], w), LEGACY: legacy}
        gate_mask = rejection_masks(records, support["folds"][name]["metrics"])[:, 0]
        choices = np.column_stack([proposed[n] for n in POLICY_NAMES])
        choices[gate_mask] = BASE_ID
        fold_rows = []
        for i, r in enumerate(records):
            rec = {field: r[field] for field in ("id", "intent", "unknown", "dialogue_id", "top1")}
            fold_rows.append({**rec, "fold": name, "rejected": bool(gate_mask[i]),
                              "choices": choices[i].tolist(), "exact": ledger[r["id"]]["exact"],
                              "tool": ledger[r["id"]]["tool"]})
        block = summarize(fold_rows, POLICY_NAMES, f["related_known"])
        for group in ("all", "known", "unknown"):
            previous = old["folds"][name][group]["routes"]
            for policy, history in [(LEGACY, "adaptive_from_cal"), ("always_base", "always_base")]:
                stats = block[group]["policies"][policy]
                if (stats["exact"] != previous[history]["exact"]["after"] or
                        stats["tool"] != previous[history]["service_method"]["after"]):
                    raise ValueError(f"Historical results disagree at {name}/{group}/{policy}")
        all_predictions.extend(fold_rows)
        report[name] = block
    out = completed_output(args.out)
    path = out / "choices.jsonl.gz"
    write_gzip(path, all_predictions)
    save_json(out / "results.json", {"policy_names": POLICY_NAMES, "executor_order": ORDER,
              "bootstrap_unit": "dialogue", "bootstrap_draws": BOOTSTRAPS, "bootstrap_seed": 1118,
              "development_sha256": digest(args.development), "old_calls_sha256": old["ledger_sha256"],
              "all_outcomes_sha256": ledger_lock["outcomes_sha256"], "support_predictions_sha256": support["prediction_sha256"],
              "verified_old_calls": ledger_lock["verified_calls"], "all_calls": ledger_lock["all_calls"],
              "unique_requests": len(ledger), "fold_observations": len(all_predictions), "choices_sha256": digest(path),
              "related_known": {name: f["related_known"] for name, f in lock["folds"].items()}, "folds": report})
    out.rename(args.out)
    print(f"Evaluated {len(all_predictions)} paired observations on {len(ledger)} unique requests.")


def replay(args):
    report = json.loads((args.out / "results.json").read_bytes())
    data_file = args.out / "choices.jsonl.gz"
    if report["choices_sha256"] != digest(data_file) or report["development_sha256"] != digest(args.development):
        raise ValueError("Development lock or paired choices differ")
    rows = read_gzip(data_file)
    if len(rows) != report["fold_observations"] or {r["fold"] for r in rows} != set(report["folds"]):
        raise ValueError("Fold coverage differs")
    for name, expected in report["folds"].items():
        fold = [r for r in rows if r["fold"] == name]
        if summarize(fold, report["policy_names"], report["related_known"][name]) != expected:
            raise ValueError("Metrics differ at " + name)
    print(f"Reproduced {len(rows)} paired observations across {len(report['folds'])} folds.")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=["ledger", "evaluate", "replay"])
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--legacy", type=Path, default=Path("artifacts/verified-minilm"))
    p.add_argument("--data", type=Path, default=Path("data/sgd-support-holdout"))
    p.add_argument("--catalog", type=Path, default=Path("configs/intent_catalog.json"))
    p.add_argument("--transaction", type=Path, default=Path("reports/transaction_execution"))
    p.add_argument("--support", type=Path, default=Path("reports/support_validation"))
    p.add_argument("--support-models", type=Path, default=Path("artifacts/support-development"))
    p.add_argument("--ledger", type=Path, default=Path("artifacts/refreshed-routing-ledger"))
    p.add_argument("--development", type=Path, default=Path("reports/refreshed_routing/development.json"))
    p.add_argument("--model-path", default="../code/models/models--sentence-transformers--all-MiniLM-L6-v2")
    p.add_argument("--cache", type=Path, default=Path("artifacts/window-study-cache/features"))
    args = p.parse_args()
    if args.stage == "ledger":
        build_ledger(args)
    elif args.stage == "evaluate":
        evaluate(args)
    else:
        replay(args)


if __name__ == "__main__":
    main()
