"""Lock routing improvements using only known supervision from the original SGD pilot."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np

from .data import load_sgd
from .metrics import call_parts
from .refresh_routing import BASE_INDEX, EXECUTORS, SELECTION_ORDER, choose_all, fit_tables
from .support_study import STUDY, bank_for, score_matrix
from .transaction_executor import RepairedRetrievalPool
from .window_frozen import digest
from .window_scoring import weights
from .window_splits import _components, build_folds


def group_folds(rows):
    components = _components(rows)
    ordered = sorted(components, key=lambda gid: hashlib.sha256(("refresh/" + gid).encode()).digest())
    if len(ordered) < 5:
        raise ValueError("Expected at least five dialogue/template groups")
    part_by_id = {r["id"]: i % 5 for i, gid in enumerate(ordered) for r in components[gid]}
    return np.array([part_by_id[r["id"]] for r in rows])


def fit_fold(router, known, w, oos, outcomes, old_experts):
    intents = np.array([r["intent"] for r in router])
    groups = group_folds(router)
    predictions = {"frozen": old_experts}
    for part in range(5):
        train = groups != part
        fold = fit_tables(w[train], intents[train], known, outcomes[train])
        for policy, ix in choose_all(fold, w[~train]).items():
            if policy not in predictions:
                predictions[policy] = np.empty(len(router), dtype=int)
            predictions[policy][~train] = ix
    metrics = {}
    for name, ix in predictions.items():
        deployed = np.where(oos, BASE_INDEX, ix)
        correct = outcomes[np.arange(len(router)), deployed]
        metrics[name] = {"exact": int(correct.sum()),
                         "executor_counts": dict(Counter(EXECUTORS[i] for i in deployed))}
    table = fit_tables(w, intents, known, outcomes)
    table["selected"] = max(SELECTION_ORDER, key=lambda name: metrics[name]["exact"])
    return {**table, "oof": metrics, "router_count": len(router), "oos_bypassed": int(oos.sum()),
            "oof_fold_sizes": np.bincount(groups, minlength=5).tolist()}


def prepare(args):
    legacy = args.legacy
    support_file = args.support / "development_lock.json"
    lock = json.loads(support_file.read_bytes())
    if digest(support_file) != digest(Path("reports/support_validation/development_lock.json")):
        raise ValueError("Support gate differs from published development lock")
    if digest(legacy / "pool.joblib") != lock["expert_pool_sha256"]:
        raise ValueError("Retrieval fit pool differs from the frozen reference")
    if digest(args.data / "manifest.json") != lock["original_corpus_sha256"] or digest(STUDY) != lock["holdout_config_sha256"]:
        raise ValueError("Pilot data or intent holdout definitions differ")
    saved = joblib.load(legacy / "pool.joblib")
    pool = RepairedRetrievalPool(saved["fit"], saved["schema"], saved["config"]["seed"])
    rows = load_sgd(args.data, args.catalog)
    folds = build_folds(args.data, args.catalog, legacy, STUDY)
    vectors, bank = bank_for(rows, args)
    if bank.model_fingerprint != lock["model"]:
        raise ValueError("Frozen sentence encoder changed")
    permitted = set(json.loads((legacy / "split.json").read_bytes())["router"])
    requests = {r["id"]: r for fold in folds.values() for r in fold["router"]}
    if not set(requests) <= permitted:
        raise ValueError("Unexpected non-router supervision")
    outcomes = {}
    for uid, row in sorted(requests.items()):
        outcomes[uid] = [int(call_parts(pool.execute(name, row["text"]), row["gold"])[0]) for name in EXECUTORS]
    print("Reconstructed", len(outcomes), "four-way exact labels on old known requests.", flush=True)
    payload = {"executor_order": list(EXECUTORS), "selection_order": list(SELECTION_ORDER),
               "selection_metric": "grouped_five_fold_oof_exact_with_frozen_adaptive_from_cal_gate",
               "pool_sha256": lock["expert_pool_sha256"], "support_lock_sha256": digest(support_file),
               "repair_source_sha256": digest(Path(__file__).with_name("transaction_executor.py")),
               "known_supervision": outcomes, "folds": {}}
    for name, split in folds.items():
        frozen = lock["folds"][name]
        model_path = args.support / (name + ".joblib")
        if digest(model_path) != frozen["model_sha256"] or split["known"] != frozen["known"]:
            raise ValueError("Known inventory or fitted recognition model differs")
        gate, original = joblib.load(model_path)
        rr = split["router"]
        if len(rr) != frozen["counts"]["router"] or any(r["role"] != "known" for r in rr):
            raise ValueError("Expected the frozen known-only router supervision")
        x = vectors[[bank.index[r["id"]] for r in rr]]
        w = weights(gate.recognizer._cosines(x), frozen["temperature"])
        scores = score_matrix(gate.components([r["text"] for r in rr], x), frozen["choice"]["weights"])
        oos = scores["adaptive"] > frozen["cutoffs"]["adaptive"]
        y = np.array([outcomes[r["id"]] for r in rr])
        result = fit_fold(rr, split["known"], w, oos, y, original.choose(w))
        result["router_ids"] = [r["id"] for r in rr]
        payload["folds"][name] = result
        print(name, result["selected"], {m: s["exact"] for m,s in result["oof"].items()}, flush=True)
    data = (json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.out.exists() and args.out.read_bytes() != data:
        raise FileExistsError("Choose a fresh development lock path")
    args.out.write_bytes(data)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--data", type=Path, default=Path("data/sgd"))
    p.add_argument("--catalog", type=Path, default=Path("configs/intent_catalog.json"))
    p.add_argument("--legacy", type=Path, default=Path("artifacts/verified-minilm"))
    p.add_argument("--support", type=Path, default=Path("artifacts/support-development"))
    p.add_argument("--cache", type=Path, default=Path("artifacts/window-study-cache/features"))
    p.add_argument("--model-path", default="../code/models/models--sentence-transformers--all-MiniLM-L6-v2")
    prepare(p.parse_args())
