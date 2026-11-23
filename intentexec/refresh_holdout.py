"""Confirm frozen routing on new SGD shards, quarantining all earlier dialogues and templates."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

import joblib
import numpy as np

from .data import load_sgd, public_schema
from .executors import BASE, EXPERTS
from .experiment import completed_output, save_json
from .metrics import call_parts
from .support_holdout import _select
from .support_study import score_matrix
from .transaction_executor import RepairedRetrievalPool
from .transaction_study import read_gzip, write_gzip, source_hash
from .window_frozen import digest
from .window_scoring import WindowFeatures, weights
from .window_splits import _dialogue_labels


ROOT = Path(__file__).resolve().parents[1]
SOURCES = (("sgd", "sgd_pilot_sha256"), ("sgd-support-holdout", "support_holdout_sha256"),
           ("sgd-routing-confirmation", "refresh_holdout_sha256"))
NAMES = (*EXPERTS, BASE)


def read_sources(data, catalog, sources=SOURCES):
    """Authenticate the evaluation corpus and every previous source used in the quarantine."""
    rows, dialogues, manifests = [], [], {}
    seen = set()
    for directory, lock in sources:
        path = data / directory
        manifest = json.loads((path / "manifest.json").read_bytes())
        expected = json.loads((ROOT / "configs" / (lock + ".json")).read_bytes())
        if manifest["sha256"] != expected or seen & (expected.keys() - {"schema.json"}):
            raise ValueError("Shard inventory changed or overlaps previous evaluation data")
        seen.update(expected)
        if manifests:
            original = next(iter(manifests.values()))
            if (any(manifest[k] != original[k] for k in ("revision", "split", "repository"))
                    or expected["schema.json"] != original["sha256"]["schema.json"]):
                raise ValueError("SGD versions or public schema differ")
        manifests[directory] = manifest
        rows.append(load_sgd(path, catalog))
        dialogues.append(_dialogue_labels(path, json.loads(catalog.read_bytes())))
    return rows, dialogues, manifests


def prepare(args):
    import torch
    torch.set_num_threads(4)
    sources = SOURCES
    if args.supplement:
        sources += (("sgd-routing-supplement", "refresh_supplement_sha256"),)
    sets, dialogues, manifests = read_sources(args.data, args.catalog, sources)
    prior = [r for part in sets[:-1] for r in part]
    prior_dialogues = {uid: labels for part in dialogues[:-1] for uid, labels in part.items()}
    fresh = sets[-1]
    gate_lock = json.loads((args.support / "development_lock.json").read_bytes())
    transaction_lock = json.loads((args.transaction / "results.json").read_bytes())
    saved_pool = args.legacy / "pool.joblib"
    if (digest(saved_pool) != gate_lock["expert_pool_sha256"] or
            digest(args.support / "development_lock.json") != transaction_lock["frozen_inputs"]["development_lock.json"] or
            source_hash(Path(__file__).with_name("transaction_executor.py")) != transaction_lock["repair_source_sha256_lf"]):
        raise ValueError("Frozen support model, executor or fit pool changed")
    saved = joblib.load(saved_pool)
    if saved["schema"] != public_schema(args.data / sources[-1][0]):
        raise ValueError("Public schema changed")
    bank = WindowFeatures(fresh, "minilm", args.model_path, args.cache)
    if bank.model_fingerprint != gate_lock["model"]:
        raise ValueError("Frozen embedding model differs")
    vectors = bank._embedding_matrix("short")
    pool = RepairedRetrievalPool(saved["fit"], saved["schema"], saved["config"]["seed"])
    outcomes = {}
    audit = {"shards": manifests[sources[-1][0]], "quarantined_prior_requests": len(prior),
             "quarantined_prior_dialogues": len(prior_dialogues), "new_requests_before_filter": len(fresh),
             "new_dialogues_before_filter": len(dialogues[-1]), "folds": {},
             "prior_manifests_sha256": {d: digest(args.data / d / "manifest.json") for d, _ in sources[:-1]},
             "encoder": bank.model_fingerprint, "pool_sha256": digest(saved_pool),
             "gate_lock_sha256": digest(args.support / "development_lock.json"),
             "repair_source_sha256_lf": transaction_lock["repair_source_sha256_lf"]}
    groups = []
    for name, fold in gate_lock["folds"].items():
        rows, isolation = _select(fresh, prior, dialogues[-1], prior_dialogues, fold)
        model_file = args.support / (name + ".joblib")
        if digest(model_file) != fold["model_sha256"]:
            raise ValueError("Frozen gate differs")
        gate, legacy = joblib.load(model_file)
        x = vectors[[bank.index[r["id"]] for r in rows]]
        text = [r["text"] for r in rows]
        w = weights(gate.recognizer._cosines(x), fold["temperature"])
        novelty = score_matrix(gate.components(text, x), fold["choice"]["weights"])["adaptive"]
        chosen = legacy.choose(w)
        rejected = novelty > fold["cutoffs"]["adaptive"]
        for idx, r in enumerate(rows):
            uid = r["id"]
            if uid not in outcomes:
                exact, tool = zip(*(call_parts(pool.execute(e, r["text"]), r["gold"])[:2] for e in NAMES))
                outcomes[uid] = {"exact": list(map(int, exact)), "tool": list(map(int, tool))}
            groups.append({"fold": name, "id": uid, "dialogue_id": r["dialogue_id"], "intent": r["intent"],
                           "unknown": r["role"] == "test_unknown", "rejected": bool(rejected[idx]),
                           "legacy": int(chosen[idx]), "weights": w[idx].tolist(), **outcomes[uid]})
        counts = Counter(r["role"] for r in rows)
        audit["folds"][name] = {"counts": {role: counts[role] for role in ("known", "test_unknown")}, **isolation,
                                "dialogues": len({r["dialogue_id"] for r in rows}),
                                "related_known": fold["related_known"], "known_order": fold["known"]}
        print(name, audit["folds"][name]["counts"], flush=True)
    out = completed_output(args.out)
    write_gzip(out / "observations.jsonl.gz", groups)
    audit.update({"unique_requests": len(outcomes), "fold_records": len(groups),
                  "observations_sha256": digest(out / "observations.jsonl.gz")})
    save_json(out / "cohort.json", audit)
    out.rename(args.out)


def evaluate(args):
    from .refresh_evaluation import POLICY_NAMES, summarize
    from .refresh_routing import choose_all
    observed = args.out / "observations.jsonl.gz"
    audit = json.loads((args.out / "cohort.json").read_bytes())
    if digest(observed) != audit["observations_sha256"]:
        raise ValueError("Independent cohort observations changed")
    lock = json.loads(args.lock.read_bytes())
    groups = defaultdict(list)
    for row in read_gzip(observed):
        groups[row.pop("fold")].append(row)
    if groups.keys() != lock["folds"].keys() or groups.keys() != audit["folds"].keys():
        raise ValueError("Development and confirmation folds differ")
    report = {"development_sha256": digest(args.lock), "observations_sha256": digest(observed),
              "policy_names": list(POLICY_NAMES), "folds": {}}
    for name, rows in groups.items():
        fold = lock["folds"][name]
        if fold["known"] != audit["folds"][name]["known_order"]:
            raise ValueError("Known intent dimensions differ from the frozen classifier")
        matrix = np.asarray([r["weights"] for r in rows])
        choices = choose_all(fold, matrix)
        for i, row in enumerate(rows):
            original = row["legacy"]
            row["choices"] = [len(EXPERTS) if row["rejected"] else
                              original if policy == "legacy" else int(choices[policy][i]) for policy in POLICY_NAMES]
        report["folds"][name] = summarize(rows, POLICY_NAMES, audit["folds"][name]["related_known"])
    path = args.out / "results.json"
    if args.stage == "replay":
        if report != json.loads(path.read_bytes()):
            raise ValueError("Independent confirmation metrics changed")
        print("Reproduced independent confirmation from the archived weights and executor outcomes.")
    elif path.exists():
        raise FileExistsError(path)
    else:
        save_json(path, report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "evaluate", "replay"))
    parser.add_argument("--supplement", action="store_true", help="Prepare separate transaction-enriched shards")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--lock", type=Path, default=Path("reports/refreshed_routing/development.json"))
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--catalog", type=Path, default=Path("configs/intent_catalog.json"))
    parser.add_argument("--support", type=Path, default=Path("artifacts/support-development"))
    parser.add_argument("--transaction", type=Path, default=Path("reports/transaction_execution"))
    parser.add_argument("--legacy", type=Path, default=Path("artifacts/verified-minilm"))
    parser.add_argument("--model-path", default="../code/models/models--sentence-transformers--all-MiniLM-L6-v2")
    parser.add_argument("--cache", type=Path, default=Path("artifacts/window-study-cache/features"))
    args = parser.parse_args()
    (prepare if args.stage == "prepare" else evaluate)(args)
