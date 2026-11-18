"""Lock support scores on the pilot shards, then confirm on different SGD dialogues."""
import argparse
from collections import defaultdict
import gzip
import json
from pathlib import Path

import joblib
import numpy as np

from .data import load_sgd, public_schema
from .executors import EXPERTS, BASE, RetrievalPool
from .experiment import completed_output, save_json
from .metrics import call_parts
from .routing import Router
from .support_gate import SIGNALS, SupportGate, known_budget_cutoff
from .support_selection import select_weights
from .window_frozen import digest
from .window_scoring import WindowFeatures, calibrate, weights
from .window_splits import _dialogue_labels, build_folds


RATE = 29 / 377
STUDY = Path("configs/support_validation_folds.json")


def score_matrix(parts, chosen):
    z = np.column_stack([parts["z_" + k] for k in SIGNALS])
    return {"centroid": z[:, 0], "semantic": z[:, 1], "lexical": z[:, 2], "adaptive": z @ chosen}


def bank_for(rows, args):
    bank = WindowFeatures(rows, "minilm", args.model_path, args.cache)
    return bank._embedding_matrix("short"), bank


def prepare(args):
    out = completed_output(args.out)
    data = load_sgd(args.data, args.catalog)
    folds = build_folds(args.data, args.catalog, args.legacy, STUDY)
    vec, bank = bank_for(data, args)
    active = _dialogue_labels(args.data, json.loads(args.catalog.read_bytes()))
    supervision = json.loads((args.legacy / "router_supervision.json").read_bytes())
    if list(EXPERTS) != supervision["expert_order"]:
        raise ValueError("Expert order differs from the fixed pool")
    targets = dict(zip(supervision["row_ids"], supervision["success"], strict=True))
    spec = json.loads(STUDY.read_bytes())
    manifest = {"rate": RATE, "window": "recent5", "model": bank.model_fingerprint,
                "expert_pool_sha256": digest(args.legacy / "pool.joblib"),
                "supervision_sha256": digest(args.legacy / "router_supervision.json"),
                "holdout_config_sha256": digest(STUDY), "original_corpus_sha256": digest(args.data / "manifest.json"),
                "new_shard_lock_sha256": digest(args.extra_lock), "folds": {}}
    for name, split in folds.items():
        known = split["known"]
        fit, router, cal = (split[p] for p in ("fit","router","cal"))
        x, y, c = (vec[[bank.index[r["id"]] for r in phase]] for phase in (fit, router, cal))
        choice = select_weights(fit, x, known, router, y, dialogue_intents=active)
        gate = SupportGate(fit, x, known, router, y)
        cal_raw = gate.recognizer._cosines(c)
        legacy_threshold, temperature = calibrate(cal_raw, cal, known)
        model = Router("matrix").fit(weights(gate.recognizer._cosines(y), temperature),
                                     [r["intent"] for r in router], known, [targets[r["id"]] for r in router])
        parts = gate.components([r["text"] for r in cal], c)
        scores = score_matrix(parts, choice["weights"])
        in_scope = np.array([r["intent"] in known for r in cal])
        cutoffs = {method: float(known_budget_cutoff(values[in_scope], RATE)) for method, values in scores.items()}
        model_file = out / (name + ".joblib")
        joblib.dump((gate, model), model_file, compress=1)
        manifest["folds"][name] = {"known": known, "dev_unknown": split["dev_unknown"],
               "test_unknown": split["test_unknown"], "related_known": spec[name].get("related_known"),
               "choice": choice, "cutoffs": cutoffs, "legacy_raw_cut": -legacy_threshold, "temperature": temperature,
               "counts": {"fit":len(fit), "router":len(router), "cal_known":int(in_scope.sum())},
               "model_sha256": digest(model_file)}
        print(name, choice["candidate"], len(choice["rotations"]), flush=True)
    save_json(out / "development_lock.json", manifest)
    out.rename(args.out)


def validate(args):
    from .support_holdout import fresh_rows
    from .support_metrics import evaluate_records
    lock = json.loads((args.lock / "development_lock.json").read_bytes())
    if (lock["expert_pool_sha256"] != digest(args.legacy / "pool.joblib")
            or lock["original_corpus_sha256"] != digest(args.data / "manifest.json")
            or lock["new_shard_lock_sha256"] != digest(args.extra_lock)):
        raise ValueError("Experiment inputs differ from development lock")
    old = load_sgd(args.data, args.catalog)
    all_fresh = load_sgd(args.extra, args.catalog)
    fresh_digest = json.loads((args.extra / "manifest.json").read_bytes())["sha256"]
    expected = json.loads(args.extra_lock.read_bytes())
    if fresh_digest != expected.get("sha256", expected):
        raise ValueError("Validation corpus does not match its published digest list")
    x, bank = bank_for(all_fresh, args)
    if bank.model_fingerprint != lock["model"]:
        raise ValueError("Frozen encoder changed")
    work = completed_output(args.out)
    pool_data = joblib.load(args.legacy / "pool.joblib")
    if pool_data["config"]["backend"] != "retrieval" or pool_data["schema"] != public_schema(args.data):
        raise ValueError("Expected original local retrieval pool")
    pool = RetrievalPool(pool_data["fit"], pool_data["schema"], pool_data["config"]["seed"])
    answers = {}
    report = {"lock_sha256": digest(args.lock / "development_lock.json"),
              "fresh_manifest_sha256": digest(args.extra / "manifest.json"), "folds": {}}
    with (work / "predictions.jsonl.gz").open("wb") as raw:
        with gzip.GzipFile(filename="",fileobj=raw,mode="wb",mtime=0) as stream:
            for name, fold in lock["folds"].items():
                source = args.lock / (name + ".joblib")
                if digest(source) != fold["model_sha256"]:
                    raise ValueError("Scorer differs from development lock")
                gate, matrix = joblib.load(source)
                rows, meta = fresh_rows(args.data, args.extra, args.catalog, fold, old)
                vectors = x[[bank.index[r["id"]] for r in rows]]
                parts = gate.components([r["text"] for r in rows], vectors)
                candidate = score_matrix(parts, fold["choice"]["weights"])
                raw_intent = gate.recognizer._cosines(vectors)
                chosen = matrix.choose(weights(raw_intent, fold["temperature"]))
                pred = []
                for i, row in enumerate(rows):
                    uid, text = row["id"], row["text"]
                    selected = EXPERTS[chosen[i]]
                    # Same request and executor has the same immutable answer in every comparison.
                    for executor in (selected, BASE):
                        if (uid, executor) not in answers:
                            call = pool.execute(executor, text)
                            answers[uid, executor] = call_parts(call, row["gold"])[:2]
                    ee, em = answers[uid, selected]
                    be, bm = answers[uid, BASE]
                    rec = {"id": uid, "dialogue_id": row["dialogue_id"], "intent": row["intent"],
                           "unknown": row["intent"] not in fold["known"], "top1": fold["known"][raw_intent[i].argmax()],
                           "expert": selected, "expert_exact": ee, "expert_method": em, "base_exact": be, "base_method": bm,
                           "scores": {m: float(v[i]) for m,v in candidate.items()}, "raw_centroid": float(parts["centroid"][i])}
                    pred.append(rec)
                    stream.write((json.dumps({"fold": name, **rec}, separators=(",", ":")) + "\n").encode())
                report["folds"][name] = {"data": meta, "metrics": evaluate_records(pred, fold["cutoffs"],
                                               fold["legacy_raw_cut"], fold["related_known"], RATE)}
                print(name, len(rows), "complete", flush=True)
    report["unique_frozen_executions"] = len(answers)
    report["prediction_sha256"] = digest(work / "predictions.jsonl.gz")
    save_json(work / "results.json", report)
    work.rename(args.out)


def replay(directory):
    from .support_metrics import evaluate_records
    report = json.loads((directory / "results.json").read_bytes())
    lock_file = directory / "development_lock.json"
    lock = json.loads(lock_file.read_bytes())
    pred = directory / "predictions.jsonl.gz"
    if digest(lock_file) != report["lock_sha256"] or digest(pred) != report["prediction_sha256"]:
        raise ValueError("Invalid score archive or lock")
    groups = defaultdict(list)
    with gzip.open(pred, "rt", encoding="utf8") as stream:
        for r in map(json.loads, stream):
            groups[r.pop("fold")].append(r)
    if groups.keys() != lock["folds"].keys() or groups.keys() != report["folds"].keys():
        raise ValueError("Mismatched evaluation folds")
    for name, rows in groups.items():
        f = lock["folds"][name]
        if evaluate_records(rows, f["cutoffs"], f["legacy_raw_cut"], f["related_known"], lock["rate"]) != report["folds"][name]["metrics"]:
            raise ValueError("Metrics disagree at " + name)
    print(f"Reproduced {len(groups)} folds from {sum(map(len,groups.values()))} paired records.")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=["prepare","validate","replay"])
    p.add_argument("--out", type=Path, required=True, help="Fresh run directory, or published directory for replay")
    p.add_argument("--lock", type=Path, default=Path("artifacts/support-development"))
    p.add_argument("--data", type=Path, default=Path("data/sgd"))
    p.add_argument("--extra", type=Path, default=Path("data/sgd-support-holdout"))
    p.add_argument("--extra-lock", type=Path, default=Path("configs/support_holdout_sha256.json"))
    p.add_argument("--catalog", type=Path, default=Path("configs/intent_catalog.json"))
    p.add_argument("--legacy", type=Path, default=Path("artifacts/verified-minilm"))
    p.add_argument("--cache", type=Path, default=Path("artifacts/window-study-cache/features"))
    p.add_argument("--model-path", default="../code/models/models--sentence-transformers--all-MiniLM-L6-v2")
    args = p.parse_args()
    if args.stage == "replay":
        replay(args.out)
    else:
        import torch
        torch.set_num_threads(4)
        (prepare if args.stage == "prepare" else validate)(args)


if __name__ == "__main__":
    main()
