"""Measure dependence of the frozen transaction repair on SGD's fit-derived calendar."""
import argparse
import json
from pathlib import Path

import joblib

from .data import load_sgd
from .executors import BASE
from .metrics import call_parts
from .transaction_diagnosis import FOLDS
from .transaction_executor import RepairedRetrievalPool
from .transaction_study import frozen_routes, read_gzip, rejection_masks, source_hash
from .window_frozen import digest


def check(reports, routes, data, catalog, pool_path):
    result = json.loads((reports / "results.json").read_bytes())
    ledger = reports / "calls.jsonl.gz"
    if result["ledger_sha256"] != digest(ledger) or result["expert_pool_sha256"] != digest(pool_path):
        raise ValueError("Expected the archived paired study and its original fit pool")
    if source_hash(Path(__file__).with_name("transaction_executor.py")) != result["repair_source_sha256_lf"]:
        raise ValueError("Executor differs from the frozen repair")
    saved = joblib.load(pool_path)
    pool = RepairedRetrievalPool(saved["fit"], saved["schema"], saved["config"]["seed"])
    trained_day, trained_year = pool.anchor, pool.year
    observed = {r["id"]: r for r in read_gzip(ledger)}
    rows = {r["id"]: r for r in load_sgd(data, catalog)}
    by_fold, _, source, _ = frozen_routes(routes)
    settings = [("as_trained", trained_day, trained_year), ("year_only", None, trained_year),
                ("no_fit_calendar", None, None)]
    output = {"fit_reference_day": str(trained_day) if trained_day else None, "fit_year": trained_year,
              "policy": "adaptive_from_cal", "ledger_sha256": result["ledger_sha256"], "unknown": {}}
    for fold in FOLDS:
        records = by_fold[fold]
        deployed = rejection_masks(records, source["folds"][fold]["metrics"])[:, 0]
        targets = [(r["id"], BASE if reject else r["expert"])
                   for r, reject in zip(records, deployed, strict=True) if r["unknown"]]
        counts = {"n": len(targets)}
        for label, day, year in settings:
            pool.anchor, pool.year = day, year
            correct = 0
            for uid, executor in targets:
                call = pool.execute(executor, rows[uid]["text"])
                if label == "as_trained" and call != observed[uid]["new"][executor]:
                    raise ValueError(f"Published call differs for {uid}/{executor}")
                correct += call_parts(call, observed[uid]["gold"])[0]
            counts[label] = correct
        output["unknown"][fold] = counts
    return output


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reports", type=Path, default=Path("reports/transaction_execution"))
    p.add_argument("--routes", type=Path, default=Path("reports/support_validation"))
    p.add_argument("--data", type=Path, default=Path("data/sgd-support-holdout"))
    p.add_argument("--catalog", type=Path, default=Path("configs/intent_catalog.json"))
    p.add_argument("--pool", type=Path, default=Path("artifacts/verified-minilm/pool.joblib"))
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    report = check(args.reports, args.routes, args.data, args.catalog, args.pool)
    with args.out.open("x", encoding="utf8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
