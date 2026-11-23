"""Measure the frozen argument-grounding candidate on the original pilot router split."""
import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path

import joblib

from intentexec.data import load_sgd
from intentexec.executors import BASE, EXPERTS, RetrievalPool
from intentexec.metrics import call_parts
from intentexec.transaction_executor import RepairedRetrievalPool


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, default=Path("artifacts/verified-minilm"))
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    saved = joblib.load(args.run / "pool.joblib")
    split = json.loads((args.run / "split.json").read_bytes())
    rows = {r["id"]: r for r in load_sgd("data/sgd", "configs/intent_catalog.json")}
    if [r["id"] for r in saved["fit"]] != split["fit"] or set(split["router"]) & set(split["fit"]):
        raise ValueError("Original expert fit and router partitions must be disjoint and preserved")
    fixed = RepairedRetrievalPool(saved["fit"], saved["schema"], saved["config"]["seed"])
    original = RetrievalPool(saved["fit"], saved["schema"], saved["config"]["seed"])
    methods = (("legacy", original), ("grounded", fixed))
    groups = ("all", "BuyBusTicket", "ReserveLodging", "BuyEventTickets")
    n, counts = Counter(), Counter()
    for uid in split["router"]:
        row = rows[uid]
        included = ("all", row["intent"]) if row["intent"] in groups[1:] else ("all",)
        n.update(included)
        for variant, pool in methods:
            for name in (BASE, *EXPERTS):
                exact, method, _ = call_parts(pool.execute(name, row["text"]), row["gold"])
                for group in included:
                    counts[group, variant, name, "exact"] += int(exact)
                    counts[group, variant, name, "service_and_method"] += int(method)
    sources = [args.run / "pool.joblib", args.run / "split.json", Path("intentexec/transaction_executor.py")]
    result = {"source_sha256": {s.name: sha256(s.read_bytes()).hexdigest() for s in sources},
              "development_partition": "router", "training_rows": len(saved["fit"]),
              "calendar_anchor_from_fit": fixed.anchor.isoformat() if fixed.anchor else None,
              "year_from_fit": fixed.year, "groups": {}}
    for group in groups:
        result["groups"][group] = {"requests": n[group], **{
            variant: {name: {key: counts[group, variant, name, key] for key in ("exact", "service_and_method")}
                      for name in (BASE, *EXPERTS)} for variant, _ in methods}}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    blob = (json.dumps(result, indent=2) + "\n").encode()
    if args.out.exists() and args.out.read_bytes() != blob:
        raise FileExistsError(f"Choose a new path instead of overwriting an earlier run at {args.out}")
    args.out.write_bytes(blob)
    print(f"{n['all']} original router requests scored. Results saved to {args.out}.")


if __name__ == "__main__":
    main()
