"""Freeze and verify the existing retrieval pool on the original non-fit requests."""
from hashlib import sha256
import json
from pathlib import Path

import joblib

from .data import public_schema
from .executors import BASE, EXPERTS, RetrievalPool
from .experiment import completed_output, read_rows, save_json, write_rows
from .metrics import call_parts


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def frozen_outcomes(rows, data_root, legacy, cache_root):
    """Reuse only a ledger made with the same saved pool, corpus and scoring code."""
    legacy, cache_root = Path(legacy), Path(cache_root)
    pkg = Path(__file__).parent
    names = (*EXPERTS, BASE)
    files = {name: digest(legacy / name) for name in
             ("pool.joblib", "split.json", "references.jsonl", "public.jsonl", "router_supervision.json")}
    files.update({name: digest(pkg / name) for name in ("data.py", "executors.py", "metrics.py", "window_frozen.py")})
    files["data_manifest"] = digest(Path(data_root) / "manifest.json")
    observed = sha256()
    for row in sorted(rows, key=lambda r: r["id"]):
        observed.update(json.dumps([row["id"], row["text"], row["intent"], row["gold"]],
                                   sort_keys=True, separators=(",", ":")).encode() + b"\n")
    files["input_rows"] = observed.hexdigest()
    key = sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
    folder = cache_root / key
    if not folder.exists():
        work = completed_output(folder)
        by_id = {r["id"]: r for r in rows}
        split = json.loads((legacy / "split.json").read_bytes())
        saved = joblib.load(legacy / "pool.joblib")
        if saved["config"]["backend"] != "retrieval" or saved["schema"] != public_schema(data_root):
            raise ValueError("Expected the saved local retrieval pool and its original SGD schema")
        if ([r["id"] for r in saved["fit"]] != split["fit"]
                or any(r != by_id[r["id"]] for r in saved["fit"])):
            raise ValueError("The saved expert fit data differs from its original split or corpus")
        pool = RetrievalPool(saved["fit"], saved["schema"], saved["config"]["seed"])
        original = {r["id"]: r for r in read_rows(legacy / "references.jsonl")}
        public = {r["id"]: r["text"] for r in read_rows(legacy / "public.jsonl")}
        supervision = json.loads((legacy / "router_supervision.json").read_bytes())
        if supervision["expert_order"] != list(EXPERTS):
            raise ValueError("Frozen expert order is inconsistent")
        target = dict(zip(supervision["row_ids"], supervision["success"], strict=True))
        eligible = sorted(split["router"] + split["cal"] + split["test"])
        if len(set(eligible)) != len(eligible) or set(eligible) & set(split["fit"]):
            raise ValueError("Legacy expert data partitions overlap")
        if not (set(original) <= set(split["test"]) and set(target) <= set(split["router"])
                and original.keys() == public.keys()):
            raise ValueError("Original records are inconsistent with the expert data split")
        records = []
        for uid in eligible:
            row = by_id[uid]
            calls = {n: pool.execute(n, row["text"]) for n in names}
            parts = [list(call_parts(calls[n], row["gold"])[:2]) for n in names]
            if uid in original and (original[uid]["calls"] != calls or original[uid]["gold"] != row["gold"]
                                    or public[uid] != row["text"]):
                raise ValueError(f"Frozen execution differs from original calls at {uid}")
            if uid in target and [int(exact) for exact, _ in parts[:-1]] != target[uid]:
                raise ValueError(f"Frozen execution differs from original supervision at {uid}")
            records.append({"id": uid, "outcomes": parts})
        write_rows(work / "outcomes.jsonl", records)
        save_json(work / "manifest.json", {"key": key, "files": files, "experts": list(names),
                  "outcome_order": ["exact", "service_and_method"], "unique_requests": len(eligible),
                  "expert_fit": len(saved["fit"]), "verified_reference_rows": len(original),
                  "verified_router_rows": len(target), "outcomes_sha256": digest(work / "outcomes.jsonl")})
        work.rename(folder)
    meta = json.loads((folder / "manifest.json").read_bytes())
    if meta["files"] != files or digest(folder / "outcomes.jsonl") != meta["outcomes_sha256"]:
        raise ValueError("Invalid cached expert outcomes")
    table = {r["id"]: r["outcomes"] for r in read_rows(folder / "outcomes.jsonl")}
    return table, meta
