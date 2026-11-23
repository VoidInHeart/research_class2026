"""Publish a new inference run with a repaired local execution pool and a refreshed matrix."""
import argparse
import json
from pathlib import Path
from shutil import copyfile

import joblib

from .data import load_sgd, public_schema, split_rows
from .executors import BASE, EXPERTS, make_pool
from .experiment import all_calls, completed_output, read_rows, save_json
from .routing import Router
from .window_frozen import digest


POLICIES = ["matrix", "matrix_no_gate", "always_base"]


def source_rows(rows, saved, split, previous):
    original = split_rows(rows, seed=saved["config"]["seed"])
    for part in ("fit", "router", "cal", "test"):
        if [r["id"] for r in original[part]] != split[part]:
            raise ValueError(f"Saved {part} partition differs from original corpus and seed")
    if (saved["fit"] != original["fit"] or split["known"] != original["known"]
            or any(split[k] != original[k] for k in ("dev_unknown", "test_unknown"))):
        raise ValueError("Saved fit examples or intent inventories differ from original corpus")
    used = split["router_used"]
    router = {r["id"]: r for r in original["router"]}
    if (not used or len(set(used)) != len(used) or not set(used) <= router.keys()
            or previous["expert_order"] != list(EXPERTS) or previous["row_ids"] != used):
        raise ValueError("Saved router supervision or executor order differs from router_used")
    return [router[uid] for uid in used]


def refresh(directory, destination, data, catalog):
    directory, destination = Path(directory).resolve(), Path(destination).resolve()
    data, catalog = Path(data), Path(catalog)
    if destination == directory or directory in destination.parents:
        raise ValueError("Choose an output outside the source run directory")
    if destination.exists():
        raise FileExistsError(f"Output already exists: {destination}")
    names = ["pool.joblib", "routing.joblib", "split.json", "router_supervision.json", "summary.json"]
    public = directory / "public.jsonl"
    if public.exists():
        names.append("public.jsonl")
    inputs = {name: digest(directory / name) for name in names}
    saved = joblib.load(directory / "pool.joblib")
    bundle = joblib.load(directory / "routing.joblib")
    split = json.loads((directory / "split.json").read_bytes())
    previous = json.loads((directory / "router_supervision.json").read_bytes())
    summary = json.loads((directory / "summary.json").read_bytes())
    if (saved["config"]["backend"] not in ("retrieval", "retrieval-grounded")
            or saved["schema"] != public_schema(data) or summary["catalog_sha256"] != digest(catalog)):
        raise ValueError("Expected a local retrieval source run with its original schema and catalog")
    manifest = data / "manifest.json"
    if summary["data_manifest"] != (json.loads(manifest.read_bytes()) if manifest.exists() else None):
        raise ValueError("Source data manifest differs")
    rows = load_sgd(data, catalog)
    router_rows = source_rows(rows, saved, split, previous)
    recognizer = bundle["recognizer"]
    if (bundle["experts"] != list(EXPERTS) or bundle["base"] != BASE
            or bundle["known"] != split["known"] or recognizer.known != split["known"]):
        raise ValueError("Frozen recognizer or executor inventory inconsistent with the source run")
    if public.exists():
        public_rows = read_rows(public)
        texts = {r["id"]: r["text"] for r in rows}
        test_ids = [r["id"] for r in public_rows]
        if (test_ids != split["test_used"] or len(set(test_ids)) != len(test_ids)
                or not set(test_ids) <= set(split["test"])
                or any(texts[r["id"]] != r["text"] for r in public_rows)):
            raise ValueError("Saved public profiling inputs differ from source requests")
    config = {**saved["config"], "backend": "retrieval-grounded"}
    pool = make_pool(saved["fit"], saved["schema"], config)
    _, outcomes = all_calls(pool, router_rows, EXPERTS)
    input_weights, _ = recognizer.predict([r["text"] for r in router_rows])
    matrix = Router("matrix").fit(input_weights, [r["intent"] for r in router_rows], split["known"], outcomes)
    work = completed_output(destination)
    copyfile(directory / "split.json", work / "split.json")
    if public.exists():
        copyfile(public, work / "public.jsonl")
    joblib.dump({**bundle, "routers": {"matrix": matrix}, "supported_policies": POLICIES}, work / "routing.joblib")
    joblib.dump({**saved, "config": config}, work / "pool.joblib", compress=3)
    save_json(work / "router_supervision.json", {"row_ids": split["router_used"], "expert_order": list(EXPERTS),
                                                "success": outcomes.astype(int).tolist()})
    report = {"source_sha256": inputs, "fit_requests": len(saved["fit"]), "router_requests": len(router_rows),
              "expert_successes": dict(zip(EXPERTS, outcomes.sum(axis=0).astype(int).tolist())),
              "backend": config["backend"], "window": recognizer.window, "encoder": recognizer.encoder,
              "oos_gate": "adaptive-support" if recognizer.support_gate is not None else "centroid",
              "supported_policies": POLICIES, "data_manifest": summary["data_manifest"],
              "catalog_sha256": summary["catalog_sha256"]}
    save_json(work / "summary.json", report)
    if any(digest(directory / name) != hash_value for name, hash_value in inputs.items()):
        raise ValueError("Source run changed during migration")
    work.rename(destination)
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--sgd-root", type=Path, default=Path("data/sgd"))
    p.add_argument("--catalog", type=Path, default=Path("configs/intent_catalog.json"))
    a = p.parse_args()
    print(json.dumps(refresh(a.run_dir, a.out, a.sgd_root, a.catalog), indent=2))
