"""Fit on the specified partitions, route from public inputs, then score execution."""
from collections import Counter
from hashlib import sha256
from pathlib import Path
from importlib.metadata import PackageNotFoundError, version
import json
import platform
import sys
import time

import joblib
import numpy as np

from .data import load_sgd, public_schema, split_rows
from .executors import make_pool
from .metrics import call_parts, paired_interval, rate, summarize
from .routing import EmbeddingRouter, Recognizer, Router

POLICIES = ("always_base", "fixed", "lookup", "matrix", "mlp", "embedding", "matrix_no_gate")


def save_json(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf8")


def write_rows(path, rows):
    with path.open("w", encoding="utf8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def read_rows(path):
    with path.open(encoding="utf8") as handle:
        return [json.loads(line) for line in handle]


def sample_rows(rows, count, seed):
    if count == 0 or count >= len(rows):
        return list(rows)
    if count < 0:
        raise ValueError("Sample size must be nonnegative")
    return sorted(rows, key=lambda r: sha256(f"{seed}/{r['id']}".encode()).digest())[:count]


def completed_output(path):
    if path.exists():
        raise FileExistsError(f"Run already exists: {path}; choose a fresh output path")
    path.parent.mkdir(parents=True, exist_ok=True)
    # Publish only after the full run succeeds. A unique work directory retains interrupted progress.
    import tempfile
    return Path(tempfile.mkdtemp(prefix=path.name + ".working-", dir=path.parent))


def measure_ms(start):
    return (time.perf_counter() - start) * 1000


def provenance(extra_modules=()):
    package = Path(__file__).parent
    versions = {}
    for name in ("numpy", "scipy", "scikit-learn", "joblib", "torch", "transformers", "peft"):
        try:
            versions[name] = version(name)
        except PackageNotFoundError:
            versions[name] = None
    modules = ("experiment", "routing", "executors", "lora", "data", "metrics") + extra_modules
    return {"versions": versions,
            "source_sha256": {m: sha256((package / (m + ".py")).read_bytes()).hexdigest() for m in modules}}


def all_calls(pool, rows, names):
    outputs, success = {}, []
    for row in rows:
        results = {name: pool.execute(name, row["text"]) for name in names}
        outputs[row["id"]] = results
        success.append([call_parts(results[name], row["gold"])[0] for name in names])
    return outputs, np.asarray(success, dtype=np.float32)


def policy_decisions(bundle, public_rows):
    texts = [r["text"] for r in public_rows]
    encoder = bundle["recognizer"]
    features = encoder.encode(texts)
    weights, oos = encoder.predict_encoded(features, texts)
    experts, base = bundle["experts"], bundle["base"]
    accepted = ~np.asarray(oos, dtype=bool)
    ids = {}
    for name, router in bundle["routers"].items():
        chosen = np.full(len(texts), base, dtype=object)
        if accepted.any():
            inputs = features if name == "embedding" else weights
            chosen[accepted] = np.asarray(experts)[router.choose(inputs[accepted])]
        ids[name] = chosen
    ungated = np.asarray(experts)[bundle["routers"]["matrix"].choose(weights)]
    records = []
    for i, row in enumerate(public_rows):
        selections = {"always_base": base, "matrix_no_gate": str(ungated[i])}
        selections.update({p: str(v[i]) for p, v in ids.items()})
        records.append({"id": row["id"], "pred_oos": bool(oos[i]), "weights": weights[i].tolist(),
                        "selected": selections})
    return records


def infer_one(bundle, pool, policy, text):
    """Deployment path. One executor call, including when the detector bypasses stage two."""
    start = time.perf_counter()
    gate_ms = route_ms = 0.
    gate = None
    intent_result = None
    name = bundle["base"]
    if policy != "always_base":
        features = bundle["recognizer"].encode([text])
        weights, oos = bundle["recognizer"].predict_encoded(features, [text])
        gate = bool(oos[0])
        intent_result = "OOS" if gate else dict(zip(bundle["known"], weights[0].tolist()))
        gate_ms = measure_ms(start)
        if not gate or policy == "matrix_no_gate":
            then = time.perf_counter()
            router = bundle["routers"]["matrix" if policy == "matrix_no_gate" else policy]
            chosen = router.choose(features if policy == "embedding" else weights)
            name = bundle["experts"][int(chosen[0])]
            route_ms = measure_ms(then)
    then = time.perf_counter()
    call = pool.execute(name, text)
    exec_ms = measure_ms(then)
    return {"executor": name, "pred_oos": gate, "intent_result": intent_result, "call": call, "recognition_ms": gate_ms,
            "route_ms": route_ms, "execution_ms": exec_ms, "total_ms": measure_ms(start)}


def scored(row, predicted, executor, known, base, error=False):
    complete, method, arguments = call_parts(predicted, row["gold"])
    return {"id": row["id"], "dialogue_id": row["dialogue_id"], "known": row["intent"] in known,
            "went_base": executor == base, "executor": executor, "failed_decision": error,
            "correct": complete, "method_correct": method, "arguments_correct": arguments}


def benchmark(bundle, pool, rows, batches, seed):
    """Warm single-client, round-robin interleaving; no offline response lookup."""
    if not rows:
        return {}, []
    samples = {name: [] for name in POLICIES}
    trace = []
    for name in POLICIES:
        infer_one(bundle, pool, name, rows[0]["text"])
    rng = np.random.default_rng(seed)
    for _ in range(batches):
        for row in rows:
            for raw_name in rng.permutation(POLICIES):
                name = str(raw_name)
                result = infer_one(bundle, pool, name, row["text"])
                measurements = {k: v for k, v in result.items() if k.endswith("_ms")}
                samples[name].append(measurements)
                trace.append(dict(id=row["id"], policy=name, executor=result["executor"], **measurements))
    timings = {}
    for name, entries in samples.items():
        timings[name] = {"n": len(entries)}
        for metric in entries[0]:
            values = [r[metric] for r in entries]
            timings[name][metric] = {"p50": float(np.median(values)), "p95": float(np.percentile(values, 95))}
        routed = [r["route_ms"] for r in entries if r["route_ms"] > 0]
        timings[name]["route_invocations"] = len(routed)
        timings[name]["route_mean_ms_if_invoked"] = rate(routed)
        timings[name]["sequential_requests_per_second"] = 1000 / np.mean([r["total_ms"] for r in entries])
    return timings, trace


def fit_support(encoder, split, router_rows, root, catalog):
    """Build support on fit; standardize on router; set the cutoff on known calibration."""
    from .support_gate import KNOWN_FRR, SupportGate
    from .support_selection import select_weights
    from .window_splits import _dialogue_labels

    fit, known = split["fit"], split["known"]
    fit_x = encoder.encode([r["text"] for r in fit])
    encoder.fit_centroids(fit_x, [r["intent"] for r in fit], known)
    router_x = encoder.encode([r["text"] for r in router_rows])
    active = _dialogue_labels(root, json.loads(catalog.read_bytes()))
    selection = select_weights(fit, fit_x, known, router_rows, router_x, dialogue_intents=active)
    weights = selection["weights"]
    gate = SupportGate(fit, fit_x, known, router_rows, router_x)
    known_cal = [r for r in split["cal"] if r["intent"] in known]
    cal_x = encoder.encode([r["text"] for r in known_cal])
    scores = encoder.calibrate_support(gate, weights, known_cal, cal_x)
    metadata = {"selected_candidate": selection["candidate"], "weights": weights,
                "cutoff": encoder.support_cutoff, "target_known_frr": KNOWN_FRR,
                "known_calibration_rows": len(scores),
                "known_calibration_rejected": int(np.sum(scores > encoder.support_cutoff))}
    return router_x, metadata, selection


def run(args):
    oos_gate = getattr(args, "oos_gate", "centroid")
    window = getattr(args, "recognition_window", "recent5")
    if oos_gate not in ("centroid", "adaptive-support"):
        raise ValueError("Unknown OOS gate")
    if oos_gate != "centroid" and (args.encoder, window) != ("minilm", "recent5"):
        raise ValueError("Support gates require --encoder minilm --recognition-window recent5")
    root = Path(args.sgd_root).resolve()
    catalog = Path(args.catalog).resolve()
    rows = load_sgd(root, catalog)
    split = split_rows(rows, seed=args.seed)
    known = list(split["known"])
    router_rows = sample_rows(split["router"], args.limit_router, args.seed)
    tests = sample_rows(split["test"], args.limit_test, args.seed)
    if not all([split["fit"], router_rows, split["cal"], tests, known]):
        raise ValueError("All partitions and the known class list must be nonempty")
    out = Path(args.out).resolve()
    work = completed_output(out)
    print(f"Run staging at {work}", flush=True)
    save_json(work / "split.json", {**{k: split[k] for k in ("known", "dev_unknown", "test_unknown")},
              **{k: [r["id"] for r in split[k]] for k in ("fit", "router", "cal", "test")},
              "router_used": [r["id"] for r in router_rows], "test_used": [r["id"] for r in tests]})
    config = {"backend": args.backend, "seed": args.seed,
              "base_path": str(Path(args.base_path).resolve()) if args.base_path else None,
              "adapters_dir": str(Path(args.adapters_dir).resolve()) if args.adapters_dir else None,
              "device": args.device, "max_length": args.max_length, "max_new_tokens": args.max_new_tokens}
    schema = public_schema(root)
    start = time.perf_counter()
    pool = make_pool(split["fit"], schema, config)
    names, base = list(pool.experts), pool.base
    encoder = Recognizer(encoder=args.encoder, model_path=args.model_path, window=window)
    support_metadata, features = {}, None
    gate_sources = ()
    if oos_gate == "centroid":
        encoder.fit(split["fit"], known).calibrate(split["cal"])
    else:
        features, support_metadata, selection = fit_support(encoder, split, router_rows, root, catalog)
        save_json(work / "support_selection.json", selection)
        gate_sources = ("support_gate", "support_selection", "window_splits", "near_gate_eval")
    encoder_setup_ms = measure_ms(start)
    start = time.perf_counter()
    _, success = all_calls(pool, router_rows, names)
    train_call_ms = measure_ms(start)
    save_json(work / "router_supervision.json",
              {"expert_order": names, "row_ids": [r["id"] for r in router_rows], "success": success.astype(int).tolist()})
    start = time.perf_counter()
    router_texts = [r["text"] for r in router_rows]
    if features is None:
        features = encoder.encode(router_texts)
    weights, _ = encoder.predict_encoded(features, router_texts)
    routers = {name: Router(name, seed=args.seed).fit(weights, [r["intent"] for r in router_rows], known, success)
               for name in ("fixed", "lookup", "matrix", "mlp")}
    routers["embedding"] = EmbeddingRouter(seed=args.seed).fit(features, success)
    fit_ms = measure_ms(start)
    bundle = dict(recognizer=encoder, routers=routers, experts=names, base=base, known=known)
    # Keep trained routing separate from the executor. Adapter weights live in their own directory.
    joblib.dump(bundle, work / "routing.joblib")
    joblib.dump({"fit": split["fit"], "schema": schema, "config": config}, work / "pool.joblib", compress=3)
    public = [{"id": r["id"], "text": r["text"]} for r in tests]
    write_rows(work / "public.jsonl", public)
    decisions = policy_decisions(bundle, public)
    write_rows(work / "decisions.jsonl", decisions)
    # Only after persisting selections do we join label and reference-call information.
    names_and_base = names + [base]
    outputs, all_success = all_calls(pool, tests, names_and_base)
    records = {p: [] for p in POLICIES}
    for row, decision in zip(tests, decisions, strict=True):
        for p in POLICIES:
            name = decision["selected"][p]
            records[p].append(scored(row, outputs[row["id"]][name], name, known, base))
    write_rows(work / "references.jsonl", [dict(id=r["id"], intent=r["intent"], gold=r["gold"],
               dialogue_id=r["dialogue_id"], calls=outputs[r["id"]]) for r in tests])
    quality = {p: summarize(values) for p, values in records.items()}
    for p, values in records.items():
        quality[p]["executor_counts"] = dict(Counter(v["executor"] for v in values))
    unknown = np.array([r["intent"] not in known for r in tests])
    gates = np.array([r["pred_oos"] for r in decisions])
    expert_oracle = all_success[:, :-1].max(axis=1)
    oracle = {"pool_of_experts": rate(expert_oracle), "experts_and_base": rate(all_success.max(axis=1)),
              "oracle_with_actual_gate": rate(np.where(gates, all_success[:, -1], expert_oracle)),
              "unknown_base_failed_some_expert_correct": rate((expert_oracle[unknown] > all_success[unknown, -1])),
              "oos_recall": rate(gates[unknown]), "known_false_oos": rate(gates[~unknown]),
              "train_unique_wins": dict(zip(names, ((success.sum(axis=1) == 1)[:, None] * success).sum(axis=0).astype(int).tolist()))}
    paired = {f"{p}_minus_{b}": paired_interval(records[p], records[b], args.seed)
              for p, b in [("mlp","lookup"),("matrix","lookup"),("mlp","matrix"),("mlp","embedding"),
                           ("matrix","matrix_no_gate"),("matrix","always_base")]}
    live_rows = sample_rows(tests, args.live_n, args.seed + 1) if args.live_n else []
    timings, trace = benchmark(bundle, pool, live_rows, args.repeats, args.seed)
    write_rows(work / "latency.jsonl", trace)
    manifest = root / "manifest.json"
    sources = gate_sources
    if args.backend == "retrieval-grounded":
        sources += ("transaction_executor",)
    summary = {"quality": quality, "diagnostics": oracle, "paired": paired, "warm_single_client": timings,
               "provenance": provenance(sources),
               "split_metadata": split["metadata"],
               "recognition": {"window": encoder.window, "oos_gate": oos_gate, "threshold": encoder.threshold,
                               "temperature": encoder.temperature, **support_metadata},
               "data_manifest": json.loads(manifest.read_text(encoding="utf8")) if manifest.is_file() else None,
               "setup_ms": encoder_setup_ms, "router_supervision_ms": train_call_ms, "router_fit_ms": fit_ms,
               "offline_router_executor_calls": len(router_rows)*len(names),
               "offline_test_executor_calls": len(tests)*len(names_and_base),
               "config": vars(args), "hardware": platform.platform(), "python": sys.version.split()[0],
               "catalog_sha256": sha256(catalog.read_bytes()).hexdigest(),
               "counts": {p:len(split[p]) for p in ("fit","router","cal","test")},
               "evaluated": len(tests), "expert_ids": names, "known": known}
    save_json(work / "summary.json", summary)
    work.rename(out)
    print(f"{out}: {len(tests)} requests, {len(names)} execution experts", flush=True)
    return summary


def load_run(directory):
    path = Path(directory)
    bundle = joblib.load(path / "routing.joblib")
    saved = joblib.load(path / "pool.joblib")
    return bundle, make_pool(saved["fit"], saved["schema"], saved["config"])
