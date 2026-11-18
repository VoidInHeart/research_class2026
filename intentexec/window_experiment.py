"""Offline paired comparison of three recognition windows with one frozen retrieval pool."""
import argparse
import gzip
import json
from pathlib import Path

from .data import load_sgd
from .executors import EXPERTS, BASE
from .experiment import completed_output, save_json
from .routing import Router
from .window_frozen import digest, frozen_outcomes
from .window_metrics import paired, summarize
from .window_scoring import WindowFeatures, calibrate, weights
from .window_splits import build_folds

# First element supplies the OOS gate, second the known-intent vector.
WINDOWS = {"original": ("long", "long"), "recent5": ("short", "short"), "dual": ("long", "short")}
CONTRASTS = (("original", "recent5"), ("original", "dual"), ("recent5", "dual"))


def collect_predictions(rows, known, table, routed, probabilities, gate_scores, threshold):
    predictions = []
    names = (*EXPERTS, BASE)
    top = probabilities.argmax(axis=1)
    maximum = gate_scores.max(axis=1)
    for i, row in enumerate(rows):
        reject = bool(maximum[i] < threshold)
        chosen = len(EXPERTS) if reject else int(routed[i])
        outcomes = table[row["id"]]
        exact, method = outcomes[chosen]
        predictions.append({"id": row["id"], "dialogue_id": row["dialogue_id"], "role": row["role"],
                            "intent": row["intent"], "top1": known[top[i]], "gate_oos": reject,
                            "oos_score": -float(maximum[i]), "selected": names[chosen],
                            "correct": exact, "method_correct": method, "base_correct": outcomes[-1][0],
                            "expert_best_correct": any(c[0] for c in outcomes[:-1])})
    return predictions


def evaluate_fold(split, features, table, emit):
    known = split["known"]
    phases = {p: split[p] for p in ("router", "cal", "test")}
    scores, thresholds, temperatures = {}, {}, {}
    for view in ("long", "short"):
        scores[view] = features.score(view, split["fit"], phases, known)
        thresholds[view], temperatures[view] = calibrate(scores[view]["cal"], split["cal"], known)
    # The ability matrix depends only on the true training intents and frozen execution success.
    supervision = [[entry[0] for entry in table[r["id"]][:-1]] for r in split["router"]]
    matrix = Router("matrix").fit(weights(scores["long"]["router"], temperatures["long"]),
                                  [r["intent"] for r in split["router"]], known, supervision)
    results, predictions = {}, {}
    for variant, (gate_view, intent_view) in WINDOWS.items():
        probabilities = weights(scores[intent_view]["test"], temperatures[intent_view])
        chosen = matrix.choose(probabilities)
        rows = collect_predictions(split["test"], known, table, chosen, probabilities,
                                   scores[gate_view]["test"], thresholds[gate_view])
        predictions[variant] = rows
        results[variant] = summarize(rows, known)
        for row in rows:
            emit(variant, row)
    paired_results = {f"{b}_minus_{a}": paired(predictions[a], predictions[b], known)
                      for a, b in CONTRASTS}
    return {"variants": results, "paired": paired_results, "thresholds": thresholds,
            "temperatures": temperatures, "ability": matrix.ability.tolist()}


def run(args):
    from importlib.metadata import version
    if "minilm" in args.encoders:
        import torch
        torch.set_num_threads(args.threads)
    out = Path(args.out)
    work = completed_output(out)
    all_rows = load_sgd(args.data, args.catalog)
    folds = build_folds(args.data, args.catalog, args.frozen_run, args.config)
    table, frozen = frozen_outcomes(all_rows, args.data, args.frozen_run, Path(args.cache) / "execution")
    fold_manifest = {}
    for fold_name, split in folds.items():
        fold_manifest[fold_name] = {key: split[key] for key in ("known", "dev_unknown", "test_unknown", "metadata")}
        fold_manifest[fold_name]["ids"] = {p: [r["id"] for r in split[p]] for p in ("fit", "router", "cal", "test")}
        print(f"{fold_name}: " + ", ".join(f"{p}={len(split[p])}" for p in ("fit","router","cal","test")), flush=True)
    save_json(work / "folds.json", fold_manifest)
    report = {"protocol": {"windows_gate_and_intent": WINDOWS, "second_layer": "matrix",
                           "bootstrap_unit": "dialogue", "bootstrap_draws": 2000, "seed": 1118,
                           "holdout_config_sha256": digest(Path(args.config)),
                           "data_manifest_sha256": digest(Path(args.data) / "manifest.json"),
                           "catalog_sha256": digest(Path(args.catalog)), "frozen": frozen,
                           "versions": {p: version(p) for p in ("numpy","scipy","scikit-learn","joblib")}},
              "encoders": {}}
    pkg = Path(__file__).parent
    report["protocol"]["source_sha256"] = {
        name: digest(pkg / name) for name in ("data.py", "executors.py", "routing.py", "metrics.py",
            "window_splits.py", "window_scoring.py", "window_frozen.py", "window_metrics.py", "window_experiment.py")}
    fold_names = list(folds)
    tests = [set(fold_manifest[f]["ids"]["test"]) for f in fold_names]
    report["test_overlap"] = {f: {g: len(tests[i] & tests[j]) for j, g in enumerate(fold_names)}
                              for i, f in enumerate(fold_names)}
    with (work / "predictions.jsonl.gz").open("wb") as raw:
        with gzip.GzipFile(filename="", fileobj=raw, mode="wb", mtime=0) as trace:
            for encoder in args.encoders:
                features = WindowFeatures(all_rows, encoder, args.model_path, Path(args.cache) / "features")
                if encoder == "minilm":
                    report["protocol"]["minilm_fingerprint"] = features.model_fingerprint
                    report["protocol"]["versions"].update({p: version(p) for p in ("torch","transformers","tokenizers")})
                results = {}
                for fold_name, split in folds.items():
                    def emit(variant, row):
                        line = {"fold": fold_name, "encoder": encoder, "window": variant, **row}
                        trace.write((json.dumps(line, ensure_ascii=False, separators=(",", ":")) + "\n").encode())
                    results[fold_name] = evaluate_fold(split, features, table, emit)
                    deltas = results[fold_name]["paired"]["recent5_minus_original"]
                    print(encoder, fold_name, "recent5-original",
                          {k: round(100 * deltas[k]["difference"], 2)
                           for k in ("known_top1", "gate_balanced_accuracy", "exact")}, flush=True)
                report["encoders"][encoder] = results
    report["predictions_sha256"] = digest(work / "predictions.jsonl.gz")
    report["folds_sha256"] = digest(work / "folds.json")
    save_json(work / "results.json", report)
    work.rename(out)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/sgd")
    parser.add_argument("--catalog", default="configs/intent_catalog.json")
    parser.add_argument("--config", default="configs/window_holdouts.json")
    parser.add_argument("--frozen-run", default="artifacts/verified-minilm",
                        help="Prior trusted local retrieval run containing its saved pool and split")
    parser.add_argument("--model-path", default="../code/models/models--sentence-transformers--all-MiniLM-L6-v2")
    parser.add_argument("--encoders", choices=("tfidf", "minilm"), nargs="+", default=["tfidf", "minilm"])
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--cache", default="artifacts/window-study-cache")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    if args.threads < 1 or len(args.encoders) != len(set(args.encoders)):
        parser.error("Threads must be positive and encoders unique")
    run(args)


if __name__ == "__main__":
    main()
