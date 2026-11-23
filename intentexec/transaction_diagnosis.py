"""Explain the original frozen retrieval calls on the three near-domain unknown cohorts."""
import argparse
from collections import Counter
import gzip
import json
from pathlib import Path

import joblib

from .data import load_sgd, public_schema
from .executors import BASE, EXPERTS, RetrievalPool
from .metrics import call_parts, canonical_value
from .window_frozen import digest


FOLDS = {
    "near_domain_bus_ticket": "BuyBusTicket",
    "near_lodging_booking": "ReserveLodging",
    "near_event_booking": "BuyEventTickets",
}


def slot_errors(predicted, expected):
    actual = canonical_value(predicted)
    target = canonical_value(expected)
    return {
        "missing": sorted(target.keys() - actual.keys()),
        "extra": sorted(actual.keys() - target.keys()),
        "wrong_value": sorted(k for k in actual.keys() & target.keys() if actual[k] != target[k]),
    }


def tool(call):
    return f"{call['service']}.{call['method']}" if call else "(no call)"


def sibling(actual, gold):
    return (actual is not None and actual["service"] != gold["service"] and
            actual["service"].rsplit("_", 1)[0] == gold["service"].rsplit("_", 1)[0])


def summarize_path(samples, path):
    first_error = Counter()
    slot_request_counts = Counter()
    slot_counts = {kind: Counter() for kind in ("missing", "extra", "wrong_value")}
    service_pairs, tool_pairs = Counter(), Counter()
    service_ok = tool_ok = raw_method_ok = args_ok = sibling_count = 0
    for s in samples:
        call, gold = s["calls"][path], s["gold"]
        exact, correct_tool, correct_args = call_parts(call, gold)
        service = call is not None and call["service"] == gold["service"]
        service_ok += service
        tool_ok += correct_tool
        raw_method_ok += call is not None and call["method"] == gold["method"]
        args_ok += correct_args
        sibling_count += sibling(call, gold)
        if not service:
            first_error["service"] += 1
            wrong_service = call["service"] if call else "(no call)"
            service_pairs[gold["service"], wrong_service] += 1
        elif not correct_tool:
            first_error["method_given_correct_service"] += 1
            tool_pairs[tool(gold), tool(call)] += 1
        else:
            first_error["exact" if exact else "arguments_given_correct_tool"] += 1
            for kind, keys in slot_errors(call["arguments"], gold["arguments"]).items():
                slot_request_counts[kind] += bool(keys)
                slot_counts[kind].update(keys)
    return {
        "n": len(samples), "service_correct": service_ok, "raw_method_name_correct": raw_method_ok,
        "service_and_method_correct": tool_ok, "argument_dict_correct_regardless_of_tool": args_ok,
        "first_error": {k: first_error[k] for k in ("service", "method_given_correct_service",
                                                    "arguments_given_correct_tool", "exact")},
        "wrong_service_same_domain": sibling_count,
        "wrong_service_pairs": [{"expected": a, "predicted": b, "n": n}
                                for (a, b), n in service_pairs.most_common()],
        "wrong_method_pairs_given_correct_service": [{"expected": a, "predicted": b, "n": n}
                                                    for (a, b), n in tool_pairs.most_common()],
        "given_correct_tool": {
            "n": tool_ok,
            "requests_with_slot_error": dict(sorted(slot_request_counts.items())),
            "error_instances_by_slot": {k: dict(v.most_common()) for k, v in slot_counts.items()},
        },
    }


def examples(samples):
    """Pick short illustrative dialogues with a distinct base failure, in fixed category order."""
    selected, seen = [], set()
    for category in ("arguments_given_correct_tool", "method_given_correct_service", "sibling_service"):
        choices = []
        for s in samples:
            call, gold = s["calls"]["base"], s["gold"]
            if call is None or s["dialogue_id"] in seen:
                continue
            exact, method, _ = call_parts(call, gold)
            if category == "arguments_given_correct_tool":
                match = method and not exact
            elif category == "method_given_correct_service":
                match = call["service"] == gold["service"] and not method
            else:
                match = sibling(call, gold)
            if match:
                choices.append(s)
        if choices:
            case = min(choices, key=lambda r: (len(r["public_prefix"]), r["id"]))
            seen.add(case["dialogue_id"])
            base = case["calls"]["base"]
            selected.append({"base_first_error": category,
                             **{k: case[k] for k in ("id", "dialogue_id", "public_prefix", "expert", "gold")},
                             "selected_expert_call": case["calls"]["expert"], "base_call": base,
                             "base_slot_diff_if_tool_correct": slot_errors(base["arguments"], case["gold"]["arguments"])
                             if category == "arguments_given_correct_tool" else None})
    return selected


def diagnose(archive, data, catalog, pool_file):
    report = json.loads((archive / "results.json").read_bytes())
    lock_file = archive / "development_lock.json"
    lock = json.loads(lock_file.read_bytes())
    scores = archive / "predictions.jsonl.gz"
    if (digest(scores) != report["prediction_sha256"] or digest(lock_file) != report["lock_sha256"] or
            digest(pool_file) != lock["expert_pool_sha256"] or
            digest(data / "manifest.json") != report["fresh_manifest_sha256"]):
        raise ValueError("Frozen pool, corpus or support evaluation digest mismatch")
    saved = joblib.load(pool_file)
    if saved["config"]["backend"] != "retrieval" or saved["schema"] != public_schema(data):
        raise ValueError("Expected the original local retrieval pool and public schema")
    pool = RetrievalPool(saved["fit"], saved["schema"], saved["config"]["seed"])
    data_rows = {r["id"]: r for r in load_sgd(data, catalog)}
    samples = {name: [] for name in FOLDS}
    with gzip.open(scores, "rt", encoding="utf8") as stream:
        for r in map(json.loads, stream):
            fold = r["fold"]
            if fold not in FOLDS or not r["unknown"]:
                continue
            x = data_rows[r["id"]]
            if (x["intent"] != FOLDS[fold] or x["intent"] != r["intent"] or
                    x["dialogue_id"] != r["dialogue_id"]):
                raise ValueError(f"Frozen request misaligned at {r['id']}")
            calls = {name: pool.execute(name, x["text"]) for name in (*EXPERTS, BASE)}
            calls["expert"] = calls[r["expert"]]
            for path in ("expert", "base"):
                if call_parts(calls[path], x["gold"])[:2] != (r[path + "_exact"], r[path + "_method"]):
                    raise ValueError(f"Original {path} output disagrees with frozen metrics at {r['id']}")
            # Diagnostic upper bound for this fixed span extractor. Supply the true
            # tool and set of argument names only here, after all public-input calls.
            gold = x["gold"]
            calls["gold_tool_gold_keys"] = {**gold, "arguments": pool.lexicon.extract(gold["arguments"], x["text"])}
            samples[fold].append({"id": x["id"], "dialogue_id": x["dialogue_id"], "public_prefix": x["text"],
                                  "expert": r["expert"], "gold": gold, "calls": calls})
    result = {"pool_sha256": lock["expert_pool_sha256"], "prediction_sha256": report["prediction_sha256"],
              "fresh_manifest_sha256": report["fresh_manifest_sha256"], "fit_rows": len(saved["fit"]), "folds": {}}
    for fold, group in samples.items():
        count = len({s["id"] for s in group})
        if count != len(group) or count != report["folds"][fold]["metrics"]["nu"]:
            raise ValueError(f"Unexpected unknown cohort for {fold}")
        label = FOLDS[fold]
        tools_in_fit = Counter(tool(r["gold"]) for r in saved["fit"] if r["intent"] == label)
        any_exact = sum(any(call_parts(s["calls"][name], s["gold"])[0] for name in (*EXPERTS, BASE))
                        for s in group)
        result["folds"][fold] = {
            "intent": label, "requests": count, "dialogues": len({s["dialogue_id"] for s in group}),
            "target_tools_in_fit": dict(tools_in_fit.most_common()),
            "target_tools_in_cohort": dict(Counter(tool(s["gold"]) for s in group).most_common()),
            "selected_expert_frequency": dict(Counter(s["expert"] for s in group).most_common()),
            "selected_expert": summarize_path(group, "expert"), "base": summarize_path(group, BASE),
            "gold_tool_gold_keys_with_original_span_extractor": summarize_path(group, "gold_tool_gold_keys"),
            "any_of_all_four_exact": any_exact,
            "all_four_exact_by_executor": {
                name: sum(call_parts(s["calls"][name], s["gold"])[0] for s in group)
                for name in (*EXPERTS, BASE)
            },
            "examples": examples(group),
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=Path("reports/support_validation"))
    parser.add_argument("--data", type=Path, default=Path("data/sgd-support-holdout"))
    parser.add_argument("--catalog", type=Path, default=Path("configs/intent_catalog.json"))
    parser.add_argument("--pool", type=Path, default=Path("artifacts/verified-minilm/pool.joblib"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = diagnose(args.archive, args.data, args.catalog, args.pool)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode()
    with args.out.open("xb") as handle:
        handle.write(encoded)
    print(f"Wrote {sum(v['requests'] for v in result['folds'].values())} reproduced unknown requests to {args.out}")


if __name__ == "__main__":
    main()
