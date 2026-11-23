"""Live paired comparison of hosted Jev vs local routing, on a frozen cohort."""
import json
import os
from pathlib import Path
import time
from urllib.error import HTTPError, URLError

import numpy as np

from .experiment import (completed_output, infer_one, load_run, measure_ms, read_rows,
                         sample_rows, save_json, scored)
from .jev import Jev, vector_state
from .metrics import paired_interval, summarize


DESCRIPTIONS = {
    "search": "Specialized in finding or querying entities across services.",
    "transaction": "Specialized in bookings, purchases, transfers and other actions.",
    "context": "Specialized in calls requiring information across dialogue turns.",
    "base": "General-purpose executor; use for requests outside the known intent inventory."
}


def criteria(bundle, direct):
    expert_ids = bundle["experts"]
    ability = bundle["routers"]["matrix"].ability
    result = {}
    for j, name in enumerate(expert_ids):
        profile = {label: round(float(ability[i, j]), 4) for i, label in enumerate(bundle["known"])}
        result[name] = DESCRIPTIONS[name] + " Development call success by intent: " + json.dumps(profile)
    if direct:
        result[bundle["base"]] = DESCRIPTIONS["base"]
    return result


def jev_step(bundle, pool, client, options, direct, text):
    start = time.perf_counter()
    recognition_ms, decision_ms, execution_ms = 0., 0., 0.
    executor = bundle["base"]
    send = direct
    gate = None
    state = text
    if not direct:
        weights, oos = bundle["recognizer"].predict([text])
        gate = bool(oos[0])
        send = not gate
        if send:
            state = vector_state(bundle["known"], weights[0])
        recognition_ms = measure_ms(start)
    error = None
    call = None
    if send:
        then = time.perf_counter()
        try:
            executor = client.choose(state, options)
        except (HTTPError, URLError, OSError, ValueError, TypeError, AttributeError) as exc:
            executor = None
            error = f"{type(exc).__name__}" + (f" HTTP {exc.code}" if isinstance(exc, HTTPError) else "")
        decision_ms = measure_ms(then)
    if executor is not None:
        then = time.perf_counter()
        call = pool.execute(executor, text)
        execution_ms = measure_ms(then)
    return dict(executor=executor, call=call, failed_decision=bool(error), error=error, pred_oos=gate,
                usage=client.last_usage if send else {}, resolved_model=client.resolved_model if send else None,
                api_called=send, recognition_ms=recognition_ms, route_ms=decision_ms,
                execution_ms=execution_ms, total_ms=measure_ms(start))


def costs(trace, instance_rate, input_rate, output_rate, per_k):
    seconds = sum(row["total_ms"] for row in trace)/1000
    local = None if instance_rate is None else instance_rate * seconds/3600
    sent = [r for r in trace if r.get("api_called")]
    api = None
    if not sent:
        api = 0.
    elif (input_rate is not None and output_rate is not None or per_k is not None):
        # A missing usage record or a failed response makes its actual charge unknown.
        if not any(r.get("failed_decision") for r in sent):
            if input_rate is None and output_rate is None:
                api = len(sent)*per_k/1000
            elif all(all(isinstance(r["usage"].get(k), (int, float))
                         for k in ("input_tokens", "output_tokens")) for r in sent):
                api = sum((r["usage"]["input_tokens"]*input_rate+r["usage"]["output_tokens"]*output_rate)
                          /1_000_000 for r in sent) + len(sent)*(per_k or 0)/1000
    total = None if local is None or api is None else local+api
    successes = sum(r["correct"] for r in trace)
    return {"estimated_api_usd": api, "estimated_local_instance_usd": local,
            "estimated_total_usd_per_1000": None if total is None else 1000*total/len(trace),
            "estimated_usd_per_correct": total/successes if total is not None and successes else None}


def compare(args, client=None):
    path = Path(args.run_dir)
    summary_path = path / "summary.json"
    if summary_path.is_file():
        supported = json.loads(summary_path.read_bytes()).get("supported_policies")
        if supported is not None and not {"matrix", "lookup", "mlp"}.issubset(supported):
            raise ValueError("Hosted comparison needs a full experiment with lookup, mlp and reference calls")
    if client is None and not os.environ.get("TYPESAFE_API_KEY"):
        raise ValueError("TYPESAFE_API_KEY is required for the hosted comparison.")
    out = Path(args.out).resolve()
    if (args.input_usd_million is None) != (args.output_usd_million is None):
        raise ValueError("Supply both token prices or leave both unspecified")
    client = client or Jev(args.model, args.timeout)
    bundle, pool = load_run(path)
    cohort = sample_rows(read_rows(path/"public.jsonl"), args.limit, 1118)
    work = completed_output(out)
    profiles = {p: criteria(bundle, p == "jev_direct") for p in ("jev_direct", "jev_vector")}
    save_json(work/"jev_options.json", profiles)
    methods = ("matrix", "lookup", "mlp", "jev_direct", "jev_vector")
    rng = np.random.default_rng(1118)
    trace = []
    with (work / "live.jsonl").open("w", encoding="utf8") as log:
        for row in cohort:
            for p in rng.permutation(methods).tolist():
                if p in profiles:
                    item = jev_step(bundle, pool, client, profiles[p], p == "jev_direct", row["text"])
                else:
                    item = infer_one(bundle, pool, p, row["text"])
                item.update(id=row["id"], policy=p)
                trace.append(item)
                log.write(json.dumps(item, ensure_ascii=False) + "\n")
                log.flush()
    refs = {r["id"]:r for r in read_rows(path/"references.jsonl")}
    by_method = {}
    for item in trace:
        row = refs[item["id"]]
        outcome = scored(row, item["call"], item["executor"], bundle["known"],
                         bundle["base"], item.get("failed_decision",False))
        by_method.setdefault(item["policy"], []).append(dict(item, **outcome))
    results = {}
    for p, records in by_method.items():
        results[p] = {"quality":summarize(records), "requests_to_jev":sum(r.get("api_called", False) for r in records),
                      "cost":costs(records, args.local_usd_hour, args.input_usd_million, args.output_usd_million,
                                   args.usd_per_1000_decisions),
                      "latency":{k: dict(zip(("p50","p95"),np.percentile([r[k] for r in records], [50,95]).tolist()))
                                 for k in ("recognition_ms","route_ms","execution_ms","total_ms")}}
    differences = {p: paired_interval(by_method["matrix"], by_method[p]) for p in ("jev_direct","jev_vector")}
    save_json(work/"comparison.json", {"config":vars(args),"results":results,"matrix_minus":differences,
              "ids":[r["id"] for r in cohort], "timing":"single client, interleaved, network included, no retries"})
    work.rename(out)
    print(f"Compared {len(cohort)} matched requests at {out}")
