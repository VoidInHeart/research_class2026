"""Profile a saved local policy with a burst of public requests.

python -m intentexec.profile --run-dir artifacts/pilot1118 --policy matrix \
    --limit 32 --concurrency 1 4 --repeats 2 --out artifacts/profile.json
"""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import sys
from time import perf_counter

import numpy as np

from .experiment import POLICIES, infer_one, load_run, read_rows, sample_rows


def distribution(values):
    return {name: float(np.percentile(values, percentile))
            for name, percentile in (("p50", 50), ("p95", 95))}


def timed_request(bundle, pool, policy, row, queued):
    entered = perf_counter()
    result = infer_one(bundle, pool, policy, row["text"])
    finished = perf_counter()
    return {"id": row["id"], "executor": result["executor"], "queue_ms": (entered - queued) * 1000,
            "service_ms": (finished - entered) * 1000, "end_to_end_ms": (finished - queued) * 1000,
            "recognition_ms": result["recognition_ms"], "route_ms": result["route_ms"],
            "execution_ms": result["execution_ms"], "returned_call": result["call"] is not None}


def warm_scenario(bundle, pool, policy, rows, concurrency, repeats, hourly_rate):
    records, occupied = [], 0.
    with ThreadPoolExecutor(max_workers=concurrency) as workers:
        for _ in range(repeats):
            # Arrival time is each submission. Requests wait in the client's
            # local queue if all workers are busy. Drain one burst before the next.
            begin = perf_counter()
            futures = [workers.submit(timed_request, bundle, pool, policy, row, perf_counter())
                       for row in rows]
            burst = [future.result() for future in futures]
            occupied += perf_counter() - begin
            records.extend(burst)
    metrics = ["queue_ms", "service_ms", "end_to_end_ms", "recognition_ms", "route_ms", "execution_ms"]
    output = {"workers": concurrency, "requests": len(records), "burst_size": len(rows),
              "repeats": repeats, "occupied_seconds": occupied,
              "requests_per_second": len(records) / occupied,
              "latency_ms": {key.removesuffix("_ms"): distribution([r[key] for r in records])
                             for key in metrics},
              "executor_counts": dict(Counter(r["executor"] for r in records)),
              "returned_calls": sum(r["returned_call"] for r in records)}
    routed = [r["route_ms"] for r in records if r["route_ms"] > 0]
    output["stage2_invocations"] = len(routed)
    output["stage2_if_invoked_ms"] = distribution(routed) if routed else None
    if hourly_rate is not None:
        cost = occupied * hourly_rate / 3600
        output["estimated_usd"] = cost
        output["estimated_usd_per_1000_requests"] = cost * 1000 / len(records)
    return output


def run_profile(args):
    if args.limit < 1 or args.repeats < 1:
        raise ValueError("limit and repeats must be positive")
    if not args.concurrency or min(args.concurrency) < 1 or len(set(args.concurrency)) != len(args.concurrency):
        raise ValueError("concurrency values must be distinct positive integers")
    if args.usd_hour is not None and (not math.isfinite(args.usd_hour) or args.usd_hour < 0):
        raise ValueError("usd-hour must be nonnegative and finite")
    if args.out.exists():
        raise FileExistsError(f"Output already exists: {args.out}")
    rows = sample_rows(read_rows(args.run_dir / "public.jsonl"), args.limit, args.seed)
    if not rows:
        raise ValueError("public.jsonl has no requests")
    start = perf_counter()
    bundle, pool = load_run(args.run_dir)
    loaded = perf_counter()
    # Includes lazy model/tokenizer initialization on the first actual request.
    first = infer_one(bundle, pool, args.policy, rows[0]["text"])
    finished = perf_counter()
    startup = {"load_ms": (loaded - start) * 1000, "first_request_ms": (finished - loaded) * 1000,
               "load_plus_first_request_ms": (finished - start) * 1000, "first_id": rows[0]["id"],
               "first_executor": first["executor"], "recognition_ms": first["recognition_ms"],
               "route_ms": first["route_ms"], "execution_ms": first["execution_ms"]}
    if args.usd_hour is not None:
        startup["estimated_usd"] = (finished - start) * args.usd_hour / 3600
    output = {"run_dir": str(args.run_dir.resolve()), "policy": args.policy,
              "recognizer": bundle["recognizer"].encoder, "executor_backend": type(pool).__name__,
              "host": platform.platform(), "python": sys.version.split()[0], "seed": args.seed,
              "usd_per_instance_hour_assumption": args.usd_hour,
              "request_ids_sha256": hashlib.sha256(json.dumps([r["id"] for r in rows],
                                                    ensure_ascii=False).encode()).hexdigest(),
              "startup": startup, "scenarios": [], "measurement": {
                  "startup": "New in-process bundle + pool load and one serial real inference. OS cache may be warm.",
                  "load": "Includes rebuilding the saved retrieval indices if that backend is selected.",
                  "load_pattern": "One burst per repetition; all scenarios replay the same texts in the same order.",
                  "queue": "Local worker-queue wait measured from per-request submission to worker entry.",
                  "end_to_end": "Queue wait plus synchronous recognition, routing and selected execution.",
                  "execution": "LoRA adapter selection and generation serialize under the shared model lock; lock wait is inside execution_ms.",
                  "cost": "Assumed whole-instance hourly rate times occupied wall time, divided by 3600. Warm costs exclude cold load, training, and idle time."}}
    for concurrency in args.concurrency:
        scenario = warm_scenario(bundle, pool, args.policy, rows, concurrency, args.repeats, args.usd_hour)
        output["scenarios"].append(scenario)
    loaded_torch = sys.modules.get("torch")
    output["threads"] = {"logical_cpu_count": os.cpu_count(), "OMP_NUM_THREADS": os.getenv("OMP_NUM_THREADS"),
                         "MKL_NUM_THREADS": os.getenv("MKL_NUM_THREADS"),
                         "torch_intraop_threads": loaded_torch.get_num_threads() if loaded_torch else None}
    text = json.dumps(output, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    args.out.parent.mkdir(exist_ok=True, parents=True)
    with args.out.open("x", encoding="utf8") as handle:
        handle.write(text)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--policy", choices=POLICIES, default="matrix")
    parser.add_argument("--limit", type=int, default=32)
    parser.add_argument("--concurrency", nargs="+", type=int, default=[1, 4])
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--seed", type=int, default=1118)
    parser.add_argument("--usd-hour", type=float, help="Assumed price of the whole instance in USD/hour")
    parser.add_argument("--out", type=Path, required=True, help="Fresh JSON file")
    args = parser.parse_args()
    result = run_profile(args)
    for scenario in result["scenarios"]:
        print(f"{scenario['workers']} workers: {scenario['requests']} requests, "
              f"{scenario['requests_per_second']:.1f} req/s, "
              f"end-to-end p95 {scenario['latency_ms']['end_to_end']['p95']:.2f} ms")
    print(args.out)


if __name__ == "__main__":
    main()

