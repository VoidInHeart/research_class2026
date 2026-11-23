"""Post-hoc ceilings for coarse intent routing, with OOS decisions held fixed."""
import argparse
from collections import Counter, defaultdict
import gzip
import json
from pathlib import Path

from .executors import EXPERTS, BASE
from .window_frozen import digest


def opportunity(rows):
    """Retrospective accuracy if each first-ranked intent always chose its best executor."""
    names = (*EXPERTS, BASE)
    by_intent = defaultdict(Counter)
    overall = Counter()
    rejected_correct = eligible = base_only = specialist_only = oracle = 0
    for row in rows:
        successes = dict(zip(names, row["exact"], strict=True))
        if row["rejected"]:
            rejected_correct += successes[BASE]
            continue
        eligible += 1
        by_intent[row["top1"]].update(successes)
        overall.update(successes)
        best_specialist = max(successes[name] for name in EXPERTS)
        base_only += int(successes[BASE] and not best_specialist)
        specialist_only += int(best_specialist and not successes[BASE])
        oracle += max(successes.values())
    return {"n": len(rows), "accepted": eligible, "rejected_correct": rejected_correct,
            "always_base_exact": rejected_correct + overall[BASE],
            "best_fixed_on_accepted_exact": rejected_correct + max(overall.values(), default=0),
            "best_per_predicted_top1_on_accepted_exact": rejected_correct + sum(max(c.values()) for c in by_intent.values()),
            "best_per_request_on_accepted_exact": rejected_correct + oracle,
            "only_base_correct_among_accepted": base_only,
            "some_specialist_correct_and_base_wrong_among_accepted": specialist_only}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--records", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--development", type=Path, default=Path("reports/refreshed_routing/development.json"))
    args = p.parse_args()
    groups = defaultdict(list)
    folds = json.loads(args.development.read_bytes())["folds"]
    with gzip.open(args.records, "rt", encoding="utf8") as stream:
        for row in map(json.loads, stream):
            if "top1" not in row:
                known = folds[row["fold"]]["known"]
                scored = list(zip(known, row["weights"], strict=True))
                row["top1"] = max(scored, key=lambda pair: pair[1])[0]
            groups[row["fold"]].append(row)
    result = {"records_sha256": digest(args.records), "folds": {}}
    for name, rows in groups.items():
        stats = {"all": opportunity(rows), "unknown": opportunity([r for r in rows if r["unknown"]])}
        if name == "near_domain_bus_ticket":
            for label in ("FindBus", "BuyBusTicket"):
                group = [r for r in rows if r["top1"] == "FindBus" and r["intent"] == label]
                stats["top1_FindBus_actual_" + label] = opportunity(group)
        result["folds"][name] = stats
    with args.out.open("x", encoding="utf8") as out:
        json.dump(result, out, indent=2, allow_nan=False)
        out.write("\n")


if __name__ == "__main__":
    main()
