"""Task quality is measured on a complete call, including its arguments."""
import unicodedata
import numpy as np


def canonical_value(value):
    if isinstance(value, str):
        return " ".join(unicodedata.normalize("NFKC", value).casefold().split())
    if isinstance(value, list):
        return [canonical_value(item) for item in value]
    if isinstance(value, dict):
        return {key: canonical_value(val) for key, val in value.items()}
    return value


def call_parts(predicted, expected):
    if not isinstance(predicted, dict) or set(predicted) != {"service", "method", "arguments"}:
        return False, False, False
    args = predicted["arguments"]
    if not isinstance(args, dict):
        return False, False, False
    method = (predicted["service"], predicted["method"]) == (expected["service"], expected["method"])
    arguments = canonical_value(args) == canonical_value(expected["arguments"])
    return bool(method and arguments), bool(method), bool(arguments)


def rate(values):
    return float(np.mean(values)) if len(values) else None


def summarize(decisions):
    """Each item is an evaluated request from one policy."""
    result = {"n": len(decisions)}
    for metric in ("correct", "method_correct", "arguments_correct", "went_base", "failed_decision"):
        result[metric] = rate([d[metric] for d in decisions])
    result["known_accuracy"] = rate([d["correct"] for d in decisions if d["known"]])
    result["unknown_accuracy"] = rate([d["correct"] for d in decisions if not d["known"]])
    for known, group in [(True, "known"), (False, "unknown")]:
        for fallback, path in [(True, "base"), (False, "expert")]:
            sub = [d for d in decisions if d["known"] == known and d["went_base"] == fallback
                   and not d["failed_decision"]]
            result[f"{group}_{path}"] = {"n": len(sub), "accuracy": rate([d["correct"] for d in sub])}
    return result


def paired_interval(a, b, seed=1118, repeats=1000):
    """Dialogue-cluster paired bootstrap, conditional on one split and its fitted experts."""
    if not a or len(a) != len(b):
        raise ValueError("paired non-empty observations required")
    groups = {}
    for left, right in zip(a, b):
        if (left["id"], left["dialogue_id"]) != (right["id"], right["dialogue_id"]):
            raise ValueError("unpaired requests")
        g = groups.setdefault(left["dialogue_id"], [0, 0])
        g[0] += int(left["correct"]) - int(right["correct"])
        g[1] += 1
    group = np.array(list(groups.values()))
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(repeats):
        sums = group[rng.integers(len(group), size=len(group))].sum(axis=0)
        diffs.append(sums[0] / sums[1])
    return {"difference": float(group[:, 0].sum()/group[:, 1].sum()),
            "ci95": list(np.quantile(diffs, [.025, .975])), "dialogues": len(group)}
