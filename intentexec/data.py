"""Build paired intent/execution examples from the public SGD training release."""
import hashlib
import json
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path


def _json(path):
    return json.loads(Path(path).read_bytes())


def public_schema(root):
    """All public tools in the train schema, shared verbatim by every executor."""
    return _json(Path(root) / "train" / "schema.json")


def _index(catalog):
    by_method = {}
    for label, methods in catalog["classes"].items():
        for method in methods:
            if method in by_method:
                raise ValueError(f"Method assigned twice: {method}")
            by_method[method] = label
    dev, test = set(catalog["dev_unknown"]), set(catalog["test_unknown"])
    if dev & test or (dev | test) - catalog["classes"].keys():
        raise ValueError("Holdout intentions must be disjoint, declared semantic intentions")
    role = {k: ("test_unknown" if k in test else "dev_unknown" if k in dev else "known")
            for k in catalog["classes"]}
    return by_method, role


def _extract(dialogue, method_index, role):
    turns = dialogue["turns"]
    did = dialogue["dialogue_id"]
    # A later mention of an unseen intent can quarantine the whole dialogue,
    # even when that mention has no matching executable next turn.
    dialogue_roles = sorted({role[method_index[frame["state"]["active_intent"]]]
                             for turn in turns if turn["speaker"] == "USER"
                             for frame in turn["frames"] if frame["state"]["active_intent"] != "NONE"})
    prefix = []
    for i, turn in enumerate(turns[:-1]):
        prefix.append(f"{turn['speaker']}: {turn['utterance']}")
        following = turns[i + 1]
        if (turn["speaker"], following["speaker"]) != ("USER", "SYSTEM"):
            continue
        calls = [(f["service"], f["service_call"]) for f in following["frames"] if "service_call" in f]
        if len(calls) != 1:
            continue
        service, call = calls[0]
        active = [f["state"]["active_intent"] for f in turn["frames"] if f["service"] == service]
        if active != [call["method"]] or active == ["NONE"]:
            continue
        label = method_index[call["method"]]
        yield {"id": f"{did}:{i}", "dialogue_id": did, "text": "\n".join(prefix),
               "intent": label, "role": role[label], "dialogue_roles": dialogue_roles,
               "gold": {"service": service, "method": call["method"], "arguments": call["parameters"]}}


def load_sgd(root, catalog_path):
    root = Path(root)
    train = root / "train"
    method_index, role = _index(_json(catalog_path))
    shards = sorted(train.glob("dialogues_*.json"))
    if not shards:
        raise FileNotFoundError(f"No SGD train shards under {train}. Run scripts/fetch_sgd.py first.")
    manifest = root / "manifest.json"
    if manifest.exists():
        digests = _json(manifest)["sha256"]
        if {p.name for p in shards} | {"schema.json"} != set(digests):
            raise ValueError("Training files differ from the source manifest")
        for name, digest in digests.items():
            if hashlib.sha256((train / name).read_bytes()).hexdigest() != digest:
                raise ValueError(f"Source file differs from its manifest: {name}")
    rows = []
    seen = set()
    for path in shards:
        for dialogue in _json(path):
            did = dialogue["dialogue_id"]
            if did in seen:
                raise ValueError(f"Duplicate dialogue ID: {did}")
            seen.add(did)
            rows.extend(_extract(dialogue, method_index, role))
    return rows


def _template(text):
    """Exact normalization plus numbers (dates/times/prices) and punctuation.

    This identifies literal/number-substituted duplicate prefixes, not arbitrary
    paraphrases. Comparing whole prefixes keeps common short answers in context.
    """
    text = unicodedata.normalize("NFKC", text).casefold()
    text = re.sub(r"\d+", " NUMBER ", text)
    return " ".join(re.findall(r"\w+", text))


def split_rows(rows, seed=1118):
    """Group entire dialogues and duplicate input templates; no eval label enters fit.

    Mixed dev/test components are excluded. Components with one held-out role
    go wholly to its evaluation partition (including their known examples).
    Remaining components are partitioned by seeded SHA-256 into 60/20/10/10.
    """
    parent = {r["dialogue_id"]: r["dialogue_id"] for r in rows}

    def find(did):
        while did != parent[did]:
            parent[did] = parent[parent[did]]
            did = parent[did]
        return did

    templates = {}
    roles = {}
    ids = set()
    for r in rows:
        if r["id"] in ids:
            raise ValueError(f"Duplicate row ID {r['id']}")
        ids.add(r["id"])
        if roles.setdefault(r["intent"], r["role"]) != r["role"]:
            raise ValueError(f"Conflicting role for {r['intent']}")
        did = r["dialogue_id"]
        representative = templates.setdefault(_template(r["text"]), did)
        a, b = find(did), find(representative)
        parent[max(a, b)] = min(a, b)

    groups = defaultdict(list)
    for r in rows:
        groups[find(r["dialogue_id"])].append(r)
    splits = {key: [] for key in ("fit", "router", "cal", "test")}
    mixed_rows = 0
    for gid in sorted(groups):
        group = groups[gid]
        group_roles = {role for r in group for role in r.get("dialogue_roles", [r["role"]])}
        if {"dev_unknown", "test_unknown"} <= group_roles:
            mixed_rows += len(group)
            continue
        if "test_unknown" in group_roles:
            phase = "test"
        elif "dev_unknown" in group_roles:
            phase = "cal"
        else:
            bucket = int.from_bytes(hashlib.sha256(f"{seed}|{gid}".encode()).digest()[:8], "big") % 100
            phase = "fit" if bucket < 60 else "router" if bucket < 80 else "cal" if bucket < 90 else "test"
        splits[phase].extend(group)
    for group in splits.values():
        group.sort(key=lambda r: r["id"])
    trained = {r["intent"] for r in splits["fit"]}
    untrained = {r["intent"] for group in splits.values() for r in group if r["role"] == "known"} - trained
    if untrained:
        raise ValueError(f"Known intents without fit data: {sorted(untrained)}. Broaden the data or revise the catalog.")
    metadata = {"seed": seed, "dialogue_template_groups": len(groups), "excluded_mixed_rows": mixed_rows,
                "counts_by_role": {phase: dict(Counter(r["role"] for r in group))
                                   for phase, group in splits.items()}}
    return {**splits, "known": sorted(trained),
            "dev_unknown": sorted({r["intent"] for r in splits["cal"] if r["role"] == "dev_unknown"}),
            "test_unknown": sorted({r["intent"] for r in splits["test"] if r["role"] == "test_unknown"}),
            "metadata": metadata}
