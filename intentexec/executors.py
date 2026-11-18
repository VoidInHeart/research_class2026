"""Execution pools. An executor only receives the public dialogue prefix at inference.

The offline retrieval pool is a CPU engineering proxy; its fit-only span lexicon and
retrieved call templates are deliberately simple. The LoRA backend generates calls.
"""
import hashlib
import json
import re
from collections import defaultdict

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


EXPERTS = ("search", "transaction", "context")
BASE = "base"
_ACTION = re.compile(r"book|reserve|buy|purchase|make|transfer|pay|add|cancel|play|send|schedule|order|rent", re.I)


def function_catalog(schema):
    """Expose all public tools, regardless of the true domain or OOS status."""
    catalog = {}
    for service in schema:
        for item in service["intents"]:
            key = (service["service_name"], item["name"])
            catalog[key] = item
    if not catalog:
        raise ValueError("The shared execution schema must contain at least one public tool")
    return catalog


def valid_call(value, catalog):
    if not isinstance(value, dict) or set(value) != {"service", "method", "arguments"}:
        return False
    service, method, args = value["service"], value["method"], value["arguments"]
    if not isinstance(service, str) or not isinstance(method, str) or not isinstance(args, dict):
        return False
    spec = catalog.get((service, method))
    if spec is None:
        return False
    allowed = set(spec.get("required_slots", ())) | set(spec.get("optional_slots", {}))
    return all(key in allowed and isinstance(val, str) for key, val in args.items())


def specialist_rows(rows, expert, seed=1118, limit=None):
    """Equal row budgets. Reserve 20% of each adapter's examples for shared skills."""
    if expert not in EXPERTS or not rows:
        raise ValueError("Specialist training requires fit rows and a valid expert")
    if limit is not None and limit < 1:
        raise ValueError("Per-expert example budget must be positive")
    budget = min(len(rows), limit if limit is not None else max(1, len(rows) // 2))
    def stable(row):
        return hashlib.sha256(f"{seed}/{row['id']}".encode()).digest()
    shuffled = sorted(rows, key=stable)
    shared_count = budget // 5
    shared, remaining = shuffled[:shared_count], shuffled[shared_count:]
    if expert == "context":
        ordered = sorted(remaining, key=lambda r: -r["text"].count("\n"))
    else:
        action = expert == "transaction"
        ordered = sorted(remaining, key=lambda r: bool(_ACTION.search(r["gold"]["method"])) != action)
    return shared + ordered[:budget-shared_count]


def _words(text):
    return re.findall(r"\w+", text.casefold())


class SpanLexicon:
    def __init__(self, rows, schema):
        # All values come from fit calls or the public schema. For a slot choose the
        # latest explicitly mentioned span, preferring the longer span on a tie.
        self.values = defaultdict(lambda: defaultdict(dict))
        def add(slot, val):
            if not isinstance(val, str):
                return
            tokens = tuple(_words(val))
            if tokens:
                self.values[slot][tokens[0]].setdefault(tokens, val)
        for service in schema:
            for slot in service.get("slots", []):
                for val in slot.get("possible_values", []):
                    add(slot["name"], val)
        for row in rows:
            for slot, val in row["gold"]["arguments"].items():
                add(slot, val)

    def extract(self, slots, text):
        tokens = _words(text)
        args = {}
        for slot in sorted(slots):
            best = (-1, -1)
            by_first_token = self.values.get(slot, {})
            for i, start in enumerate(tokens):
                for key, value in by_first_token.get(start, {}).items():
                    end = i + len(key)
                    if (end, len(key)) > best and tuple(tokens[i:end]) == key:
                        best, args[slot] = (end, len(key)), value
        return args


def _tool_name(name):
    return re.sub(r"([a-z])([A-Z])", r"\1 \2", name.replace("_", " "))


def _perspective(text, expert):
    turns = text.splitlines()
    if expert == "transaction":
        return " ".join(turns[-5:] + turns[-1:])
    if expert == "search":
        return " ".join(t[5:] for t in turns if t.startswith("USER:")) or text
    return text


class TemplateIndex:
    def __init__(self, expert, rows, catalog):
        self.expert, self.rows, self.catalog = expert, rows, catalog
        self.keys = sorted(catalog)
        descriptions = [
            f"{_tool_name(service)} {_tool_name(method)} {catalog[service, method].get('description', '')}"
            for service, method in self.keys
        ]
        self.vectorizer = TfidfVectorizer(sublinear_tf=True, ngram_range=(1, 2),
                                           token_pattern=r"(?u)\b\w+\b")
        docs = [_perspective(r["text"], expert) for r in rows] + descriptions
        self.matrix = self.vectorizer.fit_transform(docs).tocsr()

    def predict(self, text, lexicon):
        v = self.vectorizer.transform([_perspective(text, self.expert)])
        scores = (self.matrix @ v.T).toarray().ravel()
        split = len(self.rows)
        # The general executor gives the public tool descriptions more weight. All
        # four can select tools absent from their fit examples.
        scores[split:] *= 1.5 if self.expert == BASE else 0.85
        best = int(np.argmax(scores))
        if scores[best] <= 0:
            return None
        if best < split:
            template = self.rows[best]["gold"]
            key = (template["service"], template["method"])
            slots = set(template["arguments"])
        else:
            key = self.keys[best - split]
            slots = set(self.catalog[key].get("optional_slots", {}))
        slots.update(self.catalog[key].get("required_slots", []))
        return {"service": key[0], "method": key[1], "arguments": lexicon.extract(slots, text)}


class RetrievalPool:
    experts = list(EXPERTS)
    base = BASE

    def __init__(self, fit_rows, schema, seed=1118):
        self.catalog = function_catalog(schema)
        if not fit_rows:
            raise ValueError("No fit examples for retrieval executors")
        self.lexicon = SpanLexicon(fit_rows, schema)
        self.indices = {}
        for name in (*EXPERTS, BASE):
            rows = fit_rows if name == BASE else specialist_rows(fit_rows, name, seed)
            self.indices[name] = TemplateIndex(name, rows, self.catalog)

    def execute(self, executor_id, text):
        return self.indices[executor_id].predict(text, self.lexicon)


def make_pool(fit_rows, schema, config):
    backend = config.get("backend", "retrieval")
    if backend == "retrieval":
        return RetrievalPool(fit_rows, schema, config.get("seed", 1118))
    if backend == "lora":
        from .lora import LoRAPool
        return LoRAPool(schema, config, fit_rows)
    raise ValueError(f"Unknown execution backend: {backend}")
