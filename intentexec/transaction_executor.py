"""Fit-only transaction tool selection and argument grounding in the public dialogue."""
import calendar
from collections import Counter, defaultdict
from datetime import date, timedelta
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression

from .executors import BASE, EXPERTS, RetrievalPool


_MONTH = {name.casefold(): i for i in range(1, 13)
          for name in (calendar.month_name[i], calendar.month_abbr[i])}
_MONTH_WORD = "|".join(sorted(_MONTH, key=len, reverse=True))
_WEEKDAYS = {name.casefold(): i for i, name in enumerate(calendar.day_name)}
_DAY = r"([0-3]?\d)(?:st|nd|rd|th)?"
_RELATIVE = {"today": 0, "tomorrow": 1, "day after tomorrow": 2, "yesterday": -1}
_REL_EXPR = re.compile(r"\b(day after tomorrow|today|tomorrow|yesterday)\b", re.I)
_DATE_EXPRESSIONS = [
    ("iso", re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b")),
    ("month_day", re.compile(rf"\b({_MONTH_WORD})\.?\s+(?:the\s+)?{_DAY}\b(?:,?\s+(\d{{4}})\b)?", re.I)),
    ("day_month", re.compile(rf"\b{_DAY}\s+(?:of\s+)?({_MONTH_WORD})\b(?:,?\s+(\d{{4}})\b)?", re.I)),
    ("ordinal", re.compile(r"\b(\d{1,2})(?:st|nd|rd|th)\b(?:\s+of this month)?", re.I)),
    ("relative", _REL_EXPR),
    ("weekday", re.compile(r"\b(?:(this|next)\s+)?(" + "|".join(_WEEKDAYS) + r")\b", re.I)),
]
_CLOCK = re.compile(r"\b(\d{1,2})(?::([0-5]\d))?(?:\s+o['’]clock)?\s*(a\.?m\.?|p\.?m\.?|in the morning|in the afternoon|in the evening|at night)\b"
                    r"|\b([01]?\d|2[0-3]):([0-5]\d)\b|\b(noon|midnight)\b", re.I)
_NUMBER_WORDS = dict(zip("zero one two three four five six seven eight nine ten".split(), range(11)))
_NUMBER = r"(\d{1,2}|" + "|".join(_NUMBER_WORDS) + ")"
_UNITS = {"rooms": r"rooms?", "duration": r"days?|nights?", "adults": r"adults?",
          "people": r"people|persons?|passengers?|travelers?|travellers?|tickets?|seats?|riders?|guests?"}
_COUNT = {key: re.compile(r"\b" + _NUMBER + r"\s+(?:(?:adult|economy|economy extra|flexible|bus|event|hotel)\s+){0,2}(?:"
                          + unit + r")\b", re.I) for key, unit in _UNITS.items()}
_QUESTION = {key: re.compile(r"\b(?:how many|number of)\b.{0,36}\b(?:" + unit + r")\b", re.I)
             for key, unit in _UNITS.items()}
_BARE = re.compile(r"\b" + _NUMBER + r"\b", re.I)


def _calendar_anchor(rows):
    """Accept only a consistent fit-derived reference for otherwise ambiguous deictic dates."""
    votes, years = Counter(), Counter()
    for row in rows:
        dates = set()
        for value in row["gold"]["arguments"].values():
            if isinstance(value, str) and re.fullmatch(r"\d{4}-\d\d-\d\d", value):
                try:
                    dates.add(date.fromisoformat(value))
                except ValueError:
                    continue
        years.update(d.year for d in dates)
        recent = " ".join(row["text"].splitlines()[-2:])
        shifts = {_RELATIVE[m.group().casefold()] for m in _REL_EXPR.finditer(recent)}
        if len(shifts) == len(dates) == 1:
            votes[dates.pop() - timedelta(days=shifts.pop())] += 1

    def consensus(counts):
        if not counts:
            return None
        value, n = counts.most_common(1)[0]
        return value if n >= 5 and n >= .90 * sum(counts.values()) else None

    return consensus(votes), consensus(years)


def _date_mentions(text, anchor, year):
    found = []
    for kind, regex in _DATE_EXPRESSIONS:
        for m in regex.finditer(text):
            groups = m.groups()
            try:
                if kind == "iso":
                    value = date(*map(int, groups))
                elif kind == "month_day" or kind == "day_month":
                    month, day = groups[:2] if kind == "month_day" else (groups[1], groups[0])
                    value = date(int(groups[2] or year), _MONTH[month.casefold()], int(day))
                elif anchor is None:
                    continue
                elif kind == "relative":
                    value = anchor + timedelta(days=_RELATIVE[groups[0].casefold()])
                elif kind == "weekday":
                    delta = (_WEEKDAYS[groups[1].casefold()] - anchor.weekday()) % 7
                    value = anchor + timedelta(days=delta or (7 if groups[0] == "next" else 0))
                else:
                    value = anchor.replace(day=int(groups[0]))
            except (ValueError, TypeError):
                continue
            # Prefer the explicit month/year over an ordinal sharing the same span.
            found.append((m.start(), m.end(), value.isoformat()))
    ordered = []
    for start, end, value in sorted(found, key=lambda x: (x[0], -x[1])):
        if not ordered or start >= ordered[-1][1]:
            ordered.append((start, end, value))
    return ordered


def _read_date(text, anchor, year):
    mentions = _date_mentions(text, anchor, year)
    return mentions[-1][2] if mentions else None


def _date_range(slots, lines, anchor, year):
    start = [s for s in slots if any(w in s for w in ("check_in", "departure", "pickup", "start"))]
    end = [s for s in slots if any(w in s for w in ("check_out", "return", "dropoff", "end"))]
    if len(start) != 1 or len(end) != 1:
        return {}
    for line in reversed(lines):
        mentions = _date_mentions(line, anchor, year)
        for a, b in reversed(list(zip(mentions, mentions[1:]))):
            if a[2] <= b[2] and re.search(r"\b(?:to|through|until|ending|returning|check(?:ing)? out)\b",
                                          line[a[1]:b[0]], re.I):
                return {start[0]: a[2], end[0]: b[2]}
    return {}


def _read_clock(text):
    value = None
    for match in _CLOCK.finditer(text.casefold()):
        hour, minute, meridiem, military, minutes24, special = match.groups()
        if special:
            value = "12:00" if special == "noon" else "00:00"
        elif military:
            value = f"{int(military):02d}:{minutes24}"
        elif 1 <= int(hour) <= 12:
            afternoon = meridiem.startswith("p") or meridiem in {"in the afternoon", "in the evening", "at night"}
            value = f"{int(hour) % 12 + 12 * afternoon:02d}:{int(minute or 0):02d}"
    return value


def _count_kind(slot):
    if "number_of_rooms" == slot:
        return "rooms"
    if slot in ("number_of_days", "number_of_nights"):
        return "duration"
    if "number_of_adults" == slot:
        return "adults"
    if slot in ("group_size", "party_size", "travelers", "passengers", "number_of_tickets",
                "number_of_seats", "number_of_riders"):
        return "people"


def _read_count(lines, kind):
    value = None
    for i, line in enumerate(lines):
        matches = [m[1].casefold() for m in _COUNT[kind].finditer(line)]
        if not matches and i and line.startswith("USER:") and _QUESTION[kind].search(lines[i-1]):
            numbers = _BARE.findall(line.casefold())
            if len(numbers) == 1:
                matches = numbers
        if matches:
            val = matches[-1]
            value = str(_NUMBER_WORDS[val]) if val in _NUMBER_WORDS else str(int(val))
    return value


def _city_roles(slots):
    origin = [slot for slot in slots if slot in {"from_location", "origin", "origin_city"}]
    destination = [slot for slot in slots if slot in {"to_location", "destination", "destination_city"}]
    return (origin[0], destination[0]) if len(origin) == len(destination) == 1 else None


def _read_route(lines, pattern):
    origin, destination = None, None
    for line in lines:
        matches = list(pattern.finditer(line))
        for i, m in enumerate(matches):
            prefix = line[:m.start()].casefold()
            if re.search(r"\b(?:from|leaving|departing)\s+$", prefix):
                origin = m.group()
            elif re.search(r"\b(?:to|arriving in|going into)\s+$", prefix):
                destination = m.group()
            if i and re.fullmatch(r",?\s+to\s+", line[matches[i-1].end():m.start()], re.I):
                origin, destination = matches[i-1].group(), m.group()
    return origin, destination


def _decision_text(text):
    # Confirmation and correction often follow a search in the same conversation.
    turns = text.splitlines()
    return " ".join(turns[-5:] + turns[-2:])


_NAME_TOKEN = r"(?:[A-Z](?:[A-Za-z0-9'’.-]*[A-Za-z0-9])?\.?|[0-9]+[A-Za-z]+)"
_PROPER = re.compile(r"(?<!\w)" + _NAME_TOKEN + r"(?:[ \t]+(?:(?:of|and|&|vs\.?|the|by|at)[ \t]+)?"
                     + _NAME_TOKEN + r")*")
_OPENING = re.compile(r"^(?:(?:Yes|No|Okay|Ok|Please|Sure|You|I|Can|What|Which|The)\s+)+")
_NON_ENTITY = {"is", "it", "i", "i'm", "i'd", "yes", "no", "please", "sure", "okay", "ok", "what", "you", "the",
               "which", "can", "thanks", "that's", "i'll", "could", "would", "a", "an", *_MONTH, *_WEEKDAYS}


def _entity_slots(slots):
    names = [s for s in slots if s.endswith("_name") and "station" not in s]
    cities = [s for s in slots if s in {"city", "city_of_event", "destination", "location", "where_to"}]
    result = {}
    if len(names) == 1:
        result["name"] = names[0]
    if len(cities) == 1 and not _city_roles(slots):
        result["place"] = cities[0]
    return result


def _mentions(lines, method):
    for i, line in enumerate(lines):
        speaker, _, utterance = line.partition(":")
        for match in _PROPER.finditer(utterance):
            value = _OPENING.sub("", match.group())
            if not re.search(r"\b[A-Z]\.[A-Z]\.$", value):
                value = value.rstrip(".")
            if not value or value.casefold() in _NON_ENTITY:
                continue
            left = re.findall(r"\w+", utterance[:match.start()].casefold())[-3:]
            right = re.findall(r"\w+", utterance[match.end():].casefold())[:2]
            features = {"speaker": speaker, "method": method, "words": str(min(5, len(value.split()))),
                        "left": " ".join(left[-2:]), "right": " ".join(right),
                        "l1": left[-1] if left else "", "r1": right[0] if right else ""}
            yield value, i, features


class _PublicNames:
    """Score novel capitalized spans from their public context; fit labels align to call arguments."""

    def __init__(self, rows, catalog):
        features, targets = [], []
        for row in rows:
            gold = row["gold"]
            spec = catalog[gold["service"], gold["method"]]
            if not spec.get("is_transactional"):
                continue
            truth = {gold["arguments"][s].casefold(): kind for kind, s in _entity_slots(gold["arguments"]).items()}
            if truth:
                for value, _, f in _mentions(row["text"].splitlines()[-5:], gold["method"]):
                    features.append(f)
                    targets.append(truth.get(value.casefold(), "other"))
        self.classifier = None
        if len(set(targets)) > 1:
            self.words = DictVectorizer()
            self.classifier = LogisticRegression(C=2, max_iter=200).fit(self.words.fit_transform(features), targets)

    def extract(self, text, key, slots, args):
        needed = {kind: s for kind, s in _entity_slots(slots).items() if s not in args}
        if not needed or self.classifier is None:
            return {}
        lines = text.splitlines()[-5:]
        spans = list(_mentions(lines, key[1]))
        if not spans:
            return {}
        probabilities = self.classifier.predict_proba(self.words.transform([f for _, _, f in spans]))
        values = {}
        for j, kind in enumerate(self.classifier.classes_):
            if kind not in needed:
                continue
            candidates = [(p - .045 * (len(lines) - 1 - i), v) for (v, i, _), p in zip(spans, probabilities[:, j])
                          if p >= .60]
            if candidates:
                values[needed[kind]] = max(candidates)[1]
        return values


class _ServiceVariants:
    """Search and transaction examples can both identify a public provider's schema variant."""

    def __init__(self, rows, catalog):
        self.by_family = {}
        self.tools = defaultdict(list)
        for key, spec in catalog.items():
            if spec.get("is_transactional"):
                self.tools[key[0]].append(key)
        families = defaultdict(list)
        for row in rows:
            families[row["gold"]["service"].rsplit("_", 1)[0]].append(row)
        for family, examples in families.items():
            labels = [r["gold"]["service"] for r in examples]
            if len(set(labels)) > 1:
                words = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)
                x = words.fit_transform([_decision_text(r["text"]) for r in examples])
                self.by_family[family] = words, LogisticRegression(C=6, max_iter=200).fit(x, labels)

    def resolve(self, key, text):
        family = key[0].rsplit("_", 1)[0]
        if family not in self.by_family:
            return key
        words, classifier = self.by_family[family]
        p = classifier.predict_proba(words.transform([_decision_text(text)]))[0]
        service = classifier.classes_[p.argmax()]
        if service != key[0] and p.max() >= .65:
            options = self.tools.get(service, [])
            if (service, key[1]) in options:
                return service, key[1]
            if len(options) == 1:
                return options[0]
        return key


class _TransactionDetector:
    def __init__(self, rows, catalog):
        self.keys = sorted(key for key, spec in catalog.items() if spec.get("is_transactional"))
        indices = {key: i for i, key in enumerate(self.keys)}
        labels = [indices.get((r["gold"]["service"], r["gold"]["method"]), -1) for r in rows]
        self.classifier = None
        if len(set(labels)) < 2:
            return
        self.words = TfidfVectorizer(sublinear_tf=True, ngram_range=(1, 2), token_pattern=r"(?u)\b\w+\b")
        x = self.words.fit_transform([_decision_text(r["text"]) for r in rows])
        self.classifier = LogisticRegression(C=8, max_iter=200).fit(x, labels)

    def predict(self, text):
        if self.classifier is None:
            return None
        probabilities = self.classifier.predict_proba(self.words.transform([_decision_text(text)]))[0]
        i = probabilities.argmax()
        label = self.classifier.classes_[i]
        return self.keys[label] if label >= 0 and probabilities[i] >= .5 else None


class RepairedRetrievalPool(RetrievalPool):
    """Add transaction confirmation recognition and schema-typed extraction to the local executors."""

    def __init__(self, fit_rows, schema, seed=1118):
        super().__init__(fit_rows, schema, seed)
        self.anchor, self.year = _calendar_anchor(fit_rows)
        self.detectors = {name: _TransactionDetector(self.indices[name].rows, self.catalog)
                          for name in (*EXPERTS, BASE)}
        self.variants = {name: _ServiceVariants(self.indices[name].rows, self.catalog) for name in (*EXPERTS, BASE)}
        self.names = _PublicNames(fit_rows, self.catalog)
        by_tool = defaultdict(list)
        cities = set()
        for row in fit_rows:
            gold = row["gold"]
            by_tool[gold["service"], gold["method"]].append(gold["arguments"])
            pair = _city_roles(gold["arguments"])
            if pair:
                cities.update(gold["arguments"][name] for name in pair)
        self.cities = (re.compile(r"(?<!\w)(?:" + "|".join(re.escape(c) for c in sorted(cities, key=lambda c: (-len(c), c)))
                                  + r")(?!\w)", re.I) if cities else None)
        self.usual = {key: {name for name in self.catalog[key].get("optional_slots", {})
                            if sum(name in a for a in args) >= .9 * len(args)} for key, args in by_tool.items()}

    def execute(self, executor_id, text):
        legacy = super().execute(executor_id, text)
        key = self.detectors[executor_id].predict(text)
        if key is None:
            if legacy is None:
                return None
            key = legacy["service"], legacy["method"]
            if not self.catalog[key].get("is_transactional"):
                return legacy
        key = self.variants[executor_id].resolve(key, text)
        spec = self.catalog[key]
        optional = spec.get("optional_slots", {})
        slots = set(spec.get("required_slots", [])) | self.usual.get(key, set())
        if legacy and (legacy["service"], legacy["method"]) == key:
            slots.update(legacy["arguments"])
        args = self.lexicon.extract(slots, text)
        lines = text.splitlines()
        dates = [s for s in slots if "date" in s.split("_")]
        times = [s for s in slots if "time" in s.split("_")]
        if len(dates) == 1:
            value = _read_date(text, self.anchor, self.year)
            if value is not None:
                args[dates[0]] = value
        elif len(dates) == 2:
            args.update(_date_range(dates, lines, self.anchor, self.year))
        if len(times) == 1:
            value = _read_clock(text)
            if value is not None:
                args[times[0]] = value
        for slot in slots:
            kind = _count_kind(slot)
            if kind:
                value = _read_count(lines, kind)
                if value is not None:
                    args[slot] = value
            if (slot not in args and slot in self.usual.get(key, ()) and
                    isinstance(optional.get(slot), str) and optional[slot].casefold() != "dontcare"):
                args[slot] = optional[slot]
        roles = _city_roles(slots)
        if roles and self.cities:
            for slot, value in zip(roles, _read_route(lines, self.cities)):
                if value is not None:
                    args[slot] = value
        args.update(self.names.extract(text, key, slots, args))
        return {"service": key[0], "method": key[1], "arguments": args}
