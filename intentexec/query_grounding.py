"""Ground the schema-typed fields of a single admitted local proposal."""
import re

from .executors import valid_call
from .routing import recent_five
from .transaction_executor import (
    _NUMBER_WORDS, _city_roles, _count_kind, _date_range, _read_clock, _read_count, _read_date, _read_route,
)


_STARS = re.compile(r"\b([1-5]|one|two|three|four|five)[ -]+stars?\b", re.I)
_AFFIRM = re.compile(r"^\s*(?:yes|yeah|yep|yup|correct|right|sounds (?:good|great)|that(?:'s| is) (?:right|correct)|that works|sure|okay|ok)\b", re.I)
_RIDE = re.compile(r"\b(?:ride|cab|taxi|car)\b", re.I)
_SHARED = re.compile(r"\b(?:shared|sharing|pool(?:ed)?)\b", re.I)
_PRIVATE = re.compile(r"\b(?:private|regular|unshared|no (?:shared|sharing)|(?:not|don't|do not) (?:want |like )?(?:a |to )?(?:shared|share|pool))\b", re.I)
_DESTINATION = re.compile(r"\b(?:to|drop(?:ped)? off at)\s+([A-Z0-9][^?!\n]*?)(?=[?.!,]*$|\s+(?:for\s+(?:\d+|one|two|three|four)\b|is that (?:right|correct)))")


def user_evidence(lines):
    """Keep user assertions and immediately affirmed system propositions."""
    evidence = []
    for i, line in enumerate(lines):
        speaker, _, words = line.partition(":")
        if speaker != "USER":
            continue
        if i and lines[i - 1].startswith("SYSTEM:") and _AFFIRM.match(words):
            evidence.append(lines[i - 1])
        evidence.append(line)
    return evidence


def explicit_count(lines, kind):
    found = None
    for i, line in enumerate(lines):
        if line.startswith("USER:"):
            # A preceding question enables a bare answer, but an arbitrary
            # number in a system listing must not become a user's preference.
            previous = lines[i - 1] if i else ""
            question = re.search(r"\b(?:how many|number of)\b[^?]*\?", previous, re.I)
            value = _read_count([question[0], line] if question else [line], kind)
            if value is not None:
                found = value
    return found


def ground_ride(args, slots, lines):
    for line in user_evidence(lines):
        lowered = line.casefold()
        if line.startswith("SYSTEM:") and not _RIDE.search(line):
            continue
        if re.match(r"^USER:\s*(?:what|is|how|why)\b", line, re.I) and "?" in line:
            continue
        if _PRIVATE.search(line):
            if "shared_ride" in slots:
                args["shared_ride"] = "False"
            if "ride_type" in slots:
                args["ride_type"] = "Regular"
        elif _SHARED.search(line):
            if "shared_ride" in slots:
                args["shared_ride"] = "True"
            if "ride_type" in slots:
                args["ride_type"] = "Pool"
        elif "luxury" in lowered and "ride_type" in slots:
            args["ride_type"] = "Luxury"
        if line.startswith("SYSTEM:") and "destination" in slots:
            match = _DESTINATION.search(line.rstrip())
            if match:
                args["destination"] = match[1].strip(" .,")


def ground_call(pool, text, call):
    """Repair an admitted proposal from the existing pool's schema and fit-only resources."""
    if not valid_call(call, pool.catalog):
        return call
    key = call["service"], call["method"]
    spec = pool.catalog[key]
    required = set(spec.get("required_slots", ()))
    optional = set(spec.get("optional_slots", ()))
    slots = required | optional
    args = dict(call["arguments"])
    lines = recent_five(text).splitlines()
    if spec.get("is_transactional"):
        if key[1] != "GetRide":
            return call
        ground_ride(args, slots, lines)
        return {**call, "arguments": args}

    users = [line for line in lines if line.startswith("USER:")]
    evidence = "\n".join(users)
    dates = sorted(slot for slot in slots if "date" in slot.split("_"))
    if dates:
        if len(dates) == 1:
            value = _read_date(evidence, pool.anchor, pool.year)
            if value is not None:
                args[dates[0]] = value
        elif len(dates) == 2:
            args.update(_date_range(dates, users, pool.anchor, pool.year))
    times = sorted(slot for slot in slots if "time" in slot.split("_"))
    if len(times) == 1:
        value = _read_clock(evidence)
        if value is not None:
            args[times[0]] = value
    counts = {slot: explicit_count(lines, kind) for slot in slots if (kind := _count_kind(slot))}
    args.update({s: val for s, val in counts.items() if val is not None})
    ratings = _STARS.findall(evidence)
    if ratings and key[0].startswith("Hotels_"):
        token = ratings[-1].casefold()
        rating = str(_NUMBER_WORDS.get(token, token))
        for slot in slots & {"star_rating", "rating"}:
            args[slot] = rating
        if ("number_of_rooms" in optional and args.get("number_of_rooms") == rating
                and explicit_count(text.splitlines(), "rooms") is None):
            args.pop("number_of_rooms")
    pair = _city_roles(slots)
    if pair and pool.cities:
        for slot, value in zip(pair, _read_route(users, pool.cities)):
            if value is not None:
                args[slot] = value
    # Use the existing location/name span model only for unfilled required slots.
    args.update(pool.names.extract("\n".join(lines), key, required, args))
    return {**call, "arguments": args}
