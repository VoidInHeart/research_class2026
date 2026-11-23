"""Request one peer proposal when the primary draft needs review."""
from .executors import BASE, EXPERTS, valid_call
from .query_grounding import ground_call


class ExpertReview:
    def __init__(self, pool, cues=None):
        self.pool = pool
        self.cues = cues

    def run(self, text, primary, propose, *, collaborate=True):
        """propose(expert, text) drafts a call. The policy returns one final tool call."""
        if primary not in (*EXPERTS, BASE):
            raise ValueError("Unknown primary executor")
        first = propose(primary, text)
        single = first if primary == BASE else ground_call(self.pool, text, first)
        result = {"baseline": first, "single": single, "review": single, "hard_switch": single,
                  "secondary": None, "reason": "none", "action": "keep"}
        if not collaborate or primary == BASE:
            return result
        if self.cues is None:
            raise ValueError("Peer review requires the frozen service predictor")
        valid = valid_call(single, self.pool.catalog)
        spec = self.pool.catalog[(single["service"], single["method"])] if valid else {}
        expected = None
        if valid and single["service"] in self.cues.services:
            scores = self.cues.transform([text])[0, :len(self.cues.services)]
            top = int(scores.argmax())
            margin = scores[top] - scores[self.cues.services.index(single["service"])]
            if scores[top] >= .5 and margin >= .3:
                expected = self.cues.services[top]
        missing = set(spec.get("required_slots", ())) - single["arguments"].keys() if valid else set()
        if expected:
            secondary = "transaction" if spec.get("is_transactional") else "search"
            secondary = "context" if primary == secondary else secondary
            reason = "service_conflict"
        elif missing or not valid:
            secondary = "context" if primary != "context" else "transaction"
            reason = "required_missing" if valid else "invalid"
        else:
            return result
        alternative = ground_call(self.pool, text, propose(secondary, text))
        result.update(secondary=secondary, reason=reason, hard_switch=alternative)
        if not valid_call(alternative, self.pool.catalog):
            return result
        if not valid or (expected and alternative["service"] == expected):
            result.update(review=alternative, action="replace")
        elif (single["service"], single["method"]) == (alternative["service"], alternative["method"]):
            added = {k: v for k, v in alternative["arguments"].items() if k in missing}
            if added:
                result.update(review={**single, "arguments": {**single["arguments"], **added}}, action="merge")
        return result
