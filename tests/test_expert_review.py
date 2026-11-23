import unittest
from types import SimpleNamespace

import numpy as np

from intentexec.expert_review import ExpertReview


class PeerReviewTests(unittest.TestCase):
    def setUp(self):
        spec = {"required_slots": ["city", "date"], "is_transactional": True}
        pool = SimpleNamespace(catalog={("A", "Reserve"): spec, ("B", "Reserve"): spec})
        self.cues = SimpleNamespace(services=["A", "B"], transform=lambda texts: np.array([[.2, .8]]))
        self.policy = ExpertReview(pool, self.cues)
        self.calls = []
        self.first = {"service": "A", "method": "Reserve", "arguments": {"city": "Lyon"}}
        self.second = {"service": "B", "method": "Reserve", "arguments": {"city": "Paris"}}

    def propose(self, name, text):
        self.calls.append(name)
        return self.first if len(self.calls) == 1 else self.second

    def test_service_conflict_requires_independent_agreement(self):
        outcome = self.policy.run("USER: hello", "search", self.propose)
        self.assertEqual(self.calls, ["search", "transaction"])
        self.assertEqual(outcome["action"], "replace")
        self.assertEqual(outcome["review"], self.second)

    def test_missing_required_slots_merge_without_overwriting_primary_values(self):
        self.cues.transform = lambda texts: np.array([[.8, .2]])
        self.second = {"service": "A", "method": "Reserve", "arguments": {"city": "Paris", "date": "2019-03-02"}}
        outcome = self.policy.run("USER: hello", "search", self.propose)
        self.assertEqual(self.calls, ["search", "context"])
        self.assertEqual(outcome["action"], "merge")
        self.assertEqual(outcome["review"]["arguments"], {"city": "Lyon", "date": "2019-03-02"})
        self.assertEqual(outcome["hard_switch"], self.second)

    def test_base_and_explicit_single_mode_use_no_second_proposal(self):
        for primary, enabled in [("base", True), ("search", False)]:
            self.calls.clear()
            self.policy.cues = None
            result = self.policy.run("USER: hello", primary, self.propose, collaborate=enabled)
            self.assertEqual(result["review"], self.first)
            self.assertEqual(self.calls, [primary])


if __name__ == "__main__":
    unittest.main()
