"""Development selection isolation and deterministic evidence accounting."""
import unittest
from unittest.mock import patch

import numpy as np

from intentexec.support_selection import CANDIDATES, select_weights


def fixture(phase, size):
    labels = ["query", "book", "cancel", "weather"]
    return [{"id": f"{phase}:{i}", "dialogue_id": f"{phase}-{i}", "intent": labels[i % 4],
             "text": f"{phase} {chr(97+i//26)}{chr(97+i%26)}"} for i in range(size)]


class SupportSelectionTests(unittest.TestCase):
    def test_rotation_quarantines_unscored_mentions_and_groups_before_fitting(self):
        fit, router = fixture("fit", 16), fixture("router", 112)
        known = ["query", "book", "cancel", "weather"]
        bank = {r["text"]: r for r in fit + router}
        active = {r["dialogue_id"]: {r["intent"]} for r in fit + router}
        # The dialogue contains an extra intent that did not produce a scored turn.
        active[fit[0]["dialogue_id"]].add("book")
        active[router[0]["dialogue_id"]].add("book")
        test = self

        class FakeSupport:
            def __init__(self, train, train_vectors, available, refs, reference_vectors):
                self.hidden, = set(known) - set(available)
                self.occupied = {r["dialogue_id"] for r in train + refs}
                test.assertTrue(all(self.hidden not in active[did] for did in self.occupied))

            def components(self, texts, vectors):
                cases = [bank[t] for t in texts]
                test.assertTrue(self.occupied.isdisjoint(r["dialogue_id"] for r in cases))
                hidden = np.asarray([r["intent"] == self.hidden for r in cases], dtype=float)
                zeros = np.zeros(len(cases))
                return {"z_centroid": zeros, "z_semantic": hidden, "z_lexical": zeros}

        with patch("intentexec.support_selection.SupportGate", FakeSupport):
            choice = select_weights(fit, np.ones((len(fit), 1)), known, router, np.ones((len(router), 1)),
                                    dialogue_intents=active)
        self.assertEqual(set(choice["rotations"]), set(known))
        self.assertEqual(choice["candidate"], "semantic")
        self.assertEqual(choice["weights"], [0., 1., 0.])
        self.assertEqual(choice["macro_recall"]["semantic"], 1)
        self.assertEqual(set(choice["macro_recall"]), set(CANDIDATES))
        for result in choice["rotations"].values():
            self.assertEqual(sum(f["hidden"] for f in result["folds"]), 28)
            for fold in result["folds"]:
                self.assertEqual(fold["known_rejected"], round(fold["known"] * 29 / 377))

    def test_cross_phase_templates_and_small_rotations_fall_back(self):
        fit, router = fixture("fit", 12), fixture("router", 4)
        router[0]["text"] = fit[0]["text"]
        known = ["query", "book", "cancel", "weather"]
        result = select_weights(fit, np.ones((12, 1)), known, router, np.ones((4, 1)))
        self.assertEqual(result["excluded_cross_phase_rows"], 2)
        self.assertEqual(result["candidate"], "centroid")
        self.assertFalse(result["rotations"])
        with self.assertRaises(ValueError):
            select_weights(fit, np.ones((12, 1)), known, router, np.ones((4, 1)), dialogue_intents={})


if __name__ == "__main__":
    unittest.main()
