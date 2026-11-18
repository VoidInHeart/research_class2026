import json
import unittest

import numpy as np

from intentexec.near_gate_eval import apply_threshold, compare_gates, evaluate, threshold


class NearGateEvalTest(unittest.TestCase):
    def test_every_budget_and_deterministic_ties(self):
        scores = np.array([.4, .4, .4, .8, .4, .9, .1])
        ids = ["A", "B", "C", "D", "E", "HIGH", "LOW"]
        known = np.array([True, True, True, True, False, False, False])
        reversed_ids = ids[::-1]
        for k in range(5):
            boundary = threshold(scores, ids, known, k)
            self.assertEqual(json.loads(json.dumps(boundary)), boundary)
            mask = apply_threshold(scores, ids, boundary)
            self.assertEqual(mask[known].sum(), k)
            self.assertTrue(mask[-2])
            self.assertFalse(mask[-1])
            self.assertEqual(threshold(scores[::-1], reversed_ids, known[::-1], k), boundary)
            np.testing.assert_equal(apply_threshold(scores[::-1], reversed_ids, boundary), mask[::-1])
        self.assertEqual(threshold(scores, ids, known, 0)["ties"], "none")
        self.assertEqual(threshold(scores, ids, known, 1)["ties"], "all")
        self.assertEqual(threshold(scores, ids, known, 2)["ties"], "id_sha256")
        self.assertEqual(threshold(scores, ids, known, 4)["ties"], "all")

    def test_known_frontier_depends_only_on_known_scores(self):
        ids = ["a", "b", "c", "d"]
        known = [True, False, True, False]
        a = threshold([.2, 1, .8, 0], ids, known, 1)
        b = threshold([.2, -10, .8, 42], ids, known, 1)
        self.assertEqual(a, b)
        np.testing.assert_equal(apply_threshold([.8, .6], ["unseen1", "unseen2"], a), [True, False])
        with self.assertRaises(ValueError):
            threshold([1], ["a"], [False], 0)
        with self.assertRaises(ValueError):
            threshold([1], ["a"], [True], 2)

    def test_fixed_executors_and_pairing(self):
        truth = [False, False, True, True]
        old, new = [True, False, False, False], [False, True, True, False]
        outcomes = ([True, True, False, False], [True, True, True, False],
                    [False, False, True, True], [False, True, True, True])
        before, after = [evaluate(m, truth, *outcomes) for m in (old, new)]
        self.assertEqual((after["nk"], after["nu"], after["known_rejected"], after["unknown_rejected"],
                          after["accepted"], after["exact_correct"], after["method_correct"]), (2, 2, 1, 1, 2, 2, 3))
        self.assertEqual(after["known_exact_correct"] + after["unknown_exact_correct"], after["exact_correct"])
        paired = compare_gates(old, new, truth, ["d1","d1","d2","d3"], *outcomes)
        for name in ("exact", "method", "known_rejected", "unknown_rejected"):
            metric = paired[name]
            col = name + "_correct" if name in ("exact", "method") else name
            self.assertEqual(metric["delta_count"], after[col] - before[col])
            self.assertEqual(metric["gain"] - metric["loss"], metric["delta_count"])
        self.assertEqual((paired["exact"]["gain"], paired["exact"]["loss"]), (2, 1))
        self.assertEqual(paired["known_rejected"]["difference"], 0)
        same = compare_gates(old, old, truth, ["d1","d1","d2","d3"], *outcomes)
        self.assertEqual(same["unknown_rejected"]["ci95"], [0, 0])
