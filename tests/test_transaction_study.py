import unittest

import numpy as np

from intentexec.transaction_study import METRICS, POLICIES, score, subgroup_metrics


class ExecutionStudyTests(unittest.TestCase):
    def test_method_name_alone_and_arguments_are_distinct_from_complete_call(self):
        gold = {"service": "Bus_1", "method": "Book", "arguments": {"city": "San Jose"}}
        other_service = {**gold, "service": "Rail_1", "arguments": {"city": "san jose"}}
        self.assertEqual(score(other_service, gold), [False, True, False, True, True, False])
        wrong_value = {**gold, "arguments": {"city": "San Diego"}}
        self.assertEqual(score(wrong_value, gold), [True, True, True, True, False, False])

    def test_bootstrap_preserves_dialogue_clusters_and_empty_unknowns(self):
        shape = (3, len(POLICIES), len(METRICS))
        before = np.zeros(shape, dtype=bool)
        after = np.zeros(shape, dtype=bool)
        before[0, :, :] = True
        after[1, :, :] = True
        # Opposite outcomes in the same dialogue cancel in every cluster draw.
        result = subgroup_metrics(before, after, np.array(["a", "a", "b"]))
        metric = result["routes"][POLICIES[0]]["exact"]
        self.assertEqual((metric["gain"], metric["loss"], metric["ci95"]), (1, 1, [0., 0.]))
        empty = subgroup_metrics(before[:0], after[:0], np.array([]))
        self.assertEqual(empty["n"], 0)
        self.assertIsNone(empty["routes"][POLICIES[0]]["exact"]["difference"])
        self.assertIsNone(empty["routes"][POLICIES[0]]["exact"]["ci95"])
