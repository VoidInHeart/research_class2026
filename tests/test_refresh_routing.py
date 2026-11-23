import unittest

import numpy as np

from intentexec.refresh_development import fit_fold, group_folds
from intentexec.refresh_routing import BASE_INDEX, choose_all, fit_tables
from intentexec.routing import Router


class RefreshRoutingTests(unittest.TestCase):
    def test_refreshed_three_way_equivalence_and_base_ties(self):
        weights = np.eye(2)[[0, 0, 0, 1, 1, 1]]
        known = ["A", "B"]
        labels = np.array(known)[weights.argmax(axis=1)]
        y = np.array([[1, 0, 0, 1]] * 3 + [[0, 1, 0, 0]] * 3)
        table = fit_tables(weights, labels, known, y)
        pick = choose_all(table, np.eye(2))
        old = Router("matrix").fit(weights, labels, known, y[:, :3])
        np.testing.assert_array_equal(pick["matrix3"], old.choose(np.eye(2)))
        np.testing.assert_array_equal(pick["matrix4"], [BASE_INDEX, 1])
        np.testing.assert_array_equal(pick["conservative"], [BASE_INDEX, 1])
        np.testing.assert_array_equal(pick["fixed_all"], [BASE_INDEX, BASE_INDEX])
        self.assertTrue(all(not len(ix) for ix in choose_all(table, np.empty((0,2))).values()))
        with self.assertRaises(ValueError):
            choose_all(table, [[1,1]])
        with self.assertRaises(ValueError):
            fit_tables(weights, labels, known, y[:, :3])

    def test_sparse_expert_wins_and_missing_intent(self):
        table = fit_tables([[1,0]], ["A"], ["A","B"], [[1,0,0,0]])
        pick = choose_all(table, np.eye(2))
        self.assertEqual(pick["matrix4"].tolist(), [0, 0])
        self.assertEqual(pick["conservative"].tolist(), [BASE_INDEX, BASE_INDEX])

    def test_oof_bypass_and_dialogue_template_group_isolation(self):
        rows = [{"id": f"{i}:0", "dialogue_id": str(i), "intent": "A",
                 "text": "USER: " + name}
                for i, name in enumerate(["rain", "sun", "lake", "beach", "grass", "road", "ROAD"])]
        groups = group_folds(rows)
        self.assertEqual(groups[-1], groups[-2])
        n = len(rows)
        table = fit_fold(rows, ["A"], np.ones((n,1)), np.ones(n,dtype=bool),
                         np.tile([1,0,0,0], (n,1)), np.zeros(n,dtype=int))
        self.assertEqual(table["selected"], "always_base")
        self.assertTrue(all(rule["exact"] == 0 for rule in table["oof"].values()))
