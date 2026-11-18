import json
import unittest

from intentexec.window_metrics import paired, summarize


def observation(uid, did, intent, top1, reject, score, exact, method, expert_win, base_win):
    return dict(id=uid, dialogue_id=did, intent=intent, top1=top1, gate_oos=reject, oos_score=score,
                correct=exact, method_correct=method, expert_best_correct=expert_win, base_correct=base_win,
                selected="base" if reject else "search", role="known" if intent in ("A", "B") else "test_unknown")


class WindowMetricTest(unittest.TestCase):
    def setUp(self):
        self.known = ["A", "B", "absent"]
        self.rows = [
            observation("a1", "d1", "A", "A", False, .1, False, True, True, False),
            observation("b1", "d1", "B", "A", True, .8, True, True, False, True),
            observation("a2", "d2", "A", "A", True, .6, False, False, True, False),
            observation("u1", "d3", "U", "A", True, .9, True, True, False, True),
            observation("u2", "d3", "U", "B", False, .5, False, True, False, True),
        ]

    def test_gate_classification_and_fixed_gate_ceiling_have_separate_denominators(self):
        s = summarize(self.rows, self.known)
        self.assertEqual((s["n"], s["dialogues"], s["n_known"]), (5, 3, 3))
        self.assertAlmostEqual(s["known_top1"], 2/3)
        self.assertAlmostEqual(s["known_macro_recall"], .5)
        self.assertEqual(s["known_accepted_top1"], 1)
        self.assertAlmostEqual(s["known_false_reject"], 2/3)
        self.assertAlmostEqual(s["unknown_recall"], .5)
        self.assertAlmostEqual(s["gate_balanced_accuracy"], (1/3 + .5) / 2)
        self.assertAlmostEqual(s["oos_auroc"], 4/6)
        self.assertAlmostEqual(s["exact"], 2/5)
        self.assertAlmostEqual(s["method_accuracy"], 4/5)
        self.assertAlmostEqual(s["gate_oracle_exact"], 3/5)
        json.dumps(s, allow_nan=False)

    def test_pairing_preserves_clusters_identity_and_direction(self):
        changed = [dict(r) for r in self.rows]
        changed[1].update(top1="B", gate_oos=False)
        changed[3].update(gate_oos=False, correct=False)
        p = paired(self.rows, list(reversed(changed)), self.known)
        self.assertEqual(p["known_top1"]["difference"], 1/3)
        self.assertAlmostEqual(p["exact"]["difference"], -.2)
        self.assertAlmostEqual(p["gate_balanced_accuracy"]["difference"], (1/3 - .5) / 2)
        self.assertEqual(p["gate_by_truth"]["known"]["rescued"], 1)
        self.assertEqual(p["gate_by_truth"]["unknown"]["harmed"], 1)
        self.assertEqual(p["gate_balanced_accuracy"]["rescued"], 1)
        self.assertEqual(p["gate_balanced_accuracy"]["harmed"], 1)
        self.assertLess(p["gate_balanced_accuracy"]["valid_draws"], p["bootstrap_draws"])
        zeros = paired(self.rows, self.rows, self.known)
        self.assertEqual(zeros["exact"]["ci95"], [0, 0])
        with self.assertRaises(ValueError):
            paired(self.rows + self.rows[:1], self.rows, self.known)
        with self.assertRaises(ValueError):
            paired(self.rows, self.rows[:-1], self.known)
        with self.assertRaises(ValueError):
            paired(self.rows, [dict(self.rows[0], intent="U"), *self.rows[1:]], self.known)
        json.dumps(p, allow_nan=False)

    def test_absent_strata_report_undefined_scores(self):
        subset = self.rows[-2:]
        s = summarize(subset, self.known)
        self.assertIsNone(s["known_top1"])
        self.assertIsNone(s["oos_auroc"])
        p = paired(subset, subset, self.known)
        self.assertIsNone(p["gate_balanced_accuracy"]["ci95"])
        self.assertEqual(p["known_top1"]["n"], 0)
        json.dumps(p, allow_nan=False)
