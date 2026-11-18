"""Paired gate evaluation uses the declared score convention and fixed base/expert output."""
import unittest

from intentexec.support_metrics import METHODS, evaluate_records


class SupportMetricsTests(unittest.TestCase):
    def test_strict_cutoff_and_equal_known_budget_have_separate_denominators(self):
        rows = []
        for i, unknown in enumerate((False, False, False, False, True, True)):
            rows.append({"id": str(i), "dialogue_id": str(i // 2), "unknown": unknown,
                         "intent": "held" if unknown else "related" if i < 2 else "other",
                         "raw_centroid": i, "scores": {m: 1 if m == "centroid" else i for m in METHODS},
                         "expert_exact": not unknown, "expert_method": True,
                         "base_exact": unknown, "base_method": unknown})
        result = evaluate_records(rows, {m: 1. for m in METHODS}, 3., "related", rate=.25)
        self.assertEqual((result["n"], result["nk"], result["nu"], result["known_budget"]), (6, 4, 2, 1))
        for m in METHODS:
            self.assertEqual(result["matched"][m]["known_rejected"], 1)
        self.assertEqual(result["from_cal"]["centroid"]["known_rejected"], 0)  # equality is accepted
        self.assertEqual(result["from_cal"]["adaptive"]["known_rejected"], 2)
        self.assertEqual(result["legacy_calibrated"]["exact_correct"], 6)
        diff = result["from_cal_paired"]["adaptive"]
        self.assertEqual((diff["unknown_rejected"]["delta_count"], diff["unknown_rejected"]["n"]), (2, 2))
        self.assertEqual((diff["known_rejected"]["delta_count"], diff["known_rejected"]["n"]), (2, 4))
        self.assertEqual((diff["exact"]["delta_count"], diff["exact"]["n"]), (0, 6))
        self.assertEqual(result["matched"]["adaptive"]["related_known_n"], 2)
        self.assertEqual(result["matched"]["adaptive"]["related_known_rejected"], 0)

    def test_known_only_holdout_preserves_estimable_metrics(self):
        rows = [{"id": str(i), "dialogue_id": str(i // 2), "intent": "near", "unknown": False,
                 "expert_exact": True, "expert_method": True, "base_exact": False, "base_method": True,
                 "raw_centroid": float(i), "scores": {m: float(i) for m in METHODS}}
                for i in range(4)]
        result = evaluate_records(rows, {m: 1 for m in METHODS}, 1, "near", rate=.25)
        self.assertEqual((result["n"], result["nk"], result["nu"]), (4, 4, 0))
        for name, rejects in (("from_cal", 2), ("matched", 1)):
            metrics = result[name]["adaptive"]
            self.assertEqual(metrics["known_rejected"], rejects)
            self.assertEqual(metrics["related_known_rejected"], rejects)
            self.assertEqual(metrics["exact_correct"], 4 - rejects)
            self.assertEqual(metrics["method_correct"], 4)
            self.assertIsNone(metrics["unknown_recall"])
            effect = result[name + "_paired"]["adaptive"]
            self.assertEqual(effect["unknown_rejected"]["n"], 0)
            self.assertIsNone(effect["unknown_rejected"]["difference"])
            self.assertIsNone(effect["unknown_rejected"]["ci95"])
            self.assertEqual(effect["unknown_rejected"]["valid_draws"], 0)
            self.assertEqual(effect["exact"]["ci95"], [0., 0.])
