"""End-to-end window/gate isolation on intentionally conflicting long and short scores."""
import unittest
from collections import defaultdict
from unittest.mock import Mock

import numpy as np

from intentexec.window_experiment import evaluate_fold


def row(uid, intent, role="known"):
    return {"id": uid, "dialogue_id": uid, "intent": intent, "role": role}


class WindowExperimentTests(unittest.TestCase):
    def test_dual_uses_long_gate_short_intent_and_shared_expert_outcomes(self):
        split = {
            "known": ["a", "b"], "fit": [row("f0", "a"), row("f1", "b")],
            "router": [row("r0", "a"), row("r1", "b")],
            "cal": [row("c0", "a"), row("c1", "b"), row("c2", "u", "dev_unknown")],
            "test": [row("t0", "a"), row("t1", "a"), row("t2", "v", "test_unknown")],
        }
        raw = {
            "long": {"router": [[1., 0.], [0., 1.]],
                     "cal": [[.8, .0], [.0, .8], [.2, .1]],
                     "test": [[.10, .88], [.30, .10], [.86, .12]]},
            "short": {"router": [[1., 0.], [0., 1.]],
                      "cal": [[.6, .0], [.0, .6], [.2, .1]],
                      "test": [[.95, .05], [.91, .08], [.13, .25]]},
        }
        yes, no, method = (True, True), (False, False), (False, True)
        # A routes to search; B routes to transaction. On t0 the correct expert
        # deliberately differs from the expert associated with the true intent.
        table = {"r0": [yes, no, no, no], "r1": [no, yes, no, no],
                 "t0": [method, yes, no, no], "t1": [no, yes, no, yes],
                 "t2": [no, no, yes, no]}
        features = Mock()
        features.score.side_effect = lambda view, fit, phases, known: {
            phase: np.array(raw[view][phase]) for phase in phases}
        emitted = defaultdict(list)
        report = evaluate_fold(split, features, table, lambda v, rec: emitted[v].append(rec))
        self.assertEqual(features.score.call_count, 2)
        self.assertEqual(report["thresholds"], {"long": .8, "short": .6})

        original, recent, dual = (emitted[k] for k in ("original", "recent5", "dual"))
        self.assertEqual([r["gate_oos"] for r in dual], [r["gate_oos"] for r in original])
        self.assertEqual([r["oos_score"] for r in dual], [r["oos_score"] for r in original])
        self.assertNotEqual([r["gate_oos"] for r in dual], [r["gate_oos"] for r in recent])
        self.assertEqual([r["top1"] for r in dual], [r["top1"] for r in recent])
        self.assertNotEqual([r["top1"] for r in dual], [r["top1"] for r in original])
        self.assertEqual([r["selected"] for r in original], ["transaction", "base", "search"])
        self.assertEqual([r["selected"] for r in recent], ["search", "search", "base"])
        self.assertEqual([r["selected"] for r in dual], ["search", "base", "transaction"])
        names = ["search", "transaction", "context", "base"]
        for predictions in emitted.values():
            for record in predictions:
                expected = table[record["id"]][names.index(record["selected"])]
                self.assertEqual((record["correct"], record["method_correct"]), expected)
        stats = report["variants"]
        self.assertEqual(stats["original"]["exact"], 2 / 3)
        self.assertEqual(stats["recent5"]["exact"], 0)
        self.assertEqual(stats["dual"]["exact"], 1 / 3)
        self.assertEqual(stats["dual"]["gate_oracle_exact"], stats["original"]["gate_oracle_exact"])
        self.assertEqual(stats["dual"]["gate_oracle_exact"], 1.0)
        self.assertEqual(report["paired"]["dual_minus_original"]["exact"]["harmed"], 1)
