import unittest

from intentexec.refresh_evaluation import summarize


class RefreshEvaluationTests(unittest.TestCase):
    def test_hard_gate_voluntary_base_and_cluster_pairing(self):
        policies = ("legacy", "always_base", "candidate")
        rows = [
            {"id":"a", "dialogue_id":"d", "intent":"known", "rejected": False, "unknown": False,
             "choices":[0,3,3], "exact":[1,0,0,0], "tool":[1,0,0,0]},
            {"id":"b", "dialogue_id":"d", "intent":"known", "rejected": False, "unknown": False,
             "choices":[0,3,2], "exact":[0,0,1,0], "tool":[0,0,1,0]},
            {"id":"c", "dialogue_id":"q", "intent":"unknown", "rejected": True, "unknown": True,
             "choices":[3,3,3], "exact":[1,0,0,0], "tool":[1,0,0,0]},
        ]
        results = summarize(rows, policies, "known")
        all_rows = results["all"]
        self.assertEqual((all_rows["oracle_exact"], all_rows["gate_oracle_exact"]), (3, 2))
        cand = all_rows["policies"]["candidate"]
        self.assertEqual((cand["accepted_to_base"], cand["vs_legacy"]["gain"], cand["vs_legacy"]["loss"]), (1, 1, 1))
        self.assertEqual(cand["vs_legacy"]["ci95"], [0., 0.])
        self.assertEqual(results["near_known"]["n"], 2)
        self.assertEqual(results["unknown"]["gate_rejected"], 1)
        empty = summarize(rows, policies, None)["near_known"]
        self.assertEqual(empty["n"], 0)
        self.assertIsNone(empty["policies"]["candidate"]["vs_base"]["ci95"])
        rows[-1]["choices"][2] = 0
        with self.assertRaisesRegex(ValueError, "hard gate"):
            summarize(rows, policies, "known")
