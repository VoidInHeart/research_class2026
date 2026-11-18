import unittest

from intentexec.support_holdout import _select


def row(did, i, text, intent="query"):
    return {"dialogue_id": did, "id": f"{did}:{i}", "text": text, "intent": intent}


class FreshHoldoutTests(unittest.TestCase):
    def test_quarantines_transitive_views_and_unscored_unknown_turns(self):
        fold = {"known": ["query"], "dev_unknown": ["balance"], "test_unknown": ["buy"]}
        prior = [row("old", 0, "USER: find 2026 tickets")]
        fresh = [row("dupe", 0, "USER: find 2019 tickets"),
                 row("dupe", 2, "USER: older\nA\nB\nC\nD\nE"),
                 row("chain", 0, "SYSTEM: other\nA\nB\nC\nD\nE"),
                 row("cal", 0, "find my bus"), row("outside", 0, "get my ride"),
                 row("clean", 0, "search for Paris"), row("clean", 2, "now buy for Paris", "buy")]
        active = {r["dialogue_id"]: {"query"} for r in fresh}
        active.update(cal={"query", "balance"}, outside={"query", "unlisted"}, clean={"query", "buy"})
        selected, excluded = _select(fresh, prior, active, {"old"}, fold)
        self.assertEqual([r["id"] for r in selected], ["clean:0", "clean:2"])
        self.assertEqual([r["role"] for r in selected], ["known", "test_unknown"])
        self.assertEqual(selected[0]["dialogue_roles"], ["known", "test_unknown"])
        self.assertEqual(excluded["excluded_rows"], {"connected_to_pilot_template": 3,
                         "contains_calibration_unknown": 1, "contains_out_of_catalog_intent": 1})

    def test_known_only_fold_retains_reason_a_test_unknown_was_quarantined(self):
        fold = {"known": ["query"], "dev_unknown": ["weather"], "test_unknown": ["play"]}
        fresh = [row("out", 0, "start it", "play"), row("ok", 0, "where is the bus")]
        rows, audit = _select(fresh, [], {"out": {"lookup", "play"}, "ok": {"query"}}, set(), fold)
        self.assertEqual([r["id"] for r in rows], ["ok:0"])
        self.assertEqual(audit["excluded_test_unknown_by_reason"], {"contains_out_of_catalog_intent": 1})
        self.assertEqual(audit["out_of_catalog_combinations_for_excluded_test_unknown"],
                         [{"outside_intents": ["lookup"], "requests": 1}])

    def test_rejects_even_unscored_shared_dialogue_id(self):
        fold = {"known": ["query"], "dev_unknown": [], "test_unknown": ["buy"]}
        with self.assertRaisesRegex(ValueError, "dialogue identifiers"):
            _select([], [], {"x": {"query"}}, {"x"}, fold)
