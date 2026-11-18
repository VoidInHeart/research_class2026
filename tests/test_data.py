import json
import tempfile
import unittest
from pathlib import Path

from intentexec.data import load_sgd, public_schema, split_rows


class SGDRowsTest(unittest.TestCase):
    def test_only_user_history_is_public_and_future_heldout_turn_quarantines_dialogue(self):
        # The held-out method is mentioned later without an execution target.
        turns = [
            {"speaker": "USER", "utterance": "Find American food.", "frames": [
                {"service": "food", "state": {"active_intent": "Find"}}]},
            {"speaker": "SYSTEM", "utterance": "A reference-only answer.", "frames": [
                {"service": "food", "service_call": {"method": "Find", "parameters": {"cuisine": "American"}}}]},
            {"speaker": "USER", "utterance": "Can it also be booked?", "frames": [
                {"service": "food", "state": {"active_intent": "Book"}}]},
            {"speaker": "SYSTEM", "utterance": "For when?", "frames": []},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            train = root / "train"
            train.mkdir()
            (root / "catalog.json").write_text(json.dumps(
                {"classes": {"find": ["Find"], "book": ["Book"]}, "dev_unknown": [], "test_unknown": ["book"]}))
            schema = [{"service_name": "food"}, {"service_name": "independent"}]
            (train / "schema.json").write_text(json.dumps(schema))
            (train / "dialogues_001.json").write_text(json.dumps(
                [{"dialogue_id": "source", "turns": turns}]))
            rows = load_sgd(root, root / "catalog.json")
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["text"], "USER: Find American food.")
            self.assertEqual(rows[0]["gold"], {"service": "food", "method": "Find",
                                               "arguments": {"cuisine": "American"}})
            self.assertEqual(rows[0]["dialogue_roles"], ["known", "test_unknown"])
            self.assertEqual(public_schema(root), schema)

    def test_groups_stay_disjoint_and_split_is_order_independent(self):
        def row(did, suffix, utterance, label, role):
            return {"id": did + suffix, "dialogue_id": did, "text": utterance,
                    "intent": label, "role": role}

        rows = [row(chr(65 + i), "-1", "context " + chr(65 + i), "lookup", "known") for i in range(20)]
        rows += [
            row("cal-X", "-1", "Look at 1 pm!", "track", "dev_unknown"),
            row("test-X", "-1", "Let me book", "book", "test_unknown"),
            # Different dialogues whose literal/number-substituted prefixes coincide.
            row("mixed-A", "-1", "Transfer $100.", "pay", "dev_unknown"),
            row("mixed-B", "-1", "transfer $300!", "unknown", "test_unknown"),
            row("mixed-B", "-2", "My second request", "lookup", "known"),
        ]
        result = split_rows(rows)
        self.assertEqual(result, split_rows(rows[::-1]))
        self.assertEqual(result["metadata"]["excluded_mixed_rows"], 3)
        self.assertEqual(result["known"], ["lookup"])
        self.assertEqual(result["dev_unknown"], ["track"])
        self.assertEqual(result["test_unknown"], ["book"])
        for phase in ("fit", "router"):
            self.assertTrue(all(r["role"] == "known" for r in result[phase]))
        dsets = [{r["dialogue_id"] for r in result[key]} for key in ("fit", "router", "cal", "test")]
        self.assertEqual(sum(map(len, dsets)), len(set.union(*dsets)))


if __name__ == "__main__":
    unittest.main()
