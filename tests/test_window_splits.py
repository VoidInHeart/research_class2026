import unittest
from collections import Counter

from intentexec.window_splits import _build_fold, _components


class WindowSplitTests(unittest.TestCase):
    def test_frozen_ancestry_and_all_window_template_quarantine(self):
        rows, source, active = [], {}, {}

        def add(did, phase, label, txt=None, other_labels=()):
            rows.append({"id": did, "dialogue_id": did, "intent": label, "text": txt or did})
            source[did] = phase
            active[did] = {label, *other_labels}

        add("fit-pure", "fit", "K")
        add("router-pure", "router", "K")
        add("cal-pure", "cal", "K")
        add("test-pure", "test", "K")
        add("new-cal", "test", "D")
        add("new-test", "cal", "T")
        add("mixed", "test", "K", other_labels=("D", "T"))
        add("future", "router", "K", other_labels=("T",))
        add("background", "test", "K", other_labels=("old-unscoped",))
        add("unsupported-fit", "fit", "X", other_labels=("T",))
        add("unsupported-eval", "test", "X")
        tail = "\nUSER: What town\nSYSTEM: Oakland\nUSER: Sounds good\nSYSTEM: How many\nUSER: "
        add("fit-overlap", "fit", "K", "USER: different prefix" + tail + "two 22")
        add("test-overlap", "test", "T", "USER: another prefix" + tail + "two 55")
        result = _build_fold(_components(rows), active, source, {"K", "X"}, Counter(K=2, X=1),
                             {"dev_unknown": ["D"], "test_unknown": ["T"]})
        self.assertEqual(result["known"], ["K"])
        self.assertEqual([r["id"] for r in result["fit"]], ["fit-pure"])
        self.assertEqual([r["id"] for r in result["router"]], ["router-pure"])
        self.assertEqual([(r["id"], r["role"]) for r in result["cal"]],
                         [("cal-pure", "known"), ("new-cal", "dev_unknown")])
        self.assertEqual([(r["id"], r["role"]) for r in result["test"]],
                         [("new-test", "test_unknown"), ("test-pure", "known")])
        self.assertEqual(result["metadata"]["known_without_clean_fit_support"], ["X"])
        self.assertEqual(result["metadata"]["excluded_rows"]["crosses_frozen_partitions"], 2)
        self.assertEqual(sum(len(result[p]) for p in ("fit", "router", "cal", "test"))
                         + sum(result["metadata"]["excluded_rows"].values()), len(rows))

    def test_components_include_transitive_long_short_matches(self):
        base = "USER: one\nSYSTEM: two\nUSER: three\nSYSTEM: four\nUSER: five"
        rows = [{"id": did, "dialogue_id": did, "text": text} for did, text in
                [("a", "USER: long\n" + base), ("b", base), ("c", "SYSTEM: another\n" + base)]]
        groups = _components(rows[::-1])
        self.assertEqual(set(groups), {"a"})
        self.assertEqual({r["id"] for r in groups["a"]}, {"a", "b", "c"})


if __name__ == "__main__":
    unittest.main()
