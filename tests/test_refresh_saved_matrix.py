from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

from intentexec.compare import compare
from intentexec.data import split_rows
from intentexec.executors import EXPERTS
from intentexec.experiment import infer_one
from intentexec.refresh_saved_matrix import POLICIES, refresh, source_rows


class SavedMatrixTests(unittest.TestCase):
    def test_only_original_router_subset_can_supply_refreshed_supervision(self):
        rows = [{"id": c + ":0", "dialogue_id": c, "text": "USER: context " + c,
                 "intent": "find", "role": "known"} for c in "abcdefghijklmnopqrstuvwxyz"]
        original = split_rows(rows, seed=1118)
        split = {k: [r["id"] for r in original[k]] for k in ("fit", "router", "cal", "test")}
        split.update({k: original[k] for k in ("known", "dev_unknown", "test_unknown")})
        split["router_used"] = list(reversed(split["router"]))[:2]
        saved = {"fit": original["fit"], "config": {"seed": 1118}}
        old = {"row_ids": split["router_used"], "expert_order": list(EXPERTS)}
        self.assertEqual([r["id"] for r in source_rows(rows, saved, split, old)], split["router_used"])
        with self.assertRaisesRegex(ValueError, "router_used"):
            source_rows(rows, saved, {**split, "router_used": split["fit"][:1]}, old)
        with self.assertRaisesRegex(ValueError, "fit examples"):
            source_rows(rows, {**saved, "fit": saved["fit"][:-1]}, split, old)

    def test_untrained_policies_fail_even_when_gate_would_bypass(self):
        with self.assertRaisesRegex(ValueError, "supports matrix"):
            infer_one({"supported_policies": POLICIES}, None, "mlp", "unknown")
        with TemporaryDirectory() as tmp:
            src = Path(tmp) / "source"
            src.mkdir()
            with self.assertRaisesRegex(ValueError, "outside"):
                refresh(src, src / "child", Path(tmp), Path(tmp))
            with self.assertRaises(FileExistsError):
                refresh(src, Path(tmp), Path(tmp), Path(tmp))
            (src / "summary.json").write_text('{"supported_policies":["matrix","always_base","matrix_no_gate"]}')
            with self.assertRaisesRegex(ValueError, "full experiment"):
                compare(SimpleNamespace(run_dir=src), client=object())
