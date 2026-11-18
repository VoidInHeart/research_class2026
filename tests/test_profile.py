import argparse
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from intentexec.profile import run_profile, warm_scenario


def infer(bundle, pool, policy, text):
    accepted = text == "known"
    return {"executor": "specialist" if accepted else "base", "route_ms": 0.25 if accepted else 0.0,
            "recognition_ms": 1.0, "execution_ms": 2.0, "call": {}}


class LocalProfileTests(unittest.TestCase):
    def test_bypass_requests_and_instance_time_accounting(self):
        rows = [{"id": 0, "text": "known"}, {"id": 1, "text": "unknown"}]
        with patch("intentexec.profile.infer_one", side_effect=infer):
            result = warm_scenario({}, None, "matrix", rows, 2, 3, 3.6)
        self.assertEqual(result["stage2_invocations"], 3)
        self.assertEqual(result["executor_counts"], {"base": 3, "specialist": 3})
        self.assertEqual(result["requests"], 6)
        self.assertEqual(result["stage2_if_invoked_ms"]["p50"], 0.25)
        self.assertAlmostEqual(result["estimated_usd_per_1000_requests"], result["occupied_seconds"] / 6)
        self.assertGreaterEqual(result["latency_ms"]["end_to_end"]["p95"],
                                result["latency_ms"]["service"]["p95"])

    def test_samples_only_public_rows_and_does_not_overwrite(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)
            (path / "public.jsonl").write_text('{"id":"one","text":"known"}\n', encoding="utf8")
            args = argparse.Namespace(run_dir=path, out=path / "profile.json", seed=1, limit=1,
                                      repeats=2, concurrency=[1], usd_hour=None, policy="matrix")
            bundle = {"recognizer": Mock(encoder="tfidf")}
            with patch("intentexec.profile.load_run", return_value=(bundle, None)), \
                    patch("intentexec.profile.infer_one", side_effect=infer) as service:
                report = run_profile(args)
                self.assertEqual(service.call_count, 3)  # cold first, two timed repeats
                self.assertEqual(report["startup"]["first_id"], "one")
                self.assertNotIn("estimated_usd", report["scenarios"][0])
                with self.assertRaises(FileExistsError):
                    run_profile(args)
                self.assertEqual(service.call_count, 3)

