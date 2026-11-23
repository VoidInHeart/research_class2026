import unittest
from unittest.mock import Mock, patch

import numpy as np

from intentexec.executors import BASE, EXPERTS
from intentexec.experiment import infer_one
from intentexec.review_runtime import deployed_service_cues
from intentexec.transaction_executor import RepairedRetrievalPool


class ArgumentRepairPredictionTests(unittest.TestCase):
    def test_predict_repairs_the_single_selected_proposal_and_bypasses_oos(self):
        pool = Mock(spec=RepairedRetrievalPool)
        pool.catalog = {("RideSharing_2", "GetRide"): {"is_transactional": True,
                        "required_slots": ["destination"], "optional_slots": {"ride_type": "Regular"}}}
        raw = {"service": "RideSharing_2", "method": "GetRide", "arguments": {"destination": "Daly City"}}
        pool.execute.return_value = raw
        recognizer, router = Mock(), Mock()
        router.choose.return_value = [0]
        bundle = {"known": ["GetRide"], "recognizer": recognizer, "routers": {"matrix": router},
                  "experts": EXPERTS, "base": BASE}
        text = "USER: I want a shared ride to Daly City."
        for rejected in (False, True):
            with self.subTest(rejected=rejected):
                recognizer.predict_encoded.return_value = (np.array([[1.]]), [rejected])
                pool.execute.reset_mock()
                result = infer_one(bundle, pool, "matrix", text, expert_review="single")
                pool.execute.assert_called_once_with(BASE if rejected else EXPERTS[0], text)
                expected = {"destination": "Daly City"}
                if not rejected:
                    expected["ride_type"] = "Pool"
                self.assertEqual(result["call"]["arguments"], expected)
                self.assertEqual(result["proposal_count"], 1)
                self.assertIsNone(result["secondary_executor"])
                self.assertEqual(raw["arguments"], {"destination": "Daly City"})

    def test_unsupported_backend_cannot_silently_ignore_the_requested_repair(self):
        with self.assertRaisesRegex(ValueError, "retrieval-grounded"):
            infer_one({}, object(), "matrix", "USER: hello", expert_review="review")

    def test_release_model_is_loadable_and_shared_across_requests(self):
        first = deployed_service_cues()
        self.assertIs(first, deployed_service_cues())
        scores = first.transform(["USER: Find a salon near San Jose"])
        self.assertEqual(scores.shape, (1, len(first.services) + len(first.slots)))

    def test_serving_matches_peer_policy_and_bypasses_feature_loader_on_oos(self):
        spec = {"required_slots": ["city", "date"], "is_transactional": True}
        pool = Mock(spec=RepairedRetrievalPool)
        pool.catalog = {("A", "Reserve"): spec, ("B", "Reserve"): spec}
        a = {"service": "A", "method": "Reserve", "arguments": {"city": "Paris"}}
        b = {"service": "B", "method": "Reserve", "arguments": {"date": "2019-03-02", "city": "Lyon"}}
        pool.execute.side_effect = lambda expert, text: b if expert == "transaction" else a
        recognizer, matrix = Mock(), Mock()
        matrix.choose.return_value = [0]
        bundle = {"known": ["Reserve"], "recognizer": recognizer, "routers": {"matrix": matrix},
                  "experts": EXPERTS, "base": BASE}
        text = "USER: please reserve"
        with patch("intentexec.review_runtime.deployed_service_cues") as loader:
            cues = loader.return_value
            cues.services = ["A", "B"]
            cues.transform.return_value = np.array([[.2, .8]])
            recognizer.predict_encoded.return_value = (np.array([[1.]]), [True])
            out = infer_one(bundle, pool, "matrix", text, expert_review="review")
            loader.assert_not_called()
            pool.execute.assert_called_once_with("base", text)
            self.assertEqual(out["proposal_count"], 1)
            recognizer.predict_encoded.return_value = (np.array([[1.]]), [False])
            for mode in ["off", "review"]:
                pool.execute.reset_mock()
                out = infer_one(bundle, pool, "matrix", text, expert_review=mode)
                if mode == "off":
                    pool.execute.assert_called_once_with("search", text)
                    self.assertNotIn("expert_review", out)
                else:
                    self.assertEqual([c.args for c in pool.execute.call_args_list], [("search", text), ("transaction", text)])
                    self.assertEqual(out["call"], b)
                    self.assertEqual((out["primary_executor"], out["secondary_executor"], out["proposal_count"],
                                      out["review_reason"], out["review_action"]),
                                     ("search", "transaction", 2, "service_conflict", "replace"))
