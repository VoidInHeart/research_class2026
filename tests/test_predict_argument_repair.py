import unittest
from unittest.mock import Mock

import numpy as np

from intentexec.executors import BASE, EXPERTS
from intentexec.experiment import infer_one
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
                result = infer_one(bundle, pool, "matrix", text, repair_arguments=True)
                pool.execute.assert_called_once_with(BASE if rejected else EXPERTS[0], text)
                expected = {"destination": "Daly City"}
                if not rejected:
                    expected["ride_type"] = "Pool"
                self.assertEqual(result["call"]["arguments"], expected)
                self.assertEqual(raw["arguments"], {"destination": "Daly City"})

    def test_unsupported_backend_cannot_silently_ignore_the_requested_repair(self):
        with self.assertRaisesRegex(ValueError, "retrieval-grounded"):
            infer_one({}, object(), "matrix", "USER: hello", repair_arguments=True)
