import unittest

import numpy as np

from intentexec.within_schema_cues import fit


class SchemaCueTests(unittest.TestCase):
    def test_scores_use_only_recent_public_utterances(self):
        rows = []
        for service, word, slot in [("Rail", "train", "station"), ("Hotels", "hotel", "city")]:
            for _ in range(12):
                rows.append({"text": f"USER: find a {word}", "gold": {"service": service, "arguments": {slot: word}}})
        model = fit(rows)
        suffix = "\n".join(["SYSTEM: yes", "USER: OK", "SYSTEM: where", "USER: near here", "USER: find a train"])
        scores = model.transform(["USER: hotel\n" + suffix, "USER: train\n" + suffix, "USER: find a hotel"])
        np.testing.assert_array_equal(scores[0], scores[1])
        np.testing.assert_allclose(scores[:, :2].sum(axis=1), 1)
        self.assertGreater(scores[0, model.services.index("Rail")], scores[2, model.services.index("Rail")])
        station = len(model.services) + model.slots.index("station")
        self.assertGreater(scores[0, station], scores[2, station])
