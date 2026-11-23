import re
import unittest
from datetime import date
from types import SimpleNamespace

from intentexec.query_grounding import ground_call


def pool_with(service, method, required, optional=(), transactional=False):
    spec = {"required_slots": required, "optional_slots": dict.fromkeys(optional, "dontcare"),
            "is_transactional": transactional}
    return SimpleNamespace(catalog={(service, method): spec}, cities=re.compile("Bath|York", re.I),
                           anchor=date(2019, 3, 1), year=2019, names=SimpleNamespace(extract=lambda *args: {}))


class QueryGroundingTests(unittest.TestCase):
    def test_explicit_unit_separates_star_from_optional_rooms_without_mutating_draft(self):
        pool = pool_with("Hotels_1", "SearchHotel", ["destination"], ["number_of_rooms", "star_rating"])
        call = {"service": "Hotels_1", "method": "SearchHotel", "arguments": {"destination": "Delhi", "number_of_rooms": "3"}}
        text = "USER: A hotel in Delhi\nSYSTEM: Here are four rooms.\nUSER: How about a three-star hotel?"
        fixed = ground_call(pool, text, call)
        self.assertEqual(fixed["arguments"], {"destination": "Delhi", "star_rating": "3"})
        self.assertEqual(call["arguments"]["number_of_rooms"], "3")
        fixed = ground_call(pool, "USER: Three rooms.\n" + text, call)
        self.assertEqual(fixed["arguments"]["number_of_rooms"], "3")

    def test_query_reuses_existing_typed_extractors_and_ignores_system_listings(self):
        slots = ["from_location", "to_location", "leaving_date", "leaving_time", "travelers"]
        pool = pool_with("Bus", "FindBus", slots)
        call = {"service": "Bus", "method": "FindBus", "arguments": {}}
        text = "USER: A bus from York to Bath tomorrow at 8 am\nSYSTEM: How many passengers?\nUSER: Two."
        self.assertEqual(ground_call(pool, text, call)["arguments"], dict(zip(slots, ("York", "Bath", "2019-03-02", "08:00", "2"))))
        fixed = ground_call(pool, "SYSTEM: I found two passengers. What date?\nUSER: Tomorrow.", call)
        self.assertNotIn("travelers", fixed["arguments"])

    def test_ride_uses_affirmed_plan_and_respects_negative_preference(self):
        pool = pool_with("RideSharing_1", "GetRide", ["shared_ride", "destination"], transactional=True)
        call = {"service": "RideSharing_1", "method": "GetRide", "arguments": {}}
        text = "SYSTEM: A shared cab to The Cartoon Museum?\nUSER: Yes, thanks."
        self.assertEqual(ground_call(pool, text, call)["arguments"], {"destination": "The Cartoon Museum", "shared_ride": "True"})
        self.assertEqual(ground_call(pool, "USER: I don't want a shared ride.", call)["arguments"], {"shared_ride": "False"})
        text = "SYSTEM: What about a pooled ride?\nUSER: What is a shared ride?"
        self.assertEqual(ground_call(pool, text, call), call)


if __name__ == "__main__":
    unittest.main()
