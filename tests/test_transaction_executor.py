import unittest
from datetime import date

from intentexec.executors import RetrievalPool
from intentexec.transaction_executor import (
    RepairedRetrievalPool, _calendar_anchor, _date_range, _read_clock, _read_count, _read_date,
)


class GroundingTests(unittest.TestCase):
    def test_fit_calendar_and_missing_or_conflicting_reference(self):
        def row(day):
            return {"text": "SYSTEM: Leaving tomorrow?\nUSER: Yes", "gold": {"arguments": {"date": day}}}

        fitted = [row("2023-11-29")] * 6
        anchor, year = _calendar_anchor(fitted)
        self.assertEqual(anchor, date(2023, 11, 28))
        self.assertEqual(_read_date("this Thursday", anchor, year), "2023-11-30")
        self.assertEqual(_read_date("March 3rd 2022", None, None), "2022-03-03")
        self.assertEqual(_read_date("on March 3rd", None, year), "2023-03-03")
        self.assertIsNone(_read_date("on March 3rd", None, None))
        self.assertIsNone(_calendar_anchor(fitted + [row("2023-11-28")] * 6)[0])
        self.assertEqual(_calendar_anchor(fitted + [row("2023-99-99")]), (anchor, year))
        self.assertIsNone(_read_date("tomorrow", None, None))

    def test_date_range_requires_two_explicit_compatible_endpoints(self):
        keys = ["check_in_date", "check_out_date"]
        lines = ["SYSTEM: From November 29 through December 3 for 2 guests."]
        self.assertEqual(_date_range(keys, lines, None, 2023),
                         {"check_in_date": "2023-11-29", "check_out_date": "2023-12-03"})
        self.assertEqual(_date_range(keys, ["USER: December 5, two rooms."], None, 2023), {})
        self.assertEqual(_date_range(keys, ["USER: December 5 to November 1."], None, 2023), {})

    def test_counts_corrections_and_12_hour_clock(self):
        lines = ["SYSTEM: 3 rooms for two nights at 4:15 pm?", "USER: Make it one room instead."]
        self.assertEqual(_read_count(lines, "rooms"), "1")
        self.assertEqual(_read_count(lines, "duration"), "2")
        self.assertEqual(_read_count(["SYSTEM: How many tickets?", "USER: Just three."], "people"), "3")
        self.assertEqual(_read_clock("previously 6 am, now 5 o'clock in the afternoon"), "17:00")
        self.assertEqual(_read_clock("12:05 am"), "00:05")
        self.assertEqual(_read_clock("at noon"), "12:00")

    def test_transaction_pool_reuses_public_schema_and_grounds_new_directions(self):
        spec = [{"service_name": "Transit", "slots": [],
                 "intents": [{"name": "Reserve", "required_slots": ["origin", "destination", "departure_time"],
                              "optional_slots": {}, "is_transactional": True}]}]
        gold = {"service": "Transit", "method": "Reserve",
                "arguments": {"origin": "York", "destination": "Bath", "departure_time": "08:15"}}
        rows = [{"id": str(i), "text": "USER: Buy a ticket from York to Bath at 8:15 am", "gold": gold}
                for i in range(5)]
        text = "SYSTEM: From Bath to York at 7:35 pm?\nUSER: Yes, buy the ticket."
        old = RetrievalPool(rows, spec).execute("base", text)
        self.assertEqual(old["arguments"].get("origin"), "York")
        pool = RepairedRetrievalPool(rows, spec)
        for name in (*pool.experts, pool.base):
            self.assertEqual(pool.execute(name, text)["arguments"],
                             {"origin": "Bath", "destination": "York", "departure_time": "19:35"})


if __name__ == "__main__":
    unittest.main()
