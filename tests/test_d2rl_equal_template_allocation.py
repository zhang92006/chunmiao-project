import unittest

from scenario_reconstruction.d2rl_equal_template_allocation import build_equal_allocation


class EqualTemplateAllocationTests(unittest.TestCase):
    def test_distributes_budget_equally_and_deterministically(self):
        reference = {
            "templates": {
                "c.json": {"source_event_id": 3},
                "a.json": {"source_event_id": 1},
                "b.json": {"source_event_id": 2},
            }
        }
        result = build_equal_allocation(reference, 8)
        self.assertEqual(result["allocation_total"], 8)
        self.assertEqual(
            {key: row["additional_rollouts"] for key, row in result["templates"].items()},
            {"a.json": 3, "b.json": 3, "c.json": 2},
        )

    def test_rejects_budget_smaller_than_template_count(self):
        with self.assertRaisesRegex(ValueError, "at least one"):
            build_equal_allocation({"templates": {"a": {}, "b": {}}}, 1)


if __name__ == "__main__":
    unittest.main()
