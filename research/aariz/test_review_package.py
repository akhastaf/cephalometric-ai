"""Synthetic checks for audit safeguards and the non-executing review notebook."""

import ast
import json
from pathlib import Path
import unittest

from research.aariz.audit_archive import landmark_config, parse_landmarks


ROOT = Path(__file__).parent


class ReviewPackageTests(unittest.TestCase):
    def setUp(self):
        self.mapping = landmark_config(ROOT / "evidence/aariz-source-config.py")
        self.records = [
            dict(landmark_id=i, value=dict(x=30, y=40)) for i in self.mapping
        ]

    def test_identity_is_checked_independently_of_annotation_array_order(self):
        raw = json.dumps({"landmarks": list(reversed(self.records))})
        ids, points = parse_landmarks(raw, self.mapping, 100, 100)
        self.assertEqual(ids, list(reversed(self.mapping)))
        self.assertEqual(set(points), set(self.mapping))

    def test_missing_duplicate_and_unknown_ids_rejected(self):
        for records in [
            self.records[:-1],
            self.records[:-1] + [self.records[0]],
            self.records[:-1] + [dict(landmark_id="unknown", value=dict(x=1, y=1))],
        ]:
            with self.subTest(records=len(records)), self.assertRaises(ValueError):
                parse_landmarks(
                    json.dumps({"landmarks": records}), self.mapping, 100, 100
                )

    def test_invalid_coordinates_rejected(self):
        for x in [-1, 100, float("nan"), float("inf"), True, "30"]:
            records = [dict(p, value=dict(x=x, y=1)) for p in self.records]
            with self.subTest(x=x), self.assertRaises(ValueError):
                parse_landmarks(
                    json.dumps({"landmarks": records}), self.mapping, 100, 100
                )

    def test_full_schema_and_projection(self):
        schema = json.loads((ROOT / "landmarks-29.json").read_text())
        points = schema["landmarks"]
        self.assertEqual([p["output_index"] for p in points], list(range(29)))
        self.assertEqual([p["source_landmark_id"] for p in points], list(self.mapping))
        self.assertEqual(len({p["code"] for p in points}), 29)
        self.assertEqual(sum(p["dentalflow_v1_code"] is not None for p in points), 14)
        self.assertEqual(
            next(p for p in points if p["code"] == "UIT")["dentalflow_v1_code"], "U1"
        )
        self.assertIsNone(
            next(p for p in points if p["code"] == "N_prime")["dentalflow_v1_code"]
        )

    def test_proposal_disables_weights_and_full_training(self):
        config = json.loads((ROOT / "hrnet-w32.proposed.json").read_text())
        self.assertEqual(config["architecture"]["name"], "HRNet-W32")
        self.assertEqual(config["architecture"]["head"]["channels"], 29)
        self.assertIsNone(config["architecture"]["pretrained_checkpoint"])
        self.assertFalse(config["pilot_limits"]["automatic_full_run"])
        self.assertEqual(config["pilot_limits"]["planned_ccu_ceiling"], 10)

    def test_notebook_is_unexecuted_and_contains_no_training_or_import_calls(self):
        nb = json.loads((ROOT / "Aariz_HRNet_W32_review_outline.ipynb").read_text())
        self.assertEqual(nb["nbformat"], 4)
        for cell in nb["cells"]:
            if cell["cell_type"] != "code":
                continue
            self.assertIsNone(cell["execution_count"])
            self.assertEqual(cell["outputs"], [])
            tree = ast.parse("".join(cell["source"]))
            self.assertFalse(
                any(isinstance(n, (ast.Import, ast.ImportFrom)) for n in ast.walk(tree))
            )
            calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)]
            self.assertTrue(
                all(
                    isinstance(n.func, ast.Name)
                    and n.func.id in ("print", "RuntimeError")
                    for n in calls
                )
            )


if __name__ == "__main__":
    unittest.main()
