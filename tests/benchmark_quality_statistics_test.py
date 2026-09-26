"""Check uncertainty calculations and prevent unsupported quality claims."""

import json
import math
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks/tools"))
from quality_statistics import analyze_bundle, analyze_pairs, student_t_critical


class BenchmarkQualityStatisticsTest(unittest.TestCase):
    def records(self, advantages, model="dense", start=10):
        metric = "rmse" if model == "lstm" else "accuracy_percent"
        sign = -1 if model == "lstm" else 1
        return [
            {"model": model, "engine": engine, "seed": seed, "status": "ok",
             "metrics": {metric: 50 + sign * delta if engine == "opennn" else 50}}
            for seed, delta in enumerate(advantages, start)
            for engine in ("opennn", "pytorch")
        ]

    def confirmed(self, records, **overrides):
        arguments = {"planned_seeds": [10, 11, 12, 13, 14],
                     "margins": {"dense": 0.5}, "confirmatory": True,
                     "protocol_predeclared": True, "minimum_quality_reviewed": True}
        arguments.update(overrides)
        return analyze_pairs(records, **arguments)

    def test_student_t_against_reference_quantiles_and_cauchy(self):
        for degrees, probability, expected in (
            (4, 0.95, 2.131846786326649),
            (4, 0.975, 2.776445105197799),
            (9, 0.975, 2.262157162854099),
            (30, 0.995, 2.749995653567030),
            (100, 0.975, 1.983971518449633),
        ):
            self.assertAlmostEqual(student_t_critical(probability, degrees), expected, places=9)
        for probability in (0.75, 0.95, 0.99, 0.999):
            self.assertAlmostEqual(student_t_critical(probability, 1),
                                   math.tan(math.pi * (probability - 0.5)), places=8)

    def test_paired_interval_known_sample(self):
        row = analyze_pairs(self.records([-2, -1, 0, 1, 2]))["rows"][0]
        expected = 2.776445105197799 / math.sqrt(2)
        self.assertAlmostEqual(row["advantage_ci95"][0], -expected)
        self.assertAlmostEqual(row["advantage_ci95"][1], expected)
        self.assertFalse(row["claims"]["equivalence"]["established"])

    def test_lower_metric_orientation_preserves_raw_difference(self):
        row = analyze_pairs(self.records([1, 2, 3, 4, 5], "lstm"))["rows"][0]
        self.assertEqual(row["difference_mean"], -3)
        self.assertEqual(row["advantage_mean"], 3)
        self.assertFalse(row["observed_deficit"])
        self.assertEqual(row["pairs"][0]["difference"], -1)

    def test_historical_margin_cannot_establish_parity(self):
        row = analyze_pairs(self.records([-.02, .01, 0, .03, -.01]),
                            planned_seeds=[10, 11, 12, 13, 14],
                            margins={"dense": 1})["rows"][0]
        self.assertEqual(row["claims"]["equivalence"]["status"], "unassessed")
        self.assertIn("protocol_not_predeclared", row["claims"]["equivalence"]["reasons"])

    def test_equivalence_noninferiority_and_superiority_are_distinct(self):
        equal = self.confirmed(self.records([-.02, .01, 0, .03, -.01]))["rows"][0]
        self.assertTrue(equal["claims"]["equivalence"]["established"])
        self.assertTrue(equal["claims"]["noninferiority"]["established"])
        self.assertFalse(equal["claims"]["superiority"]["established"])
        advantage = self.confirmed(self.records([.9, 1, 1.1, 1.2, .8]))["rows"][0]
        self.assertFalse(advantage["claims"]["equivalence"]["established"])
        self.assertTrue(advantage["claims"]["noninferiority"]["established"])
        self.assertTrue(advantage["claims"]["superiority"]["established"])

    def test_deficit_is_retained_and_prevents_noninferiority(self):
        result = self.confirmed(self.records([-.9, -1, -1.1, -1.2, -.8]))
        row = result["rows"][0]
        self.assertTrue(row["observed_deficit"])
        self.assertEqual(row["descriptive_evidence"], "interval_below_zero")
        self.assertEqual(row["claims"]["noninferiority"]["status"], "not_established")
        self.assertFalse(result["all_models_noninferior"])

    def test_missing_margin_is_unassessed_not_failed(self):
        row = self.confirmed(self.records([-.02, .01, 0, .03, -.01]), margins={})["rows"][0]
        self.assertEqual(row["claims"]["equivalence"]["status"], "unassessed")
        self.assertEqual(row["claims"]["equivalence"]["reasons"], ["margin_missing"])

    def test_failed_missing_duplicate_invalid_and_unplanned_seeds_block_claims(self):
        for problem in ("failed", "missing", "duplicate", "nonfinite", "unplanned"):
            records = self.records([-.02, .01, 0, .03, -.01])
            if problem == "failed":
                records[0]["status"] = "error"
            elif problem == "missing":
                records.pop(0)
            elif problem == "duplicate":
                records.append(dict(records[0]))
            elif problem == "nonfinite":
                records[0]["metrics"]["accuracy_percent"] = float("nan")
            else:
                records.extend(self.records([-1], start=99))
            row = self.confirmed(records)["rows"][0]
            self.assertEqual(row["claims"]["equivalence"]["status"], "unassessed", problem)
            json.dumps(row, allow_nan=False)
            if problem == "unplanned":
                self.assertEqual(row["unplanned_seeds"], [99])
                self.assertEqual(row["complete_pairs"], 6)
                self.assertTrue(row["observed_deficit"])
            else:
                self.assertEqual(row["complete_pairs"], 4)
                self.assertTrue(row["pair_issues"])

    def test_absent_model_and_seed_cannot_vacuously_pass(self):
        result = self.confirmed([], models=["dense"])
        self.assertFalse(result["all_models_equivalent"])
        self.assertEqual(len(result["rows"][0]["pair_issues"]), 5)
        self.assertFalse(analyze_pairs([])["all_models_noninferior"])

    def test_single_pair_and_zero_variance_are_inconclusive(self):
        for values in ([0], [0, 0, 0, 0, 0], [1, 1, 1, 1, 1]):
            row = self.confirmed(self.records(values))["rows"][0]
            self.assertIsNone(row["advantage_ci90"])
            self.assertFalse(row["claims"]["equivalence"]["established"])
            self.assertFalse(row["claims"]["superiority"]["established"])

    def test_smoke_or_mismatched_work_blocks_claims(self):
        for problem in ("smoke", "driver"):
            records = self.records([-.02, .01, 0, .03, -.01])
            records[0][problem] = True if problem == "smoke" else {"epochs": 99}
            row = self.confirmed(records)["rows"][0]
            self.assertFalse(row["claims"]["equivalence"]["established"])

    def test_familywise_interval_accounts_for_all_planned_models(self):
        records = self.records([-.02, .01, 0, .03, -.01])
        records += self.records([1, 2, 3, 4, 5], "lstm")
        rows = analyze_pairs(records)["rows"]
        for row in rows:
            self.assertLess(row["advantage_familywise_ci95"][0], row["advantage_ci95"][0])
            self.assertGreater(row["advantage_familywise_ci95"][1], row["advantage_ci95"][1])

    def test_invalid_plan_or_margins_are_rejected(self):
        records = self.records([-.02, .01, 0, .03, -.01])
        for arguments in ({"margins": {"dense": -1}}, {"margins": {"dense": float("nan")}},
                          {"margins": {"other": 1}}, {"planned_seeds": [10, 10]},
                          {"planned_seeds": []}, {"models": []}):
            with self.assertRaises(ValueError):
                analyze_pairs(records, **arguments)

    def test_historical_and_native_bundles_preserve_missing_pairs(self):
        historical = {"models": {"cifar10_cnn": {"metric": "accuracy_percent",
            "direction": "higher", "scores_by_seed": {
                "42": {"opennn": 70, "pytorch": 71}, "43": {"opennn": 72}}}}}
        result = analyze_bundle(historical)
        self.assertEqual(result["rows"][0]["complete_pairs"], 1)
        self.assertTrue(result["rows"][0]["observed_deficit"])
        self.assertTrue(result["rows"][0]["pair_issues"])
        native = {"runs": self.records([-.02, .01, 0, .03, -.01]),
                  "provenance": {"arguments": {"seeds": "10,11,12,13,14,15"},
                                 "manifests": {"dense": {}}}}
        result = analyze_bundle(native)
        self.assertEqual(result["rows"][0]["pair_issues"][0]["seed"], 15)
        self.assertFalse(result["all_models_noninferior"])


if __name__ == "__main__":
    unittest.main()
