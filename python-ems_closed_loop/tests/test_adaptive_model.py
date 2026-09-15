import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from adaptive_model import AdaptivePersonalizationModel
from config import AdaptiveModelConfig, SafetyConfig
from safety_manager import SafetyManager


def make_model(min_samples=5, degree=1) -> AdaptivePersonalizationModel:
    safety = SafetyManager(SafetyConfig(safety_max_intensity=100, safety_min_intensity=0))
    cfg = AdaptiveModelConfig(enabled=True, min_samples=min_samples, degree=degree)
    return AdaptivePersonalizationModel(cfg, safety)


def steady_row(intensity: int, percent: float) -> dict:
    return {
        "ems_intensity": intensity,
        "current_percent": percent,
        "control_state": "HOLDING",
    }


class TestAdaptiveModel(unittest.TestCase):
    def test_insufficient_data_falls_back_to_safe_default(self):
        model = make_model(min_samples=10)
        model.fit([steady_row(10, 20), steady_row(20, 40)])  # only 2 rows, need 10
        self.assertFalse(model.is_fitted)
        suggestion = model.suggest_initial_intensity(target_percent=80)
        self.assertEqual(suggestion, model.safety.config.safety_min_intensity)

    def test_fits_and_predicts_reasonable_linear_relationship(self):
        model = make_model(min_samples=5)
        rows = [steady_row(x, 2 * x) for x in range(10, 60, 10)]  # percent = 2 * intensity
        stats = model.fit(rows)
        self.assertTrue(model.is_fitted)
        self.assertEqual(stats.sample_count, len(rows))

        suggestion = model.suggest_initial_intensity(target_percent=60)
        # true relationship says intensity=30 for percent=60; allow slack for fit noise
        self.assertTrue(0 <= suggestion <= 100)
        self.assertAlmostEqual(suggestion, 30, delta=5)

    def test_prediction_is_always_clamped_to_safety_bounds(self):
        model = make_model(min_samples=3)
        model.safety.config.safety_max_intensity = 40
        rows = [steady_row(x, 5 * x) for x in range(10, 40, 5)]  # steep relationship
        model.fit(rows)
        suggestion = model.suggest_initial_intensity(target_percent=95)
        self.assertLessEqual(suggestion, 40)

    def test_ignores_non_steady_state_rows(self):
        model = make_model(min_samples=3)
        rows = [
            {"ems_intensity": 10, "current_percent": 5, "control_state": "INCREASING"},
            {"ems_intensity": 20, "current_percent": 10, "control_state": "DECREASING"},
        ]
        stats = model.fit(rows)
        self.assertEqual(stats.sample_count, 0)
        self.assertFalse(model.is_fitted)

    def test_disabled_model_always_returns_fallback(self):
        model = make_model(min_samples=1)
        model.config.enabled = False
        model.fit([steady_row(10, 20), steady_row(20, 40), steady_row(30, 60)])
        suggestion = model.suggest_initial_intensity(target_percent=80)
        self.assertEqual(suggestion, model.safety.config.safety_min_intensity)


if __name__ == "__main__":
    unittest.main()
