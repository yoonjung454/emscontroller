import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import SafetyConfig
from safety_manager import SafetyManager


class TestSafetyManager(unittest.TestCase):
    def test_default_max_intensity_zero_blocks_live_output(self):
        safety = SafetyManager(SafetyConfig())  # defaults -> safety_max_intensity=0
        check = safety.live_output_allowed_by_config()
        self.assertFalse(check.ok)

    def test_positive_max_intensity_allows_config_check(self):
        safety = SafetyManager(SafetyConfig(safety_max_intensity=40))
        check = safety.live_output_allowed_by_config()
        self.assertTrue(check.ok)

    def test_clamp_intensity_respects_zero_max(self):
        safety = SafetyManager(SafetyConfig(safety_max_intensity=0))
        self.assertEqual(safety.clamp_intensity(999), 0)

    def test_clamp_intensity_respects_configured_bounds(self):
        safety = SafetyManager(SafetyConfig(safety_max_intensity=50, safety_min_intensity=10))
        self.assertEqual(safety.clamp_intensity(999), 50)
        self.assertEqual(safety.clamp_intensity(-999), 10)
        self.assertEqual(safety.clamp_intensity(30), 30)

    def test_can_arm_live_requires_all_conditions(self):
        safety = SafetyManager(SafetyConfig(safety_max_intensity=50))
        result = safety.can_arm_live(
            live_mode_requested=True,
            serial_connected=True,
            handshake_ok=True,
            hand_detected=True,
            calibration_done=True,
        )
        self.assertTrue(result.ok)

    def test_can_arm_live_rejects_missing_calibration(self):
        safety = SafetyManager(SafetyConfig(safety_max_intensity=50))
        result = safety.can_arm_live(
            live_mode_requested=True,
            serial_connected=True,
            handshake_ok=True,
            hand_detected=True,
            calibration_done=False,
        )
        self.assertFalse(result.ok)

    def test_simulation_mode_always_allowed(self):
        safety = SafetyManager(SafetyConfig())  # even with max_intensity=0
        result = safety.can_arm_live(
            live_mode_requested=False,
            serial_connected=False,
            handshake_ok=False,
            hand_detected=False,
            calibration_done=False,
        )
        self.assertTrue(result.ok)

    def test_emergency_stop_calls_all_callbacks(self):
        safety = SafetyManager(SafetyConfig(safety_max_intensity=50))
        called = []
        safety.on_emergency_stop(lambda reason: called.append(reason))
        safety.trigger_emergency_stop("test reason")
        self.assertEqual(called, ["test reason"])
        self.assertEqual(safety.state.name, "TRIPPED")

    def test_runtime_check_fails_on_hand_lost(self):
        safety = SafetyManager(SafetyConfig(safety_max_intensity=50))
        result = safety.runtime_check(live_mode=True, serial_connected=True, hand_lost=True)
        self.assertFalse(result.ok)

    def test_runtime_check_ok_in_simulation_regardless_of_hand(self):
        safety = SafetyManager(SafetyConfig(safety_max_intensity=0))
        result = safety.runtime_check(live_mode=False, serial_connected=False, hand_lost=True)
        self.assertTrue(result.ok)


if __name__ == "__main__":
    unittest.main()
