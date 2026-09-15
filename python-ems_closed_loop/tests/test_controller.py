import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import ControlConfig, SafetyConfig
from controller import ClosedLoopController, ControlState
from safety_manager import SafetyManager


def make_controller(**overrides) -> ClosedLoopController:
    safety_cfg = SafetyConfig(safety_max_intensity=100, safety_min_intensity=0)
    control_cfg = ControlConfig(
        mode="proportional",
        kp=0.6,
        tolerance_percent=5.0,
        success_hold_seconds=1.0,
        max_step_up=4.0,
        max_step_down=8.0,
        control_period_s=0.0,  # no gating delay, so tests can call update() repeatedly
    )
    for key, value in overrides.items():
        setattr(control_cfg, key, value)
    safety = SafetyManager(safety_cfg)
    return ClosedLoopController(control_cfg, safety)


class TestClosedLoopController(unittest.TestCase):
    def test_no_target_is_idle(self):
        controller = make_controller()
        decision = controller.update(current_percent=50.0, now=0.0)
        self.assertEqual(decision.state, ControlState.IDLE)

    def test_missing_hand_waits(self):
        controller = make_controller()
        controller.set_target(60.0)
        decision = controller.update(current_percent=None, now=0.0)
        self.assertEqual(decision.state, ControlState.WAITING_FOR_HAND)
        self.assertEqual(decision.intensity, 0)

    def test_under_target_increases(self):
        controller = make_controller()
        controller.set_target(80.0)
        decision = controller.update(current_percent=20.0, now=1.0)
        self.assertEqual(decision.state, ControlState.INCREASING)
        self.assertGreater(decision.intensity, 0)

    def test_over_target_decreases(self):
        controller = make_controller()
        controller.set_target(20.0)
        controller._intensity = 50
        decision = controller.update(current_percent=80.0, now=1.0)
        self.assertEqual(decision.state, ControlState.DECREASING)
        self.assertLess(decision.intensity, 50)

    def test_step_size_is_capped(self):
        controller = make_controller(max_step_up=4.0, kp=10.0)  # huge gain, should still be clamped
        controller.set_target(100.0)
        decision = controller.update(current_percent=0.0, now=1.0)
        self.assertLessEqual(decision.intensity, 4)

    def test_controller_output_stays_within_generic_0_100_bound(self):
        # NOTE: the controller intentionally does NOT enforce safety_max_intensity
        # (the human-safety hardware ceiling) on its own internal math -- that
        # would make simulation mode unable to move at all whenever the config
        # default (0) is in effect. The hardware ceiling is applied exactly
        # once, at the point a value is about to be sent to real EMS hardware
        # (main.py._drive_hardware -> safety_manager.clamp_intensity), which is
        # covered by tests/test_safety_manager.py instead.
        controller = make_controller(max_step_up=50.0, kp=10.0)  # deliberately aggressive
        controller.set_target(100.0)
        for i in range(50):
            decision = controller.update(current_percent=0.0, now=float(i))
            self.assertLessEqual(decision.intensity, 100)
            self.assertGreaterEqual(decision.intensity, 0)

    def test_hardware_send_path_is_gated_by_safety_max_separately(self):
        # This is the actual safety-critical guarantee: whatever the controller
        # computes internally, SafetyManager.clamp_intensity() (used right
        # before any real serial SET command) still enforces the configured
        # hardware ceiling.
        controller = make_controller()
        controller.safety.config.safety_max_intensity = 30
        controller.set_target(100.0)
        decision = controller.update(current_percent=0.0, now=1.0)
        hardware_bound_value = controller.safety.clamp_intensity(decision.intensity)
        self.assertLessEqual(hardware_bound_value, 30)

    def test_success_after_holding_within_tolerance(self):
        controller = make_controller(success_hold_seconds=1.0, tolerance_percent=5.0)
        controller.set_target(50.0)
        d1 = controller.update(current_percent=49.0, now=0.0)
        self.assertEqual(d1.state, ControlState.HOLDING)
        d2 = controller.update(current_percent=49.0, now=1.5)
        self.assertEqual(d2.state, ControlState.SUCCESS)
        self.assertEqual(d2.intensity, 0)

    def test_success_is_terminal_not_oscillating(self):
        # Regression test: SUCCESS used to leave target_percent set, so once
        # intensity dropped to 0 and the (real or simulated) hand relaxed
        # back away from the target, error would grow again and the
        # controller would swing right back into INCREASING -- forever
        # cycling instead of actually stopping, defeating "성공 즉시 EMS를
        # 정지한다".
        controller = make_controller(success_hold_seconds=1.0, tolerance_percent=5.0)
        controller.set_target(50.0)
        controller.update(current_percent=49.0, now=0.0)
        success_decision = controller.update(current_percent=49.0, now=1.5)
        self.assertEqual(success_decision.state, ControlState.SUCCESS)
        self.assertIsNone(controller.target_percent)

        # Hand relaxes back toward 0 now that intensity is 0 -- this must
        # NOT resurrect pursuit of the old target.
        after_decision = controller.update(current_percent=10.0, now=3.0)
        self.assertEqual(after_decision.state, ControlState.IDLE)
        self.assertEqual(after_decision.intensity, 0)

    def test_force_zero_clears_target(self):
        controller = make_controller()
        controller.set_target(70.0)
        decision = controller.force_zero("test stop")
        self.assertEqual(decision.intensity, 0)
        self.assertIsNone(controller.target_percent)


if __name__ == "__main__":
    unittest.main()
