"""
controller.py
--------------
The closed-loop control law: takes (target %, current %) and produces the
next EMS control value (0-100, called `intensity_level` everywhere -- this is
NOT milliamps, see EMSChannel.setIntensity in the official openEMSstim
firmware).

Two interchangeable modes:
  - "proportional" (default): safety-limited step-wise P control. Every tick
    it can only move the intensity by at most `max_step_up`/`max_step_down`,
    so a bad reading or an aggressive command can never cause a big jump.
  - "pid": full PID with a clamped integral term (anti-windup) for when P
    alone settles too slowly. Still passes through the exact same step-size
    and final-value safety clamps.

This module never talks to hardware and never touches OpenCV/MediaPipe. It
only does arithmetic, so it is fully unit-testable (see tests/test_controller.py)
and is exactly what simulator.py exercises before any real hardware exists.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from config import ControlConfig
from safety_manager import SafetyManager


class ControlState(Enum):
    IDLE = "대기"
    WAITING_FOR_HAND = "손 인식 중"
    INCREASING = "자극 증가"
    DECREASING = "자극 감소"
    HOLDING = "목표 유지 중"
    SUCCESS = "성공"
    SAFETY_STOP = "안전 정지"


@dataclass
class ControlDecision:
    intensity: int
    state: ControlState
    error: Optional[float]
    message: str = ""


class ClosedLoopController:
    def __init__(self, config: ControlConfig, safety: SafetyManager):
        self.config = config
        self.safety = safety

        self.target_percent: Optional[float] = None
        self._intensity: int = 0

        self._success_since: Optional[float] = None
        self._last_step_time: float = 0.0

        # PID-only state
        self._integral: float = 0.0
        self._last_error: float = 0.0
        self._last_pid_time: Optional[float] = None

    # ------------------------------------------------------------------ #

    @property
    def intensity(self) -> int:
        return self._intensity

    def set_target(self, target_percent: float, initial_intensity: Optional[int] = None) -> None:
        """Start pursuing a new target. `initial_intensity`, if given (e.g.
        from adaptive_model's suggestion), seeds the starting output instead
        of 0 -- it is still clamped by SafetyManager either way, and the
        camera feedback loop corrects it from there regardless of how good
        the initial guess was."""
        self.target_percent = max(0.0, min(100.0, target_percent))
        self._success_since = None
        self._last_step_time = 0.0
        self._integral = 0.0
        self._last_pid_time = None
        if initial_intensity is not None:
            self._intensity = self.safety.clamp_working_intensity(initial_intensity)

    def force_zero(self, reason: str = "") -> ControlDecision:
        """Immediate, ungated drop to zero. Used by the safety layer -- this
        deliberately bypasses max_step_down because getting the current OFF
        is always safe, unlike ramping it up."""
        self._intensity = 0
        self.target_percent = None
        self._success_since = None
        return ControlDecision(0, ControlState.SAFETY_STOP, None, reason)

    def reset(self) -> None:
        self.target_percent = None
        self._intensity = 0
        self._success_since = None
        self._integral = 0.0
        self._last_pid_time = None

    # ------------------------------------------------------------------ #

    def update(self, current_percent: Optional[float], now: Optional[float] = None) -> ControlDecision:
        now = now if now is not None else time.monotonic()

        if self.target_percent is None:
            return ControlDecision(self._intensity, ControlState.IDLE, None, "목표 없음")

        if current_percent is None:
            self._success_since = None
            return ControlDecision(
                self._intensity, ControlState.WAITING_FOR_HAND, None, "손을 인식하지 못했습니다"
            )

        error = self.target_percent - current_percent

        if abs(error) <= self.config.tolerance_percent:
            if self._success_since is None:
                self._success_since = now
            held_for = now - self._success_since
            if held_for >= self.config.success_hold_seconds:
                self._intensity = 0
                # Clear the target so SUCCESS is a terminal state, not a
                # transient one. Without this, next tick would still see a
                # target percent while intensity=0 (stimulation just cut
                # off) -- as the hand relaxes back open, error would grow
                # again and the controller would swing right back into
                # INCREASING, defeating "성공 즉시 EMS를 정지한다" by turning
                # SUCCESS into a permanent increase/hold/success oscillation
                # instead of an actual stop. A fresh command is required to
                # pursue a new target after this.
                self.target_percent = None
                return ControlDecision(0, ControlState.SUCCESS, error, f"목표 달성 (오차 {error:+.1f}%)")
            return ControlDecision(
                self._intensity, ControlState.HOLDING, error, f"목표 유지 중 ({held_for:.1f}s)"
            )
        else:
            self._success_since = None

        # Settling time: only actually change the output every control_period_s,
        # even if update() is called once per camera frame.
        if now - self._last_step_time < self.config.control_period_s:
            state = ControlState.INCREASING if error > 0 else ControlState.DECREASING
            return ControlDecision(self._intensity, state, error, "안정화 대기 중")

        self._last_step_time = now

        if self.config.mode == "pid":
            raw_step = self._pid_step(error, now)
        else:
            raw_step = self.config.kp * error

        step = max(-self.config.max_step_down, min(self.config.max_step_up, raw_step))
        # Generic 0-100 sanity bound only -- the human-safety hardware ceiling
        # (safety_max_intensity) is applied once, at the point a value is
        # actually about to be sent to real EMS hardware (main.py._drive_hardware),
        # not here. See SafetyManager.clamp_working_intensity's docstring.
        new_intensity = self.safety.clamp_working_intensity(self._intensity + step)
        self._intensity = new_intensity

        if step > 0.01:
            state = ControlState.INCREASING
        elif step < -0.01:
            state = ControlState.DECREASING
        else:
            state = ControlState.HOLDING

        return ControlDecision(new_intensity, state, error, f"step={step:+.1f} -> {new_intensity}")

    def _pid_step(self, error: float, now: float) -> float:
        dt = (now - self._last_pid_time) if self._last_pid_time is not None else self.config.control_period_s
        dt = max(dt, 1e-3)

        self._integral += error * dt
        limit = self.config.integral_limit
        self._integral = max(-limit, min(limit, self._integral))  # anti-windup clamp

        derivative = (error - self._last_error) / dt

        output = (
            self.config.kp * error
            + self.config.ki * self._integral
            + self.config.kd * derivative
        )

        self._last_error = error
        self._last_pid_time = now
        return output
