"""
safety_manager.py
------------------
Every rule in this file exists because the project touches a person's body
with electrical stimulation. Nothing here is decorative: controller.py,
serial_link.py and gui.py all ask `SafetyManager` for permission before doing
anything that could reach real hardware, and `SafetyManager` is the only
place that is allowed to say "no".

This module does NOT decide what a safe stimulation intensity is for a human
body. That number (`config.safety.safety_max_intensity`) must be set by a
person, under expert supervision, after checking the actual hardware and the
actual participant. If it is 0 (the shipped default), live output is refused
everywhere, unconditionally.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Optional

from config import SafetyConfig


class SafetyState(Enum):
    IDLE = "대기"
    ARMED = "무장됨"
    STIMULATING = "자극 중"
    COOLDOWN = "휴식(쿨다운)"
    TRIPPED = "안전 정지"


@dataclass
class SafetyCheckResult:
    ok: bool
    reason: str = ""


class SafetyManager:
    """Tracks timers/state and answers "is it safe to do X right now?".

    This class is deliberately conservative: any ambiguous or unknown
    condition (calibration missing, hand not detected, serial not connected,
    max intensity is 0, ...) resolves to "not safe".
    """

    def __init__(self, safety_config: SafetyConfig):
        self.config = safety_config
        self.state = SafetyState.IDLE

        self._experiment_start: Optional[float] = None
        self._continuous_stim_start: Optional[float] = None
        self._cooldown_until: Optional[float] = None
        self._tripped_reason: str = ""

        # Fired on emergency stop (space bar, ESC, window close, red button,
        # hand lost, serial lost, any unhandled exception during the run).
        self._estop_callbacks: list[Callable[[str], None]] = []

    # ------------------------------------------------------------------ #
    # Wiring
    # ------------------------------------------------------------------ #

    def on_emergency_stop(self, callback: Callable[[str], None]) -> None:
        """Register a callback invoked with a human-readable reason whenever
        `trigger_emergency_stop` fires. Typically wired to serial_link.stop_all()
        and to a GUI status label."""
        self._estop_callbacks.append(callback)

    def trigger_emergency_stop(self, reason: str) -> None:
        self.state = SafetyState.TRIPPED
        self._tripped_reason = reason
        self._continuous_stim_start = None
        for cb in self._estop_callbacks:
            try:
                cb(reason)
            except Exception:
                # A misbehaving callback must never prevent the other
                # e-stop callbacks (especially "send STOP_ALL") from running.
                pass

    def reset_after_trip(self) -> None:
        """Explicit, deliberate re-arm after a TRIPPED state. Does not
        bypass any of the other checks in can_arm_live()."""
        if self.state == SafetyState.TRIPPED:
            self.state = SafetyState.IDLE
            self._tripped_reason = ""

    # ------------------------------------------------------------------ #
    # Experiment / continuous-stim timers
    # ------------------------------------------------------------------ #

    def start_experiment_if_needed(self) -> None:
        if self._experiment_start is None:
            self._experiment_start = time.monotonic()

    def experiment_elapsed_s(self) -> float:
        if self._experiment_start is None:
            return 0.0
        return time.monotonic() - self._experiment_start

    def total_time_exceeded(self) -> bool:
        return self.experiment_elapsed_s() > self.config.total_experiment_seconds

    def notify_stim_started(self) -> None:
        if self._continuous_stim_start is None:
            self._continuous_stim_start = time.monotonic()
        self.state = SafetyState.STIMULATING

    def notify_stim_stopped(self) -> None:
        self._continuous_stim_start = None
        if self.state == SafetyState.STIMULATING:
            self.state = SafetyState.ARMED

    def continuous_stim_elapsed_s(self) -> float:
        if self._continuous_stim_start is None:
            return 0.0
        return time.monotonic() - self._continuous_stim_start

    def continuous_stim_exceeded(self) -> bool:
        return self.continuous_stim_elapsed_s() > self.config.max_continuous_stim_seconds

    def start_cooldown(self) -> None:
        self._cooldown_until = time.monotonic() + self.config.cooldown_seconds
        self.state = SafetyState.COOLDOWN

    def cooldown_remaining_s(self) -> float:
        if self._cooldown_until is None:
            return 0.0
        return max(0.0, self._cooldown_until - time.monotonic())

    def is_in_cooldown(self) -> bool:
        if self._cooldown_until is None:
            return False
        if time.monotonic() >= self._cooldown_until:
            self._cooldown_until = None
            if self.state == SafetyState.COOLDOWN:
                self.state = SafetyState.ARMED
            return False
        return True

    # ------------------------------------------------------------------ #
    # Permission checks
    # ------------------------------------------------------------------ #

    def live_output_allowed_by_config(self) -> SafetyCheckResult:
        """Configuration-level gate, independent of runtime state.
        This is the check that enforces 'max intensity defaults to 0'."""
        if self.config.safety_max_intensity <= 0:
            return SafetyCheckResult(
                False,
                "안전 최대 제어값(safety_max_intensity)이 0입니다. "
                "config.json에서 담당자가 직접 값을 설정해야 실제 EMS 출력이 허용됩니다.",
            )
        if self.config.safety_max_intensity > 100:
            return SafetyCheckResult(False, "safety_max_intensity는 0~100 사이여야 합니다.")
        if self.config.safety_min_intensity < 0 or self.config.safety_min_intensity > self.config.safety_max_intensity:
            return SafetyCheckResult(False, "safety_min_intensity 설정이 올바르지 않습니다.")
        return SafetyCheckResult(True)

    def can_arm_live(
        self,
        *,
        live_mode_requested: bool,
        serial_connected: bool,
        handshake_ok: bool,
        hand_detected: bool,
        calibration_done: bool,
    ) -> SafetyCheckResult:
        """The full pre-flight check before any real-hardware ARM/SET is sent."""
        if not live_mode_requested:
            return SafetyCheckResult(True, "시뮬레이션 모드")

        if self.state == SafetyState.TRIPPED:
            return SafetyCheckResult(False, f"안전 정지 상태입니다: {self._tripped_reason}")

        cfg_check = self.live_output_allowed_by_config()
        if not cfg_check.ok:
            return cfg_check

        if not serial_connected:
            return SafetyCheckResult(False, "Arduino 시리얼 연결이 필요합니다.")
        if not handshake_ok:
            return SafetyCheckResult(False, "Arduino와 PING/PONG handshake가 완료되지 않았습니다.")
        if not calibration_done:
            return SafetyCheckResult(False, "캘리브레이션(펴짐/구부림)을 먼저 완료해야 합니다.")
        if not hand_detected:
            return SafetyCheckResult(False, "손이 인식되어야 자극을 시작할 수 있습니다.")
        if self.total_time_exceeded():
            return SafetyCheckResult(False, "전체 실험 제한시간을 초과했습니다.")
        if self.is_in_cooldown():
            return SafetyCheckResult(False, f"쿨다운 중입니다 ({self.cooldown_remaining_s():.1f}s 남음).")

        return SafetyCheckResult(True)

    def runtime_check(
        self,
        *,
        live_mode: bool,
        serial_connected: bool,
        hand_lost: bool,
    ) -> SafetyCheckResult:
        """Called every control tick while stimulation is active. Any
        failure here should be treated by the caller as "call
        trigger_emergency_stop() now"."""
        if not live_mode:
            return SafetyCheckResult(True)

        if not serial_connected:
            return SafetyCheckResult(False, "시리얼 연결이 끊어졌습니다.")
        if hand_lost:
            return SafetyCheckResult(False, "카메라가 손을 놓쳤습니다.")
        if self.continuous_stim_exceeded():
            return SafetyCheckResult(False, "최대 연속 자극 시간을 초과했습니다. 쿨다운이 필요합니다.")
        if self.total_time_exceeded():
            return SafetyCheckResult(False, "전체 실험 제한시간을 초과했습니다.")
        return SafetyCheckResult(True)

    def clamp_intensity(self, intensity: float) -> int:
        """The HARDWARE safety gate (safety_max_intensity, default 0). Every
        intensity value that could reach real EMS hardware -- i.e. anything
        about to go into serial_link.SerialLink.set_intensity() -- MUST pass
        through here first, and only here. This is deliberately NOT used for
        the controller's own internal math (see clamp_working_intensity)
        because that would make simulation mode (which never touches real
        hardware) impossible to demo/tune until someone raises this human-
        safety limit for no reason -- see config.py's SafetyConfig docstring."""
        lo = max(0, self.config.safety_min_intensity)
        hi = min(100, self.config.safety_max_intensity)
        if hi <= 0:
            return 0
        return int(round(min(max(intensity, lo), hi)))

    @staticmethod
    def clamp_working_intensity(intensity: float) -> int:
        """Generic 0-100 sanity bound for values that never leave software
        (the controller's own state, simulation feedback, GUI display).
        This is NOT a substitute for clamp_intensity() -- anything destined
        for real hardware must still pass through clamp_intensity()."""
        return int(round(min(max(intensity, 0.0), 100.0)))

    def clamped_ttl_ms(self, requested_ms: int) -> int:
        return int(min(max(requested_ms, 0), self.config.max_command_ttl_ms))
