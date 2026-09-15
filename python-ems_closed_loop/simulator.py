"""
simulator.py
--------------
Everything needed to run the whole application - GUI, natural language
commands, closed-loop controller, data logging - with no camera and no
Arduino attached.

Two independent pieces, both designed to be drop-in replacements for the
real thing (same method names/shapes as vision_tracker.VisionTracker and
serial_link.SerialLink -- Python's structural typing means main.py does not
need to care which one it was given):

- `HandDynamicsSimulator` stands in for the camera + MediaPipe pipeline. It
  turns "EMS intensity that was applied" into "finger bend percent that a
  camera would have measured", with a first-order response lag, a saturating
  non-linearity (diminishing returns at high intensity, like real muscle
  response), and measurement noise. Reducing intensity lets the simulated
  hand relax back open over time, exactly like letting go of a real
  contraction.

- `SimulatedArduino` stands in for serial_link.SerialLink. It accepts the
  exact same PING/ARM/SET/STOP/STOP_ALL/STATUS vocabulary and can be told to
  simulate a dropped connection, so the safety-manager code path for "serial
  disconnected -> stop everything" can be exercised without unplugging
  anything.
"""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass
from typing import Optional


# --------------------------------------------------------------------------- #
# Hand dynamics (stands in for vision_tracker.VisionTracker)
# --------------------------------------------------------------------------- #


@dataclass
class HandSimConfig:
    response_tau_s: float = 0.9      # first-order lag time constant (bigger = slower response)
    noise_std_percent: float = 1.5   # simulated measurement noise (percentage points, 1 std dev)
    saturation_k: float = 40.0       # controls how quickly bend saturates vs. intensity


class HandDynamicsSimulator:
    """Simulates a hand's finger-bend response to an EMS intensity command."""

    def __init__(self, config: Optional[HandSimConfig] = None, seed: Optional[int] = None):
        self.config = config or HandSimConfig()
        self._rng = random.Random(seed)
        self._true_bend_percent: float = 0.0
        self._applied_intensity: int = 0
        self._last_update: Optional[float] = None

        # Test hooks: force the simulator to report "no hand" like a real
        # camera losing tracking, without touching any physics state.
        self.simulate_hand_lost: bool = False
        self._consecutive_lost_frames = 0

    def apply_intensity(self, intensity: int) -> None:
        self._applied_intensity = max(0, min(100, intensity))

    def step(self, now: Optional[float] = None) -> None:
        """Advance the physics by one tick. Call this once per control loop
        iteration, same cadence as a real camera frame would arrive."""
        now = now if now is not None else time.monotonic()
        dt = 0.0 if self._last_update is None else max(0.0, now - self._last_update)
        self._last_update = now

        # Saturating non-linear steady-state target for the current intensity
        # (fast initial response, diminishing returns near 100 -- similar
        # shape to a real muscle recruitment curve).
        steady_state = 100.0 * (1.0 - math.exp(-self._applied_intensity / self.config.saturation_k))

        if dt > 0:
            alpha = 1.0 - math.exp(-dt / self.config.response_tau_s)
            self._true_bend_percent += (steady_state - self._true_bend_percent) * alpha

        self._true_bend_percent = max(0.0, min(100.0, self._true_bend_percent))

    def is_hand_detected(self) -> bool:
        return not self.simulate_hand_lost

    def get_current_bend_percent(self) -> Optional[float]:
        if self.simulate_hand_lost:
            return None
        noise = self._rng.gauss(0.0, self.config.noise_std_percent)
        return max(0.0, min(100.0, self._true_bend_percent + noise))


# --------------------------------------------------------------------------- #
# Simulated Arduino link (stands in for serial_link.SerialLink)
# --------------------------------------------------------------------------- #


@dataclass
class ChannelState:
    armed: bool = False
    intensity: int = 0
    active_until_ms: Optional[int] = None


class SimulatedArduino:
    """Drop-in stand-in for serial_link.SerialLink -- same public method
    names/shapes, so main.py can use either without an if/else."""

    def __init__(self, hand_dynamics: Optional[HandDynamicsSimulator] = None):
        self.hand_dynamics = hand_dynamics
        self._connected = False
        self._handshake_ok = False
        self.channels: dict[int, ChannelState] = {1: ChannelState(), 2: ChannelState()}
        self.last_error: str = ""

        # Test hook: simulate a dropped USB connection mid-run.
        self.simulate_disconnect: bool = False

    def connect(self, port: Optional[str] = None, baud_rate: int = 19200) -> bool:
        if self.simulate_disconnect:
            self.last_error = "시뮬레이션: 연결이 강제로 끊긴 상태입니다."
            return False
        self._connected = True
        self._handshake_ok = True  # simulated PING/PONG always succeeds instantly
        return True

    def disconnect(self) -> None:
        self._connected = False
        self._handshake_ok = False

    def is_connected(self) -> bool:
        return self._connected and not self.simulate_disconnect

    def handshake_ok(self) -> bool:
        return self._handshake_ok and self.is_connected()

    def ping(self) -> bool:
        return self.is_connected()

    def arm(self, channel: int) -> bool:
        if not self.is_connected() or channel not in self.channels:
            return False
        self.channels[channel].armed = True
        return True

    def set_intensity(self, channel: int, intensity: int, duration_ms: int) -> bool:
        if not self.is_connected():
            self.last_error = "시뮬레이션: 연결 안 됨"
            return False
        if channel != 1:
            self.last_error = "시뮬레이션: 채널 1만 지원합니다 (하드웨어 고장 상태 반영)"
            return False

        intensity = max(0, min(100, int(intensity)))
        state = self.channels[channel]
        state.intensity = intensity
        state.active_until_ms = int(time.monotonic() * 1000) + int(duration_ms)

        if self.hand_dynamics is not None:
            self.hand_dynamics.apply_intensity(intensity)
        return True

    def stop(self, channel: int) -> bool:
        if channel not in self.channels:
            return False
        self.channels[channel].intensity = 0
        self.channels[channel].active_until_ms = None
        if channel == 1 and self.hand_dynamics is not None:
            self.hand_dynamics.apply_intensity(0)
        return True

    def stop_all(self) -> bool:
        for ch in self.channels:
            self.stop(ch)
        return True

    def poll_ttl_expiry(self) -> None:
        """Mimics the Arduino's own millis()-based auto-deactivation when a
        SET command's duration elapses. Call periodically from the sim loop."""
        now_ms = int(time.monotonic() * 1000)
        for channel, state in self.channels.items():
            if state.active_until_ms is not None and now_ms >= state.active_until_ms:
                self.stop(channel)
