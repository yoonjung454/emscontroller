"""
main.py
--------
Entry point. Wires config -> safety -> vision/simulator -> serial/simulator
-> controller -> adaptive model -> data logger -> command interpreter, runs
the whole pipeline in a background worker thread, and hands a thread-safe
snapshot of the current state to the Tkinter GUI (gui.py), which owns the
main thread as required by Tkinter.

Run modes (see README.md "실행 방법" for the full walkthrough):

    python main.py
        Full simulation. No camera, no Arduino required. This is the
        default and the ONLY mode that runs without any flags -- by design.

    python main.py --live-camera
        Real camera + MediaPipe, simulated Arduino/EMS (hand bend is real,
        stimulation is not). Useful for tuning calibration/control without
        touching hardware.

    python main.py --live-serial
        Real Arduino over serial, simulated hand (no camera). Useful for the
        "Arduino 단독 통신 검증" step in the README before any human/LED test.

    python main.py --live
        Both real. Requires config.json's safety.safety_max_intensity > 0
        (set deliberately by a person) or every real SET command is refused
        by safety_manager regardless of this flag.
"""

from __future__ import annotations

import argparse
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from adaptive_model import AdaptivePersonalizationModel
from command_interpreter import CommandInterpreter, CommandResult
from config import AppConfig, load_config, resolve_path
from controller import ClosedLoopController, ControlState
from data_logger import DataLogger, LogRow
from safety_manager import SafetyManager
from serial_link import SerialLink
from simulator import HandDynamicsSimulator, SimulatedArduino
from vision_tracker import VisionTracker

SIM_LOOP_HZ = 30.0


@dataclass
class Snapshot:
    """Read-only-by-convention state the GUI polls. Always copy, never
    mutate a Snapshot instance you were handed."""
    live_camera: bool = False
    live_serial: bool = False

    vision_ready: bool = False
    vision_error: str = ""
    hand_detected: bool = False
    hand_lost: bool = True
    is_calibrated: bool = False
    calibration_active: bool = False
    calibration_progress: tuple[int, int] = (0, 0)

    serial_connected: bool = False
    serial_handshake: bool = False
    serial_error: str = ""

    target_percent: Optional[float] = None
    current_percent: Optional[float] = None
    ems_intensity: int = 0
    error_percent: Optional[float] = None
    control_state: str = ControlState.IDLE.value
    success: bool = False

    last_command_text: str = ""
    last_message: str = ""

    elapsed_s: float = 0.0
    cooldown_remaining_s: float = 0.0

    frame_bgr: object = None  # numpy array or None; typed loosely to avoid importing cv2/np here
    history: list = field(default_factory=list)  # list[(t, target, current, intensity)] for the GUI graphs

    quit_requested: bool = False


class ControlLoopWorker:
    def __init__(self, config: AppConfig, live_camera: bool, live_serial: bool):
        self.config = config
        self.live_camera = live_camera
        self.live_serial = live_serial

        self.safety = SafetyManager(config.safety)
        self.controller = ClosedLoopController(config.control, self.safety)
        self.interpreter = CommandInterpreter(config.presets)
        self.adaptive_model = AdaptivePersonalizationModel(config.adaptive_model, self.safety)
        self.data_logger = DataLogger(resolve_path(config.logging.log_dir))

        self._hand_sim: Optional[HandDynamicsSimulator] = None
        self.vision: Optional[VisionTracker] = None
        self.stim: Optional[SimulatedArduino | SerialLink] = None

        self._lock = threading.Lock()
        self._snapshot = Snapshot(live_camera=live_camera, live_serial=live_serial)
        self._history: list[tuple[float, Optional[float], Optional[float], int]] = []

        self._pending_command: Optional[str] = None
        self._pending_percent: Optional[float] = None  # from quick-preset buttons
        self._pending_calibration: Optional[str] = None
        self._quit_requested = False
        self._stop_event = threading.Event()
        self._armed_ch1 = False

        self.safety.on_emergency_stop(self._on_emergency_stop)

        model_path = resolve_path(config.adaptive_model.model_path)
        self.adaptive_model.load(model_path)

    # ------------------------------------------------------------------ #
    # Setup
    # ------------------------------------------------------------------ #

    def initialize(self) -> None:
        model_asset = resolve_path(config_model_path(self.config))

        if self.live_camera:
            self.vision = VisionTracker(self.config.camera, self.config.hand_tracking, model_asset)
            try:
                self.vision.start()
            except RuntimeError as exc:
                self._set(vision_error=str(exc), vision_ready=False)
                self.vision = None
        else:
            self._hand_sim = HandDynamicsSimulator()
            self._set(vision_ready=True)

        if self.live_serial:
            self.stim = SerialLink(
                baud_rate=self.config.serial.baud_rate,
                read_timeout_s=self.config.serial.read_timeout_s,
                heartbeat_interval_s=self.config.safety.heartbeat_interval_s,
                on_disconnected=self._on_serial_disconnected,
            )
        else:
            self.stim = SimulatedArduino(hand_dynamics=self._hand_sim)
            # Simulated serial "connects" immediately -- there's nothing to plug in.
            self.stim.connect()

        self.safety.start_experiment_if_needed()

    def connect_serial(self, port: Optional[str]) -> bool:
        if not isinstance(self.stim, SerialLink):
            return True  # simulated link is always "connected"
        ok = self.stim.connect(port, connect_timeout_s=self.config.serial.connect_timeout_s)
        if not ok:
            self._set(serial_error=self.stim.last_error)
        return ok

    # ------------------------------------------------------------------ #
    # Commands from the GUI thread (thread-safe: just stash and let the
    # worker loop pick them up on its next tick)
    # ------------------------------------------------------------------ #

    def submit_command_text(self, text: str) -> None:
        with self._lock:
            self._pending_command = text

    def submit_quick_percent(self, percent: float) -> None:
        with self._lock:
            self._pending_percent = percent

    def request_calibration(self, mode: str) -> None:
        with self._lock:
            self._pending_calibration = mode

    def request_quit(self) -> None:
        self._quit_requested = True
        self._stop_event.set()

    def emergency_stop(self, reason: str = "사용자 비상정지") -> None:
        self.safety.trigger_emergency_stop(reason)

    def get_snapshot(self) -> Snapshot:
        with self._lock:
            snap = Snapshot(**self._snapshot.__dict__)
            snap.history = list(self._history[-600:])
            return snap

    # ------------------------------------------------------------------ #
    # Worker thread
    # ------------------------------------------------------------------ #

    def run(self) -> None:
        try:
            self.initialize()
            self.data_logger.start_session()
            while not self._stop_event.is_set():
                self._tick()
                if not self.live_camera:
                    time.sleep(1.0 / SIM_LOOP_HZ)
        finally:
            # This finally-block is the last line of defense: whatever
            # happened above (exception, quit request, window close), make
            # absolutely sure the hardware is told to stop.
            try:
                if self.stim is not None:
                    self.stim.stop_all()
            except Exception:
                pass
            try:
                if self.vision is not None:
                    self.vision.stop()
            except Exception:
                pass
            self.data_logger.close()

    def stop(self) -> None:
        self._stop_event.set()

    # ------------------------------------------------------------------ #

    def _tick(self) -> None:
        now = time.monotonic()

        # 1) sensing --------------------------------------------------- #
        hand_detected = False
        hand_lost = True
        current_percent: Optional[float] = None
        is_calibrated = False
        calibration_active = False
        calibration_progress = (0, 0)
        frame = None

        if self.vision is not None:
            result = self.vision.process_frame()
            frame = result.annotated_frame
            hand_detected = result.hand_detected_now
            hand_lost = result.hand_lost
            current_percent = result.bend_percent
            is_calibrated = result.is_calibrated
            calibration_active = result.calibration_active
            calibration_progress = result.calibration_progress
        elif self._hand_sim is not None:
            self._hand_sim.step(now)
            hand_detected = self._hand_sim.is_hand_detected()
            hand_lost = not hand_detected
            current_percent = self._hand_sim.get_current_bend_percent()
            is_calibrated = True  # simulated hand needs no calibration

        # 2) drain pending GUI requests --------------------------------- #
        pending_command, pending_percent, pending_calibration = self._drain_pending()

        if pending_calibration is not None and self.vision is not None:
            self.vision.begin_calibration(pending_calibration)

        message = ""
        if pending_percent is not None:
            self._apply_target(pending_percent)
            message = f"목표 굽힘 {pending_percent:.0f}% 설정 (빠른 실행)"

        if pending_command:
            result: CommandResult = self.interpreter.interpret(pending_command)
            message = result.message
            if result.action == "quit":
                self.request_quit()
            elif result.action == "stop":
                self.controller.reset()
                self._safe_stop_all()
            elif result.action == "set_target":
                self._apply_target(result.target_percent)
            # "error" -> message already carries the explanation; no state change

        # 3) safety runtime check ---------------------------------------- #
        serial_connected = self._serial_connected()
        runtime_check = self.safety.runtime_check(
            live_mode=self.live_serial, serial_connected=serial_connected, hand_lost=hand_lost
        )
        if not runtime_check.ok and self.safety.state.name != "TRIPPED":
            self.safety.trigger_emergency_stop(runtime_check.reason)

        # 4) control ------------------------------------------------------ #
        decision = self.controller.update(current_percent, now)
        intensity = decision.intensity

        if self.live_serial and self.safety.state.name != "TRIPPED":
            self._drive_hardware(intensity, decision)
        elif not self.live_serial and self.stim is not None:
            # Simulated stim link still needs the intensity forwarded so
            # HandDynamicsSimulator reacts, even though there is no "real"
            # arm/handshake ceremony to perform. Deliberately NOT gated by
            # safety.clamp_intensity() -- that gate is for real hardware only
            # (see SafetyManager.clamp_intensity's docstring), otherwise the
            # default safety_max_intensity=0 would make simulation mode
            # unable to move at all.
            self.stim.set_intensity(1, self.safety.clamp_working_intensity(intensity), self.config.safety.command_ttl_ms)
            if isinstance(self.stim, SimulatedArduino):
                self.stim.poll_ttl_expiry()

        if decision.state == ControlState.SUCCESS:
            self.safety.notify_stim_stopped()
            self.safety.start_cooldown()
            self._armed_ch1 = False
        elif intensity > 0:
            self.safety.notify_stim_started()
        else:
            self.safety.notify_stim_stopped()

        # 5) log + snapshot ------------------------------------------------ #
        self.data_logger.log_row(
            LogRow(
                timestamp=time.time(),
                target_percent=self.controller.target_percent,
                current_percent=current_percent,
                ems_intensity=intensity,
                error=decision.error,
                control_state=decision.state.value,
                success=(decision.state == ControlState.SUCCESS),
                hand_detected=hand_detected,
            )
        )

        with self._lock:
            self._history.append((now, self.controller.target_percent, current_percent, intensity))
            if len(self._history) > 2000:
                self._history = self._history[-1200:]

            self._snapshot = Snapshot(
                live_camera=self.live_camera,
                live_serial=self.live_serial,
                vision_ready=self.vision is not None or self._hand_sim is not None,
                vision_error=self._snapshot.vision_error,
                hand_detected=hand_detected,
                hand_lost=hand_lost,
                is_calibrated=is_calibrated,
                calibration_active=calibration_active,
                calibration_progress=calibration_progress,
                serial_connected=serial_connected,
                serial_handshake=self._serial_handshake_ok(),
                serial_error=getattr(self.stim, "last_error", ""),
                target_percent=self.controller.target_percent,
                current_percent=current_percent,
                ems_intensity=intensity,
                error_percent=decision.error,
                control_state=decision.state.value if self.safety.state.name != "TRIPPED" else "안전 정지",
                success=(decision.state == ControlState.SUCCESS),
                last_command_text=pending_command or self._snapshot.last_command_text,
                last_message=message or self._snapshot.last_message,
                elapsed_s=self.safety.experiment_elapsed_s(),
                cooldown_remaining_s=self.safety.cooldown_remaining_s(),
                frame_bgr=frame,
                quit_requested=self._quit_requested,
            )

    def _drain_pending(self):
        with self._lock:
            cmd, pct, cal = self._pending_command, self._pending_percent, self._pending_calibration
            self._pending_command = None
            self._pending_percent = None
            self._pending_calibration = None
        return cmd, pct, cal

    def _apply_target(self, percent: float) -> None:
        initial = self.adaptive_model.suggest_initial_intensity(percent)
        self.controller.set_target(percent, initial_intensity=initial)

    def _drive_hardware(self, intensity: int, decision) -> None:
        if not self._serial_connected() or not self._serial_handshake_ok():
            return
        if not self._armed_ch1:
            if self.stim.arm(1):
                self._armed_ch1 = True
            else:
                return
        ttl = self.safety.clamped_ttl_ms(self.config.safety.command_ttl_ms)
        self.stim.set_intensity(1, self.safety.clamp_intensity(intensity), ttl)

    def _safe_stop_all(self) -> None:
        try:
            if self.stim is not None:
                self.stim.stop_all()
        except Exception:
            pass
        self.safety.notify_stim_stopped()
        self._armed_ch1 = False

    def _serial_connected(self) -> bool:
        return self.stim is not None and self.stim.is_connected()

    def _serial_handshake_ok(self) -> bool:
        if isinstance(self.stim, SimulatedArduino):
            return self.stim.is_connected()
        if isinstance(self.stim, SerialLink):
            return self.stim.handshake_ok()
        return False

    def _on_emergency_stop(self, reason: str) -> None:
        self.controller.force_zero(reason)
        self._safe_stop_all()

    def _on_serial_disconnected(self, reason: str) -> None:
        self.safety.trigger_emergency_stop(f"시리얼 연결 끊김: {reason}")

    def _set(self, **kwargs) -> None:
        with self._lock:
            for k, v in kwargs.items():
                setattr(self._snapshot, k, v)


def config_model_path(config: AppConfig) -> str:
    return config.hand_tracking.model_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AI 비전 피드백 기반 폐루프 EMS 손동작 제어 시스템")
    parser.add_argument("--config", type=str, default=None, help="config.json 경로 (기본: 프로젝트 폴더의 config.json)")
    parser.add_argument("--live", action="store_true", help="카메라와 Arduino를 모두 실제 장치로 사용합니다.")
    parser.add_argument("--live-camera", action="store_true", help="카메라만 실제 장치로 사용합니다 (Arduino는 시뮬레이션).")
    parser.add_argument("--live-serial", action="store_true", help="Arduino만 실제 장치로 사용합니다 (카메라는 시뮬레이션).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config) if args.config else None
    config = load_config(config_path) if config_path else load_config()

    live_camera = args.live or args.live_camera
    live_serial = args.live or args.live_serial

    if live_serial and config.safety.safety_max_intensity <= 0:
        print(
            "[안내] safety.safety_max_intensity 가 0입니다. 실제 EMS 출력은 계속 차단된 상태로 실행됩니다.\n"
            "       config.json에서 담당자가 안전 상한값을 직접 설정한 뒤에만 실제 자극이 나갑니다."
        )

    worker = ControlLoopWorker(config, live_camera=live_camera, live_serial=live_serial)

    # Imported here (not at module top) so `python main.py` in a headless
    # test environment without a display never fails to import tkinter.
    from gui import App

    worker_thread = threading.Thread(target=worker.run, daemon=True)
    worker_thread.start()

    try:
        app = App(worker, config)
        app.mainloop()
    finally:
        worker.request_quit()
        worker_thread.join(timeout=3.0)


if __name__ == "__main__":
    main()
