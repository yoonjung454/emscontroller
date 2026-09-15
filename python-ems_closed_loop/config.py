"""
config.py
---------
Central configuration for the "AI Vision-Feedback Closed-Loop EMS Hand Control"
prototype.

Everything a user should reasonably need to tune (control gains, presets,
safety limits, serial settings...) lives here as a JSON-backed dataclass tree.
No code elsewhere should hard-code a magic number that belongs here.

SAFETY NOTE (read this before changing anything):
    `SafetyConfig.safety_max_intensity` defaults to 0. This is intentional and
    is the single most important safety gate in the whole project: as long as
    it is 0, `safety_manager.SafetyManager` will refuse to arm the real EMS
    hardware no matter what the GUI, the controller, or a natural-language
    command asks for. A human must explicitly raise this value in config.json
    (or in the GUI, which writes it back here) after the safety upper bound
    has been decided under expert supervision. Nothing in this codebase will
    ever pick that number for you.
"""

from __future__ import annotations

import json
import typing
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Optional

PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = PROJECT_DIR / "config.json"


@dataclass
class CameraConfig:
    camera_index: int = 0
    frame_width: int = 640
    frame_height: int = 480
    # Which physical hand should be treated as the controlled ("actual") hand.
    # "Right" | "Left" | "Any" (Any = track whichever single hand appears first)
    preferred_hand: str = "Right"
    # We feed OpenCV's raw (non-mirrored) frame into MediaPipe and only mirror
    # the displayed image for the user's comfort. MediaPipe's handedness
    # classifier assumes a mirrored/selfie input, so vision_tracker.py corrects
    # for this automatically. If left/right ever comes out backwards on your
    # setup, flip this flag instead of touching the vision code.
    invert_handedness: bool = False


@dataclass
class HandTrackingConfig:
    model_path: str = "hand_landmarker.task"
    min_hand_detection_confidence: float = 0.7
    min_hand_presence_confidence: float = 0.7
    min_tracking_confidence: float = 0.7
    # Median filter window (frames) applied before the EMA filter.
    smoothing_window: int = 5
    # EMA smoothing factor applied after the median filter (0..1, higher = less smoothing).
    ema_alpha: float = 0.35
    # Consecutive frames with no hand detected before we treat the hand as
    # genuinely "lost" (vs. a single dropped frame, which we tolerate).
    hand_loss_frames_threshold: int = 10
    # Frames averaged during a flat/bent calibration capture.
    calibration_sample_count: int = 30
    # Minimum gap (in raw bend-angle-sum degrees) between the flat and bent
    # calibration values for the calibration to be considered valid.
    min_calibration_gap: float = 5.0


@dataclass
class CommandPresetsConfig:
    """Quick natural-language presets. Percent = target finger bend, 0-100."""
    light_percent: float = 30.0   # "살짝 쥐어" / "조금만 쥐어"
    half_percent: float = 60.0    # "반쯤 쥐어" / "반만 쥐어"
    strong_percent: float = 90.0  # "꽉 쥐어" / "최대한 쥐어"


@dataclass
class ControlConfig:
    # "proportional" = safety-limited step-wise P control (default, simplest).
    # "pid" = full PID with anti-windup, opt-in.
    mode: str = "proportional"
    kp: float = 0.6
    ki: float = 0.05
    kd: float = 0.02
    tolerance_percent: float = 5.0       # allowed error band counted as "on target"
    success_hold_seconds: float = 1.5    # must stay in-band this long to declare success
    max_step_up: float = 4.0             # max intensity increase per control tick
    max_step_down: float = 8.0           # max intensity decrease per control tick
    control_period_s: float = 0.5        # settling time between successive adjustments
    integral_limit: float = 50.0         # anti-windup clamp on the accumulated integral term


@dataclass
class SafetyConfig:
    # --- THE gate. Do not default this to anything but 0. ---
    safety_max_intensity: int = 0
    safety_min_intensity: int = 0

    max_continuous_stim_seconds: float = 10.0
    cooldown_seconds: float = 5.0
    total_experiment_seconds: float = 300.0

    heartbeat_interval_s: float = 0.4
    heartbeat_timeout_s: float = 1.5

    # Every SET command carries a time-to-live; Arduino auto-deactivates when
    # it elapses even if the PC never sends STOP. Mirrors the official
    # firmware's own 5000ms cap in EMSSystem::doActionCommand.
    command_ttl_ms: int = 1500
    max_command_ttl_ms: int = 5000


@dataclass
class SerialConfig:
    port: Optional[str] = None  # None => auto-detect on connect
    # Matches Serial.begin(19200) in the official arduino-openEMSstim.ino.
    # If you change Serial.begin(...) on the Arduino side, change this too.
    baud_rate: int = 19200
    connect_timeout_s: float = 2.0
    read_timeout_s: float = 0.2


@dataclass
class AdaptiveModelConfig:
    enabled: bool = True
    min_samples: int = 8
    degree: int = 1  # 1 = linear regression, 2 = quadratic
    model_path: str = "models/adaptive_model.json"


@dataclass
class LoggingConfig:
    log_dir: str = "logs"


@dataclass
class AppConfig:
    camera: CameraConfig = field(default_factory=CameraConfig)
    hand_tracking: HandTrackingConfig = field(default_factory=HandTrackingConfig)
    presets: CommandPresetsConfig = field(default_factory=CommandPresetsConfig)
    control: ControlConfig = field(default_factory=ControlConfig)
    safety: SafetyConfig = field(default_factory=SafetyConfig)
    serial: SerialConfig = field(default_factory=SerialConfig)
    adaptive_model: AdaptiveModelConfig = field(default_factory=AdaptiveModelConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)


def _merge_into_dataclass(cls, data: dict) -> Any:
    """Build a dataclass instance from a (possibly partial/outdated) dict,
    falling back to the dataclass defaults for anything missing or unknown.
    This makes config.json forward/backward compatible with new fields.

    Note: this file uses `from __future__ import annotations`, which means
    every dataclass field's `.type` is an unevaluated string (e.g.
    "AdaptiveModelConfig"), not the actual class -- `typing.get_type_hints`
    is what resolves those strings back to real types."""
    resolved_types = typing.get_type_hints(cls)
    kwargs = {}
    for f in fields(cls):
        if f.name not in data:
            continue
        value = data[f.name]
        field_type = resolved_types.get(f.name, f.type)
        if is_dataclass(field_type) and isinstance(value, dict):
            kwargs[f.name] = _merge_into_dataclass(field_type, value)
        else:
            kwargs[f.name] = value
    return cls(**kwargs)


def get_default_config() -> AppConfig:
    return AppConfig()


def load_config(path: Path = DEFAULT_CONFIG_PATH) -> AppConfig:
    """Load config.json if present, otherwise write and return the defaults.
    Any field missing from the file (new field added later, older file, etc.)
    silently falls back to its dataclass default instead of crashing."""
    path = Path(path)
    if not path.exists():
        cfg = get_default_config()
        save_config(cfg, path)
        return cfg

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        raise ValueError(
            f"설정 파일을 읽을 수 없습니다: {path}\n"
            f"JSON 형식이 올바른지 확인하세요. 오류: {exc}"
        ) from exc

    return _merge_into_dataclass(AppConfig, raw)


def save_config(config: AppConfig, path: Path = DEFAULT_CONFIG_PATH) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(config), indent=2, ensure_ascii=False), encoding="utf-8")


def resolve_path(relative: str) -> Path:
    """Resolve a config-relative path (e.g. 'models/adaptive_model.json')
    against the project directory, so the app works regardless of the
    current working directory it was launched from."""
    p = Path(relative)
    return p if p.is_absolute() else PROJECT_DIR / p
