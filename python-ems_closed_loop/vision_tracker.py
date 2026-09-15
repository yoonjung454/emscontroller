"""
vision_tracker.py
--------------------
Camera + MediaPipe Hand Landmarker wrapper. Turns a webcam frame into:
  - an annotated BGR frame (skeleton drawn, for the GUI to display)
  - whether the configured hand (Left/Right/Any) was seen this frame
  - a calibrated 0-100% finger-bend reading for that hand

Bend is computed from the MCP/PIP/DIP joint angles of the four fingers the
project spec asks for (index, middle, ring, little -- thumb excluded), using
3D "world landmarks" (metric, camera-distance independent) so the joint
angles themselves are already scale/distance normalized -- no separate
palm-size normalization is needed on top of that.

Calibration ("펴짐" / "구부림"): the caller starts a calibration capture with
`begin_calibration("flat"|"bent")`; over the next `calibration_sample_count`
frames where the target hand is visible, the raw bend metric is averaged and
stored. `get_bend_percent()` then linearly maps the live (filtered) raw bend
between those two anchors into 0-100, clamped.

Filtering: a rolling median (drops single-frame spikes) feeding an EMA
(smooths remaining jitter), per the project spec.

Hand-loss handling: a momentary miss (a few frames) does not by itself count
as "lost" -- the last filtered reading is held. Only `hand_loss_frames_threshold`
*consecutive* missed frames sets `hand_lost = True`, which is what
safety_manager treats as a reason to stop. This deliberately favors safety
over responsiveness: a real, sustained loss of the hand always wins.
"""

from __future__ import annotations

import statistics
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import mediapipe as mp
import numpy as np

from config import CameraConfig, HandTrackingConfig

# 21-point connections, drawn for the full hand (thumb included) purely for
# visual feedback -- the bend *calculation* below intentionally excludes it.
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]

# Bend is computed from these four fingers only (thumb excluded per spec).
# Each triple is (A, B, C) where the joint angle is measured at B.
FINGERS: dict[str, list[tuple[int, int, int]]] = {
    "Index": [(0, 5, 6), (5, 6, 7), (6, 7, 8)],
    "Middle": [(0, 9, 10), (9, 10, 11), (10, 11, 12)],
    "Ring": [(0, 13, 14), (13, 14, 15), (14, 15, 16)],
    "Little": [(0, 17, 18), (17, 18, 19), (18, 19, 20)],
}


def _calculate_joint_angle(a, b, c) -> float:
    v1 = np.array([a.x - b.x, a.y - b.y, a.z - b.z])
    v2 = np.array([c.x - b.x, c.y - b.y, c.z - b.z])
    denom = np.linalg.norm(v1) * np.linalg.norm(v2)
    if denom == 0:
        return 180.0
    cosine = np.clip(np.dot(v1, v2) / denom, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def _calculate_finger_bend(world_landmarks, joints: list[tuple[int, int, int]]) -> float:
    total = 0.0
    for a_idx, b_idx, c_idx in joints:
        angle = _calculate_joint_angle(world_landmarks[a_idx], world_landmarks[b_idx], world_landmarks[c_idx])
        total += max(0.0, 180.0 - angle)
    return min(total, 270.0)  # 3 joints * 90 deg max useful bend each, clamp for safety


class MedianEmaFilter:
    """Rolling median (spike rejection) feeding an EMA (jitter smoothing)."""

    def __init__(self, median_window: int, ema_alpha: float):
        self.median_window = max(1, median_window)
        self.ema_alpha = ema_alpha
        self._buffer: deque[float] = deque(maxlen=self.median_window)
        self._ema: Optional[float] = None

    def push(self, value: float) -> float:
        self._buffer.append(value)
        median_value = statistics.median(self._buffer)
        if self._ema is None:
            self._ema = median_value
        else:
            self._ema = self.ema_alpha * median_value + (1 - self.ema_alpha) * self._ema
        return self._ema

    def reset(self) -> None:
        self._buffer.clear()
        self._ema = None


@dataclass
class FrameResult:
    annotated_frame: Optional[np.ndarray]
    hand_detected_now: bool
    hand_lost: bool
    raw_bend: Optional[float]
    bend_percent: Optional[float]
    is_calibrated: bool
    calibration_active: bool
    calibration_progress: tuple[int, int]


class VisionTracker:
    def __init__(self, camera_cfg: CameraConfig, tracking_cfg: HandTrackingConfig, model_path: Path):
        self.camera_cfg = camera_cfg
        self.tracking_cfg = tracking_cfg
        self.model_path = Path(model_path)

        self._cap: Optional[cv2.VideoCapture] = None
        self._landmarker = None
        self._start_time: Optional[float] = None

        self._filter = MedianEmaFilter(tracking_cfg.smoothing_window, tracking_cfg.ema_alpha)
        self._consecutive_miss = 0
        self._last_bend_percent: Optional[float] = None

        self.calibration: dict[str, Optional[float]] = {"flat": None, "bent": None}
        self._calibration_mode: Optional[str] = None
        self._calibration_sum = 0.0
        self._calibration_count = 0

    # ------------------------------------------------------------------ #

    def start(self) -> None:
        if not self.model_path.exists():
            raise RuntimeError(
                f"MediaPipe 모델 파일을 찾을 수 없습니다: {self.model_path}\n"
                "hand_landmarker.task 파일이 이 프로젝트 폴더에 있는지 확인하세요."
            )

        self._cap = cv2.VideoCapture(self.camera_cfg.camera_index)
        if not self._cap.isOpened():
            self._cap = None
            raise RuntimeError(
                f"카메라(index={self.camera_cfg.camera_index})를 열 수 없습니다. "
                "다른 프로그램이 카메라를 사용 중인지, 카메라 인덱스가 맞는지 확인하세요."
            )
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.camera_cfg.frame_width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.camera_cfg.frame_height)

        base_options = mp.tasks.BaseOptions(model_asset_path=str(self.model_path))
        options = mp.tasks.vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_hands=2,  # detect up to 2 so we can reject a bystander/off-target hand by handedness
            min_hand_detection_confidence=self.tracking_cfg.min_hand_detection_confidence,
            min_hand_presence_confidence=self.tracking_cfg.min_hand_presence_confidence,
            min_tracking_confidence=self.tracking_cfg.min_tracking_confidence,
        )
        self._landmarker = mp.tasks.vision.HandLandmarker.create_from_options(options)
        self._start_time = time.monotonic()

    def stop(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        if self._landmarker is not None:
            self._landmarker.close()
            self._landmarker = None

    # ------------------------------------------------------------------ #
    # Calibration
    # ------------------------------------------------------------------ #

    def begin_calibration(self, mode: str) -> None:
        if mode not in ("flat", "bent"):
            raise ValueError("mode must be 'flat' or 'bent'")
        self._calibration_mode = mode
        self._calibration_sum = 0.0
        self._calibration_count = 0

    def reset_calibration(self) -> None:
        self.calibration = {"flat": None, "bent": None}
        self._calibration_mode = None

    def is_calibrated(self) -> bool:
        flat = self.calibration["flat"]
        bent = self.calibration["bent"]
        if flat is None or bent is None:
            return False
        return abs(bent - flat) >= self.tracking_cfg.min_calibration_gap

    # ------------------------------------------------------------------ #
    # Per-frame processing
    # ------------------------------------------------------------------ #

    def process_frame(self) -> FrameResult:
        if self._cap is None or self._landmarker is None:
            raise RuntimeError("VisionTracker.start()를 먼저 호출해야 합니다.")

        ok, frame = self._cap.read()
        if not ok:
            self._consecutive_miss += 1
            return self._miss_result(annotated_frame=None)

        # Flip for a natural selfie-view display. MediaPipe's handedness
        # classifier assumes a mirrored/selfie input, so this flip is done
        # *before* detection -- handedness labels then match the physical
        # hand directly (see config.camera.invert_handedness for the manual
        # override if your setup ever disagrees).
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        timestamp_ms = int((time.monotonic() - self._start_time) * 1000)

        result = self._landmarker.detect_for_video(mp_image, timestamp_ms)

        chosen_index = self._select_hand_index(result)
        if chosen_index is None:
            self._consecutive_miss += 1
            return self._miss_result(annotated_frame=frame)

        self._consecutive_miss = 0
        image_landmarks = result.hand_landmarks[chosen_index]
        world_landmarks = result.hand_world_landmarks[chosen_index]

        self._draw_skeleton(frame, image_landmarks)

        raw_bend = self._average_finger_bend(world_landmarks)
        smoothed_raw = self._filter.push(raw_bend)

        if self._calibration_mode is not None:
            self._calibration_sum += raw_bend
            self._calibration_count += 1
            if self._calibration_count >= self.tracking_cfg.calibration_sample_count:
                self.calibration[self._calibration_mode] = self._calibration_sum / self._calibration_count
                self._calibration_mode = None

        bend_percent = self._to_percent(smoothed_raw)
        if bend_percent is not None:
            self._last_bend_percent = bend_percent

        return FrameResult(
            annotated_frame=frame,
            hand_detected_now=True,
            hand_lost=False,
            raw_bend=raw_bend,
            bend_percent=bend_percent,
            is_calibrated=self.is_calibrated(),
            calibration_active=self._calibration_mode is not None,
            calibration_progress=(self._calibration_count, self.tracking_cfg.calibration_sample_count),
        )

    def _miss_result(self, annotated_frame: Optional[np.ndarray]) -> FrameResult:
        hand_lost = self._consecutive_miss >= self.tracking_cfg.hand_loss_frames_threshold
        # Hold the last known reading through a momentary miss, but report
        # None once truly lost -- callers must treat None as "do not act".
        bend_percent = None if hand_lost else self._last_bend_percent
        return FrameResult(
            annotated_frame=annotated_frame,
            hand_detected_now=False,
            hand_lost=hand_lost,
            raw_bend=None,
            bend_percent=bend_percent,
            is_calibrated=self.is_calibrated(),
            calibration_active=self._calibration_mode is not None,
            calibration_progress=(self._calibration_count, self.tracking_cfg.calibration_sample_count),
        )

    # ------------------------------------------------------------------ #

    def _select_hand_index(self, result) -> Optional[int]:
        if not result.hand_landmarks:
            return None
        if self.camera_cfg.preferred_hand == "Any":
            return 0

        for i, handedness in enumerate(result.handedness):
            label = handedness[0].category_name  # "Left" | "Right"
            if self.camera_cfg.invert_handedness:
                label = "Right" if label == "Left" else "Left"
            if label == self.camera_cfg.preferred_hand:
                return i
        return None

    def _average_finger_bend(self, world_landmarks) -> float:
        bends = [_calculate_finger_bend(world_landmarks, joints) for joints in FINGERS.values()]
        return sum(bends) / len(bends)

    def _to_percent(self, smoothed_raw: float) -> Optional[float]:
        if not self.is_calibrated():
            return None
        flat = self.calibration["flat"]
        bent = self.calibration["bent"]
        percent = (smoothed_raw - flat) / (bent - flat) * 100.0
        return max(0.0, min(100.0, percent))

    @staticmethod
    def _draw_skeleton(frame: np.ndarray, landmarks) -> None:
        h, w = frame.shape[:2]
        points = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
        for start, end in HAND_CONNECTIONS:
            cv2.line(frame, points[start], points[end], (255, 120, 0), 2)
        for x, y in points:
            cv2.circle(frame, (x, y), 4, (0, 255, 0), -1)
