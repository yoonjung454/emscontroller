"""
adaptive_model.py
-------------------
Optional personalization: different people's fingers curl by different
amounts at the same EMS control value. Instead of always starting a new
command from the same fixed intensity and waiting for the camera-feedback
loop to slowly climb there, this module fits a small, stable regression
model from past sessions' (ems_intensity -> achieved_bend_percent) pairs and
uses it to *guess a better starting intensity*.

It is explicitly NOT the controller. The closed loop in controller.py still
runs on top of whatever this suggests and will correct a bad guess using the
camera. This module's only job is to shorten "time to converge", never to
bypass the feedback loop.

Method: plain least-squares polynomial regression via numpy (degree 1 = linear,
degree 2 = quadratic -- both configurable). No scikit-learn dependency needed.
If there isn't enough data, or numpy can't fit a stable model, predict()
always falls back to a safe, low, config-defined default and the P/PID
controller does the rest.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from config import AdaptiveModelConfig
from safety_manager import SafetyManager


@dataclass
class FitStats:
    sample_count: int
    degree: int
    r_squared: Optional[float]


class AdaptivePersonalizationModel:
    def __init__(self, config: AdaptiveModelConfig, safety: SafetyManager):
        self.config = config
        self.safety = safety
        self._coeffs: Optional[np.ndarray] = None
        self._degree: int = config.degree
        self._sample_count: int = 0
        self._r_squared: Optional[float] = None

    @property
    def is_fitted(self) -> bool:
        return self._coeffs is not None and self._sample_count >= self.config.min_samples

    def fit(self, rows: list[dict]) -> FitStats:
        """rows: list of dicts as produced by data_logger.load_all_sessions().
        Only rows recorded while the controller had settled near/at the
        target (HOLDING or SUCCESS) are used, since only those pairs
        represent a steady-state (intensity -> achieved bend) relationship;
        rows captured mid-ramp would teach the model the wrong thing."""
        steady_rows = [r for r in rows if r["control_state"] in ("목표 유지 중", "성공", "HOLDING", "SUCCESS")]

        self._sample_count = len(steady_rows)
        if self._sample_count < self.config.min_samples:
            self._coeffs = None
            self._r_squared = None
            return FitStats(self._sample_count, self._degree, None)

        x = np.array([r["ems_intensity"] for r in steady_rows], dtype=float)
        y = np.array([r["current_percent"] for r in steady_rows], dtype=float)

        degree = max(1, min(2, self._degree))
        try:
            coeffs = np.polyfit(x, y, degree)
        except (np.linalg.LinAlgError, ValueError):
            self._coeffs = None
            self._r_squared = None
            return FitStats(self._sample_count, degree, None)

        predicted = np.polyval(coeffs, x)
        residual_ss = float(np.sum((y - predicted) ** 2))
        total_ss = float(np.sum((y - np.mean(y)) ** 2))
        r_squared = 1.0 - residual_ss / total_ss if total_ss > 1e-9 else None

        self._coeffs = coeffs
        self._degree = degree
        self._r_squared = r_squared
        return FitStats(self._sample_count, degree, r_squared)

    def suggest_initial_intensity(self, target_percent: float, fallback: Optional[int] = None) -> int:
        """Returns a safety-clamped starting intensity for a new target.
        Falls back to a low, safe default if the model isn't fitted yet."""
        safe_fallback = fallback if fallback is not None else self.safety.config.safety_min_intensity

        if not self.config.enabled or not self.is_fitted:
            return self.safety.clamp_intensity(safe_fallback)

        try:
            guess = self._invert(target_percent)
        except (ValueError, ArithmeticError):
            return self.safety.clamp_intensity(safe_fallback)

        if guess is None or not np.isfinite(guess):
            return self.safety.clamp_intensity(safe_fallback)

        return self.safety.clamp_intensity(guess)

    def _invert(self, target_percent: float) -> Optional[float]:
        """Solve coeffs(x) = target_percent for x, preferring a root inside
        [0, 100]."""
        assert self._coeffs is not None
        coeffs = self._coeffs

        if self._degree == 1:
            c1, c0 = coeffs
            if abs(c1) < 1e-9:
                return None
            return (target_percent - c0) / c1

        # degree == 2: c2*x^2 + c1*x + (c0 - target) = 0
        c2, c1, c0 = coeffs
        a, b, c = c2, c1, c0 - target_percent
        if abs(a) < 1e-9:
            if abs(b) < 1e-9:
                return None
            return -c / b

        discriminant = b * b - 4 * a * c
        if discriminant < 0:
            return None
        sqrt_d = discriminant ** 0.5
        root1 = (-b + sqrt_d) / (2 * a)
        root2 = (-b - sqrt_d) / (2 * a)

        candidates = [r for r in (root1, root2) if 0.0 <= r <= 100.0]
        if not candidates:
            return None
        # Prefer the candidate closest to the middle of the valid range --
        # avoids picking a spurious extreme root from a poorly-conditioned fit.
        return min(candidates, key=lambda r: abs(r - 50.0))

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "coeffs": None if self._coeffs is None else self._coeffs.tolist(),
            "degree": self._degree,
            "sample_count": self._sample_count,
            "r_squared": self._r_squared,
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def load(self, path: Path) -> bool:
        path = Path(path)
        if not path.exists():
            return False
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return False

        coeffs = payload.get("coeffs")
        self._coeffs = np.array(coeffs, dtype=float) if coeffs else None
        self._degree = payload.get("degree", self._degree)
        self._sample_count = payload.get("sample_count", 0)
        self._r_squared = payload.get("r_squared")
        return True
