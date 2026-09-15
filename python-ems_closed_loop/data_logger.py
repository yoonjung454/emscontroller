"""
data_logger.py
----------------
Writes one CSV row per control tick to `logs/session_<timestamp>.csv`, and
can reload every past session for adaptive_model.py to train on.

Columns (fixed order, header written once per file):
    timestamp, target_percent, current_percent, ems_intensity, error,
    control_state, success, hand_detected
"""

from __future__ import annotations

import csv
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

CSV_FIELDS = [
    "timestamp",
    "target_percent",
    "current_percent",
    "ems_intensity",
    "error",
    "control_state",
    "success",
    "hand_detected",
]


@dataclass
class LogRow:
    timestamp: float
    target_percent: Optional[float]
    current_percent: Optional[float]
    ems_intensity: int
    error: Optional[float]
    control_state: str
    success: bool
    hand_detected: bool

    def as_csv_dict(self) -> dict:
        return {
            "timestamp": f"{self.timestamp:.3f}",
            "target_percent": "" if self.target_percent is None else f"{self.target_percent:.2f}",
            "current_percent": "" if self.current_percent is None else f"{self.current_percent:.2f}",
            "ems_intensity": self.ems_intensity,
            "error": "" if self.error is None else f"{self.error:.2f}",
            "control_state": self.control_state,
            "success": int(self.success),
            "hand_detected": int(self.hand_detected),
        }


class DataLogger:
    def __init__(self, log_dir: Path):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._file = None
        self._writer: Optional[csv.DictWriter] = None
        self.current_path: Optional[Path] = None

    def start_session(self, session_name: Optional[str] = None) -> Path:
        self.close()
        stamp = session_name or time.strftime("%Y%m%d_%H%M%S")
        path = self.log_dir / f"session_{stamp}.csv"
        self._file = open(path, "w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._file, fieldnames=CSV_FIELDS)
        self._writer.writeheader()
        self.current_path = path
        return path

    def log_row(self, row: LogRow) -> None:
        if self._writer is None:
            self.start_session()
        self._writer.writerow(row.as_csv_dict())
        self._file.flush()

    def close(self) -> None:
        if self._file is not None:
            try:
                self._file.close()
            except OSError:
                pass
        self._file = None
        self._writer = None

    # ------------------------------------------------------------------ #
    # Historical data for adaptive_model.py
    # ------------------------------------------------------------------ #

    def load_all_sessions(self) -> list[dict]:
        """Read every session_*.csv in log_dir and return the rows as plain
        dicts with numeric fields converted back to float/int/bool. Rows with
        missing target/current values (hand not detected, etc.) are skipped
        since they carry no (intensity -> bend) training signal."""
        rows: list[dict] = []
        for csv_path in sorted(self.log_dir.glob("session_*.csv")):
            try:
                with open(csv_path, "r", newline="", encoding="utf-8") as f:
                    for raw in csv.DictReader(f):
                        try:
                            if raw["target_percent"] == "" or raw["current_percent"] == "":
                                continue
                            rows.append(
                                {
                                    "timestamp": float(raw["timestamp"]),
                                    "target_percent": float(raw["target_percent"]),
                                    "current_percent": float(raw["current_percent"]),
                                    "ems_intensity": int(raw["ems_intensity"]),
                                    "error": float(raw["error"]) if raw["error"] != "" else None,
                                    "control_state": raw["control_state"],
                                    "success": bool(int(raw["success"])),
                                    "hand_detected": bool(int(raw["hand_detected"])),
                                }
                            )
                        except (ValueError, KeyError):
                            continue  # skip malformed row, keep loading the rest
            except OSError:
                continue
        return rows
