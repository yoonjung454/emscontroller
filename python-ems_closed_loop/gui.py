"""
gui.py
-------
Tkinter GUI (chosen over PySide6 for this project because it ships with
every standard Windows Python install -- nothing extra to get working
before a demo).

Threading model: ControlLoopWorker (main.py) runs entirely in its own
background thread and only ever exposes a `Snapshot` copy through
`worker.get_snapshot()`, which is safe to call from any thread. This GUI
polls that snapshot on a Tkinter `after()` timer (i.e., on the main thread)
and only ever touches Tkinter widgets from the main thread. User actions
(typing a command, pressing a button) call thread-safe `worker.submit_*` /
`worker.request_*` methods that just stash a value for the worker thread to
pick up on its next tick -- the GUI never blocks waiting on the worker.
"""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk, messagebox
from typing import Optional

import cv2
import matplotlib
from PIL import Image, ImageTk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from config import AppConfig
from main import ControlLoopWorker
from serial_link import SerialLink

# matplotlib's default font (DejaVu Sans) has no Hangul glyphs, so every
# Korean axis label/legend entry would otherwise render as a missing-glyph
# box. Malgun Gothic ships with every Windows install since Vista, so this is
# safe to hard-code without adding a font file to the repo. unicode_minus
# must be turned off too -- Malgun Gothic doesn't have the Unicode minus
# glyph matplotlib uses by default, which would otherwise turn every negative
# number on an axis back into a missing-glyph box.
matplotlib.rcParams["font.family"] = "Malgun Gothic"
matplotlib.rcParams["axes.unicode_minus"] = False

POLL_MS = 66          # ~15 Hz GUI refresh (video + numbers)
GRAPH_POLL_MS = 250   # graphs redraw less often -- matplotlib redraw is comparatively expensive


class App(tk.Tk):
    def __init__(self, worker: ControlLoopWorker, config: AppConfig):
        super().__init__()
        self.worker = worker
        self.config = config

        self.title("AI 비전 피드백 기반 폐루프 EMS 손동작 제어 시스템")
        self.geometry("1360x800")
        self.minsize(1200, 700)
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self.bind("<space>", lambda e: self._emergency_stop())
        self.bind("<Escape>", lambda e: self._emergency_stop())

        self._video_photo: Optional[ImageTk.PhotoImage] = None
        self._build_layout()

        if worker.live_serial:
            self._refresh_ports()

        self.after(POLL_MS, self._poll)
        self.after(GRAPH_POLL_MS, self._poll_graphs)

    # ------------------------------------------------------------------ #
    # Layout
    # ------------------------------------------------------------------ #

    def _build_layout(self) -> None:
        root = ttk.Frame(self, padding=8)
        root.pack(fill="both", expand=True)

        left = ttk.Frame(root)
        left.pack(side="left", fill="both", expand=True)
        right = ttk.Frame(root, width=380)
        right.pack(side="right", fill="y")

        self._build_video_panel(left)
        self._build_graph_panel(left)

        self._build_mode_panel(right)
        self._build_command_panel(right)
        self._build_calibration_panel(right)
        self._build_readout_panel(right)
        self._build_estop_panel(right)

    def _build_video_panel(self, parent) -> None:
        frame = ttk.LabelFrame(parent, text="카메라 영상 (MediaPipe 손 관절)")
        frame.pack(fill="x", pady=(0, 8))
        self.video_label = ttk.Label(frame)
        self.video_label.pack()

    def _build_graph_panel(self, parent) -> None:
        frame = ttk.LabelFrame(parent, text="실시간 그래프")
        frame.pack(fill="both", expand=True)

        self.figure = Figure(figsize=(6.0, 4.2), dpi=90)
        self.ax_percent = self.figure.add_subplot(211)
        self.ax_intensity = self.figure.add_subplot(212)
        self.figure.tight_layout()

        self.canvas = FigureCanvasTkAgg(self.figure, master=frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

    def _build_mode_panel(self, parent) -> None:
        frame = ttk.LabelFrame(parent, text="모드 / 연결")
        frame.pack(fill="x", pady=(0, 8))

        mode_text = []
        mode_text.append("카메라: 실제 장치" if self.worker.live_camera else "카메라: 시뮬레이션")
        mode_text.append("Arduino: 실제 장치" if self.worker.live_serial else "Arduino: 시뮬레이션")
        self.mode_label = ttk.Label(frame, text=" / ".join(mode_text), foreground="#0a5")
        self.mode_label.pack(anchor="w", padx=6, pady=2)

        if self.worker.live_serial:
            row = ttk.Frame(frame)
            row.pack(fill="x", padx=6, pady=2)
            self.port_combo = ttk.Combobox(row, width=18, state="readonly")
            self.port_combo.pack(side="left")
            ttk.Button(row, text="새로고침", command=self._refresh_ports).pack(side="left", padx=4)
            ttk.Button(row, text="연결", command=self._connect_serial).pack(side="left", padx=2)
            ttk.Button(row, text="연결 해제", command=self._disconnect_serial).pack(side="left", padx=2)
        else:
            ttk.Label(frame, text="(시뮬레이션 모드에서는 시리얼 연결이 필요 없습니다)").pack(anchor="w", padx=6)

        self.serial_status_label = ttk.Label(frame, text="")
        self.serial_status_label.pack(anchor="w", padx=6, pady=(0, 4))

        self.csv_status_label = ttk.Label(frame, text="CSV 기록: -")
        self.csv_status_label.pack(anchor="w", padx=6, pady=(0, 4))

    def _build_command_panel(self, parent) -> None:
        frame = ttk.LabelFrame(parent, text="자연어 명령")
        frame.pack(fill="x", pady=(0, 8))

        row = ttk.Frame(frame)
        row.pack(fill="x", padx=6, pady=4)
        self.command_entry = ttk.Entry(row)
        self.command_entry.pack(side="left", fill="x", expand=True)
        self.command_entry.bind("<Return>", lambda e: self._submit_command())
        ttk.Button(row, text="실행", command=self._submit_command).pack(side="left", padx=4)

        presets = ttk.Frame(frame)
        presets.pack(fill="x", padx=6, pady=(0, 6))
        ttk.Button(
            presets, text=f"{self.config.presets.light_percent:.0f}%",
            command=lambda: self._quick_percent(self.config.presets.light_percent)
        ).pack(side="left", expand=True, fill="x", padx=2)
        ttk.Button(
            presets, text=f"{self.config.presets.half_percent:.0f}%",
            command=lambda: self._quick_percent(self.config.presets.half_percent)
        ).pack(side="left", expand=True, fill="x", padx=2)
        ttk.Button(
            presets, text=f"{self.config.presets.strong_percent:.0f}%",
            command=lambda: self._quick_percent(self.config.presets.strong_percent)
        ).pack(side="left", expand=True, fill="x", padx=2)

        self.command_message_label = ttk.Label(frame, text="", wraplength=340, foreground="#555")
        self.command_message_label.pack(anchor="w", padx=6, pady=(0, 6))

    def _build_calibration_panel(self, parent) -> None:
        frame = ttk.LabelFrame(parent, text="캘리브레이션 (손가락 굽힘 기준값)")
        frame.pack(fill="x", pady=(0, 8))

        row = ttk.Frame(frame)
        row.pack(fill="x", padx=6, pady=4)
        ttk.Button(row, text="① 펴짐 캘리브레이션", command=lambda: self._calibrate("flat")).pack(
            side="left", expand=True, fill="x", padx=2
        )
        ttk.Button(row, text="② 구부림 캘리브레이션", command=lambda: self._calibrate("bent")).pack(
            side="left", expand=True, fill="x", padx=2
        )
        self.calibration_status_label = ttk.Label(frame, text="캘리브레이션 필요")
        self.calibration_status_label.pack(anchor="w", padx=6, pady=(0, 4))

    def _build_readout_panel(self, parent) -> None:
        frame = ttk.LabelFrame(parent, text="현재 상태")
        frame.pack(fill="x", pady=(0, 8))

        self.readout_labels: dict[str, ttk.Label] = {}
        rows = [
            ("target", "목표 굽힘값"),
            ("current", "현재 굽힘값"),
            ("intensity", "EMS 제어값 (0~100, mA 아님)"),
            ("error", "오차"),
            ("state", "판단 상태"),
            ("elapsed", "경과 시간"),
        ]
        for key, label in rows:
            r = ttk.Frame(frame)
            r.pack(fill="x", padx=6, pady=1)
            ttk.Label(r, text=label + ":", width=24).pack(side="left")
            value_label = ttk.Label(r, text="-", font=("Segoe UI", 10, "bold"))
            value_label.pack(side="left")
            self.readout_labels[key] = value_label

    def _build_estop_panel(self, parent) -> None:
        frame = ttk.Frame(parent)
        frame.pack(fill="x", pady=(4, 0))
        self.estop_button = tk.Button(
            frame,
            text="■ 비상 정지 (SPACE / ESC)",
            command=self._emergency_stop,
            bg="#c0392b",
            fg="white",
            activebackground="#a93226",
            font=("Segoe UI", 14, "bold"),
            height=2,
        )
        self.estop_button.pack(fill="x")

    # ------------------------------------------------------------------ #
    # Actions
    # ------------------------------------------------------------------ #

    def _submit_command(self) -> None:
        text = self.command_entry.get().strip()
        if not text:
            return
        self.worker.submit_command_text(text)
        self.command_entry.delete(0, tk.END)

    def _quick_percent(self, percent: float) -> None:
        self.worker.submit_quick_percent(percent)

    def _calibrate(self, mode: str) -> None:
        self.worker.request_calibration(mode)

    def _emergency_stop(self) -> None:
        self.worker.emergency_stop("사용자 비상정지 버튼/키")

    def _refresh_ports(self) -> None:
        ports = SerialLink.list_ports()
        values = [f"{dev} ({desc})" for dev, desc in ports]
        self.port_combo["values"] = values
        if values:
            self.port_combo.current(0)

    def _connect_serial(self) -> None:
        selection = self.port_combo.get()
        port = selection.split(" ")[0] if selection else None
        ok = self.worker.connect_serial(port)
        if not ok:
            messagebox.showerror("연결 실패", "Arduino에 연결하지 못했습니다. 포트/드라이버(CH340)를 확인하세요.")

    def _disconnect_serial(self) -> None:
        if hasattr(self.worker.stim, "disconnect"):
            self.worker.stim.disconnect()

    def _on_close(self) -> None:
        self.worker.emergency_stop("창 닫힘")
        self.worker.request_quit()
        self.destroy()

    # ------------------------------------------------------------------ #
    # Polling
    # ------------------------------------------------------------------ #

    def _poll(self) -> None:
        snap = self.worker.get_snapshot()

        if snap.quit_requested:
            self.destroy()
            return

        if snap.frame_bgr is not None:
            rgb = cv2.cvtColor(snap.frame_bgr, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(rgb)
            self._video_photo = ImageTk.PhotoImage(image=image)
            self.video_label.configure(image=self._video_photo)

        self.readout_labels["target"].configure(
            text="-" if snap.target_percent is None else f"{snap.target_percent:.0f}%"
        )
        self.readout_labels["current"].configure(
            text="-" if snap.current_percent is None else f"{snap.current_percent:.0f}%"
        )
        self.readout_labels["intensity"].configure(text=str(snap.ems_intensity))
        self.readout_labels["error"].configure(
            text="-" if snap.error_percent is None else f"{snap.error_percent:+.0f}%"
        )
        self.readout_labels["state"].configure(text=snap.control_state)
        self.readout_labels["elapsed"].configure(text=f"{snap.elapsed_s:.0f}s")

        if snap.calibration_active:
            done, total = snap.calibration_progress
            self.calibration_status_label.configure(text=f"측정 중... {done}/{total}")
        elif snap.is_calibrated:
            self.calibration_status_label.configure(text="캘리브레이션 완료", foreground="#0a5")
        else:
            self.calibration_status_label.configure(text="캘리브레이션 필요", foreground="#c0392b")

        if snap.live_serial:
            status = "연결됨" if snap.serial_connected else "연결 안 됨"
            if snap.serial_connected:
                status += " / handshake " + ("OK" if snap.serial_handshake else "대기중")
            if snap.serial_error:
                status += f" ({snap.serial_error})"
            self.serial_status_label.configure(text=status)

        self.csv_status_label.configure(
            text=f"CSV 기록 중: {self.worker.data_logger.current_path}"
            if self.worker.data_logger.current_path
            else "CSV 기록: -"
        )

        if snap.last_message:
            self.command_message_label.configure(text=snap.last_message)

        self.after(POLL_MS, self._poll)

    def _poll_graphs(self) -> None:
        snap = self.worker.get_snapshot()
        history = snap.history

        self.ax_percent.clear()
        self.ax_intensity.clear()

        if history:
            t0 = history[0][0]
            ts = [h[0] - t0 for h in history]
            targets = [h[1] for h in history]
            currents = [h[2] for h in history]
            intensities = [h[3] for h in history]

            self.ax_percent.plot(ts, targets, label="목표 %", color="#2b6cb0")
            self.ax_percent.plot(ts, currents, label="실제 %", color="#c0392b")
            self.ax_percent.set_ylim(-5, 105)
            self.ax_percent.set_ylabel("굽힘 %")
            self.ax_percent.legend(loc="upper left", fontsize=8)

            self.ax_intensity.plot(ts, intensities, color="#8e44ad")
            self.ax_intensity.set_ylim(-5, 105)
            self.ax_intensity.set_ylabel("EMS 제어값")
            self.ax_intensity.set_xlabel("시간 (s)")

        self.canvas.draw_idle()
        self.after(GRAPH_POLL_MS, self._poll_graphs)
