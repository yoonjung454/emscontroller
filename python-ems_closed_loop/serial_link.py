"""
serial_link.py
----------------
USB-serial link to the Arduino Nano running
arduino/openemsstim_serial_controller.ino.

Text protocol (see that .ino and README.md for the full spec):

  PC -> Arduino:  PING | ARM,<ch> | SET,<ch>,<intensity0-100>,<duration_ms> |
                  STOP,<ch> | STOP_ALL | STATUS
  Arduino -> PC:  PONG | ARMED,<ch> | OK,<ch> | STOPPED,<ch> | STATUS,... |
                  ERROR,<reason>

Design notes:
  - A dedicated reader thread continuously reads newline-terminated lines
    into a queue.Queue. `_send_and_wait()` sends a command and then pulls
    from that queue (with a deadline) looking for the expected reply prefix,
    forwarding anything else to the `on_message` callback. This keeps the
    protocol simple and synchronous for callers (controller/main loop), while
    still being safe to call from a background worker thread -- never call
    these methods from the Tkinter GUI thread directly for anything that
    might block; main.py runs the whole control loop in its own thread.
  - A second heartbeat thread sends PING at a fixed interval so the Arduino's
    own watchdog (see the .ino) sees a live PC on the other end and the PC
    side can independently notice a silently-dead connection.
  - Every public method that talks to hardware is safe to call even when not
    connected: they return False / raise nothing, and set `self.last_error`.
"""

from __future__ import annotations

import queue
import threading
import time
from typing import Callable, Optional

import serial
import serial.tools.list_ports


class SerialLink:
    def __init__(
        self,
        baud_rate: int = 19200,
        read_timeout_s: float = 0.2,
        heartbeat_interval_s: float = 0.4,
        on_message: Optional[Callable[[str], None]] = None,
        on_disconnected: Optional[Callable[[str], None]] = None,
    ):
        self.baud_rate = baud_rate
        self.read_timeout_s = read_timeout_s
        self.heartbeat_interval_s = heartbeat_interval_s
        self.on_message = on_message
        self.on_disconnected = on_disconnected

        self._serial: Optional[serial.Serial] = None
        self._reader_thread: Optional[threading.Thread] = None
        self._heartbeat_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

        self._rx_queue: "queue.Queue[str]" = queue.Queue()
        self._write_lock = threading.Lock()

        self._handshake_ok = False
        self._last_pong_monotonic: Optional[float] = None
        self.last_error: str = ""

    # ------------------------------------------------------------------ #
    # Discovery / connection
    # ------------------------------------------------------------------ #

    @staticmethod
    def list_ports() -> list[tuple[str, str]]:
        """Returns [(device, description), ...] for the port-picker GUI."""
        return [(p.device, p.description) for p in serial.tools.list_ports.comports()]

    @staticmethod
    def _guess_port() -> Optional[str]:
        """Best-effort auto-detect: prefer a port whose description mentions
        the CH340 USB-serial chip used on most Arduino Nano clones."""
        ports = list(serial.tools.list_ports.comports())
        for p in ports:
            if "ch340" in (p.description or "").lower():
                return p.device
        return ports[0].device if ports else None

    def connect(self, port: Optional[str] = None, connect_timeout_s: float = 2.0) -> bool:
        port = port or self._guess_port()
        if not port:
            self.last_error = "사용 가능한 시리얼 포트를 찾지 못했습니다. USB 연결을 확인하세요."
            return False

        try:
            self._serial = serial.Serial(port=port, baudrate=self.baud_rate, timeout=self.read_timeout_s)
        except serial.SerialException as exc:
            self.last_error = f"포트 {port} 를 열지 못했습니다: {exc}"
            self._serial = None
            return False

        self._stop_event.clear()
        self._handshake_ok = False

        self._reader_thread = threading.Thread(target=self._reader_loop, daemon=True)
        self._reader_thread.start()

        # Give the Arduino's auto-reset-on-connect a moment before the first PING.
        time.sleep(1.0)

        reply = self._send_and_wait("PING", expect_prefix="PONG", timeout=connect_timeout_s)
        if reply is None:
            self.last_error = f"{port} 에 연결했지만 PONG 응답이 없습니다 (핸드셰이크 실패)."
            self.disconnect()
            return False

        self._handshake_ok = True
        self._last_pong_monotonic = time.monotonic()

        self._heartbeat_thread = threading.Thread(target=self._heartbeat_loop, daemon=True)
        self._heartbeat_thread.start()
        return True

    def disconnect(self) -> None:
        self._stop_event.set()
        self._handshake_ok = False
        if self._serial is not None:
            try:
                self._serial.close()
            except serial.SerialException:
                pass
        self._serial = None

    def is_connected(self) -> bool:
        return self._serial is not None and self._serial.is_open

    def handshake_ok(self) -> bool:
        return self._handshake_ok and self.is_connected()

    def is_alive(self, timeout_s: float) -> bool:
        """True if we've heard a PONG within `timeout_s`. Used by
        safety_manager/main.py to detect a silently-dead link (cable pulled,
        Arduino reset, etc.) even though the OS-level serial port object
        might still report itself as "open"."""
        if not self.is_connected() or self._last_pong_monotonic is None:
            return False
        return (time.monotonic() - self._last_pong_monotonic) <= timeout_s

    # ------------------------------------------------------------------ #
    # Protocol commands
    # ------------------------------------------------------------------ #

    def ping(self) -> bool:
        reply = self._send_and_wait("PING", expect_prefix="PONG", timeout=0.5)
        return reply is not None

    def arm(self, channel: int) -> bool:
        reply = self._send_and_wait(f"ARM,{channel}", expect_prefix=("ARMED", "ERROR"), timeout=1.0)
        return reply is not None and reply.startswith("ARMED")

    def set_intensity(self, channel: int, intensity: int, duration_ms: int) -> bool:
        intensity = max(0, min(100, int(intensity)))
        duration_ms = max(0, int(duration_ms))
        cmd = f"SET,{channel},{intensity},{duration_ms}"
        reply = self._send_and_wait(cmd, expect_prefix=("OK", "ERROR"), timeout=1.0)
        if reply is None:
            self.last_error = "SET 명령에 대한 응답이 없습니다 (연결을 확인하세요)."
            return False
        if reply.startswith("ERROR"):
            self.last_error = reply
            return False
        return True

    def stop(self, channel: int) -> bool:
        reply = self._send_and_wait(f"STOP,{channel}", expect_prefix=("STOPPED", "ERROR"), timeout=1.0)
        return reply is not None and reply.startswith("STOPPED")

    def stop_all(self) -> bool:
        """Best-effort: always attempts the send even if we're not sure the
        link is healthy, because this is the emergency-stop path -- a failed
        write here should not raise and block the rest of the shutdown."""
        try:
            reply = self._send_and_wait("STOP_ALL", expect_prefix=("STOPPED", "ERROR"), timeout=1.0)
            return reply is not None and reply.startswith("STOPPED")
        except Exception:
            return False

    def status(self) -> Optional[str]:
        return self._send_and_wait("STATUS", expect_prefix="STATUS", timeout=1.0)

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #

    def _send_and_wait(
        self,
        command: str,
        expect_prefix,
        timeout: float,
    ) -> Optional[str]:
        if self._serial is None or not self._serial.is_open:
            self.last_error = "시리얼 포트가 연결되어 있지 않습니다."
            return None

        prefixes = (expect_prefix,) if isinstance(expect_prefix, str) else tuple(expect_prefix)

        with self._write_lock:
            try:
                self._serial.write((command + "\n").encode("ascii"))
            except serial.SerialException as exc:
                self.last_error = f"쓰기 실패: {exc}"
                self._handle_disconnect(str(exc))
                return None

        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            remaining = max(0.0, deadline - time.monotonic())
            try:
                line = self._rx_queue.get(timeout=remaining)
            except queue.Empty:
                return None

            if line.startswith(prefixes):
                if line.startswith("PONG"):
                    self._last_pong_monotonic = time.monotonic()
                return line

            if line.startswith("PONG"):
                self._last_pong_monotonic = time.monotonic()
            # Not the reply we're waiting for (e.g. an unsolicited line) --
            # forward it and keep waiting for our expected prefix.
            if self.on_message:
                try:
                    self.on_message(line)
                except Exception:
                    pass

        return None

    def _reader_loop(self) -> None:
        while not self._stop_event.is_set():
            ser = self._serial
            if ser is None:
                break
            try:
                raw = ser.readline()
            except serial.SerialException as exc:
                self._handle_disconnect(str(exc))
                return

            if not raw:
                continue  # just a read timeout, normal
            try:
                line = raw.decode("ascii", errors="replace").strip()
            except UnicodeDecodeError:
                continue
            if line:
                self._rx_queue.put(line)

    def _heartbeat_loop(self) -> None:
        while not self._stop_event.is_set():
            time.sleep(self.heartbeat_interval_s)
            if self._stop_event.is_set():
                break
            if self._serial is None or not self._serial.is_open:
                continue
            with self._write_lock:
                try:
                    self._serial.write(b"PING\n")
                except serial.SerialException as exc:
                    self._handle_disconnect(str(exc))
                    return

    def _handle_disconnect(self, reason: str) -> None:
        self.last_error = reason
        self.disconnect()
        if self.on_disconnected:
            try:
                self.on_disconnected(reason)
            except Exception:
                pass
