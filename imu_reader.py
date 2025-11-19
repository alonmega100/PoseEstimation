#!/usr/bin/env python3
import serial
import time
import threading
from typing import Optional, Tuple
from tools import make_vn_cmd, parse_vn_vnrrg_08

# def make_vn_cmd(body: str) -> bytes:
#     cs = 0
#     for b in body.encode("ascii"):
#         cs ^= b
#     return f"${body}*{cs:02X}\r\n".encode("ascii")
#
#
# def parse_vn_vnrrg_08(line: str) -> Optional[Tuple[float, float, float]]:
#     if not line.startswith("$VNRRG,08"):
#         return None
#     try:
#         data_part = line.split("*", 1)[0]
#         parts = data_part.split(",")
#         if len(parts) < 5:
#             return None
#         yaw = float(parts[2])
#         pitch = float(parts[3])
#         roll = float(parts[4])
#         return yaw, pitch, roll
#     except Exception:
#         return None


class IMUReader:
    """
    Background thread that polls VN-100T (VNRRG,8) at a fixed rate
    and keeps the latest sample.
    """

    def __init__(self, port: str = "/dev/ttyUSB0", baud: int = 115200, rate_hz: float = 50.0):
        self.port = port
        self.baud = baud
        self.period = 1.0 / rate_hz

        self._ser: Optional[serial.Serial] = None
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

        self._lock = threading.Lock()
        self._latest_sample = None  # (t_sec, yaw, pitch, roll) or None
        self._t0 = None

    def start(self):
        if self._thread is not None and self._thread.is_alive():
            return

        self._stop_event.clear()

        self._ser = serial.Serial(
            self.port,
            self.baud,
            timeout=1.0,
            bytesize=serial.EIGHTBITS,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
        )

        self._t0 = time.time()

        self._thread = threading.Thread(target=self._run, name="IMUReader", daemon=True)
        self._thread.start()

    def _run(self):
        poll_cmd = make_vn_cmd("VNRRG,8")
        while not self._stop_event.is_set():
            t_loop = time.time()

            try:
                self._ser.write(poll_cmd)
                raw = self._ser.readline()
            except Exception:
                # If something goes wrong, you could add logging here
                continue

            if raw:
                try:
                    line = raw.decode("ascii", errors="replace").strip()
                    ypr = parse_vn_vnrrg_08(line)
                    if ypr is not None:
                        yaw, pitch, roll = ypr
                        t_rel = time.time() - self._t0
                        with self._lock:
                            self._latest_sample = (t_rel, yaw, pitch, roll)
                except Exception:
                    pass

            # enforce rate
            dt = time.time() - t_loop
            sleep_time = self.period - dt
            if sleep_time > 0:
                time.sleep(sleep_time)

    def get_latest(self) -> Optional[Tuple[float, float, float, float]]:
        """
        Returns (t_sec, yaw_deg, pitch_deg, roll_deg) or None if nothing yet.
        """
        with self._lock:
            return self._latest_sample

    def stop(self):
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
        if self._ser is not None:
            try:
                self._ser.close()
            except Exception:
                pass
            self._ser = None
