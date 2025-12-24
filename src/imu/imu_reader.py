#!/usr/bin/env python3
import serial
import time
import threading
from typing import Optional, Tuple, Dict, Any
from src.utils.tools import make_vn_cmd, parse_vn_vnrrg_08


class IMUReader:
    """
    Background thread that polls VN-100T (VNRRG,27) for:
      - Yaw, Pitch, Roll (Degrees)
      - Accelerometer (Body Frame, m/s^2)

    NO integration, NO gravity compensation, NO world frame conversion.
    """

    def __init__(self, port: str = "/dev/ttyUSB0", baud: int = 115200, rate_hz: float = 50.0):
        self.port = port
        self.baud = baud
        self.period = 1.0 / rate_hz

        self._ser: Optional[serial.Serial] = None
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

        self._lock = threading.Lock()
        self._latest_sample: Optional[Dict[str, Any]] = None
        self._t0 = None

    def start(self):
        if self._thread is not None and self._thread.is_alive():
            return

        self._stop_event.clear()

        try:
            self._ser = serial.Serial(
                self.port,
                self.baud,
                timeout=1.0,
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
            )
        except Exception as e:
            print(f"IMUReader Error opening serial port: {e}")
            return

        self._t0 = time.time()
        self._thread = threading.Thread(target=self._run, name="IMUReader", daemon=True)
        self._thread.start()

    def _run(self):
        # Poll Yaw, Pitch, Roll, Magnetic, Accel, Gyro (register 27, YMR)
        # Note: Ensure parse_vn_vnrrg_08 or equivalent handles the specific fields returned by reg 27
        # or that you are using the correct command for your parsing logic.
        # Assuming existing 'make_vn_cmd' and 'parse_vn_vnrrg_08' work with the device configuration.
        poll_cmd = make_vn_cmd("VNRRG,27")

        while not self._stop_event.is_set():
            t_loop = time.time()

            try:
                self._ser.write(poll_cmd)
                raw = self._ser.readline()
            except Exception:
                continue

            if raw:
                try:
                    line = raw.decode("ascii", errors="replace").strip()
                    parsed = parse_vn_vnrrg_08(line)

                    if parsed is not None:
                        # Expecting: (yaw, pitch, roll, ax, ay, az, ...)
                        # If the parser returns a list/tuple
                        yaw, pitch, roll = 0.0, 0.0, 0.0
                        accel = None

                        if isinstance(parsed, (list, tuple)):
                            if len(parsed) >= 6:
                                yaw, pitch, roll, ax, ay, az = parsed[:6]
                                accel = (float(ax), float(ay), float(az))
                            elif len(parsed) >= 3:
                                yaw, pitch, roll = parsed[:3]
                                accel = None  # No acceleration data found

                        t_rel = time.time() - self._t0

                        sample = {
                            "t_sec": t_rel,
                            "yaw_deg": float(yaw),
                            "pitch_deg": float(pitch),
                            "roll_deg": float(roll),
                            "accel": accel,  # (ax, ay, az) raw body frame
                        }

                        with self._lock:
                            self._latest_sample = sample

                except Exception:
                    pass

            # enforce rate
            dt = time.time() - t_loop
            sleep_time = self.period - dt
            if sleep_time > 0:
                time.sleep(sleep_time)

    def get_latest(self) -> Optional[Dict[str, Any]]:
        """
        Returns a dict with keys:
        {
            "t_sec": float,
            "yaw_deg": float,
            "pitch_deg": float,
            "roll_deg": float,
            "accel": (ax, ay, az) or None  # Raw body frame acceleration
        }
        or None if nothing yet.
        """
        with self._lock:
            return None if self._latest_sample is None else dict(self._latest_sample)

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