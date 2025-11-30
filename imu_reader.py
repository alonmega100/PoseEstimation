#!/usr/bin/env python3
import serial
import time
import threading
from typing import Optional, Tuple, Dict, Any
from tools import make_vn_cmd, parse_vn_vnrrg_08
import math

class IMUReader:
    """
    Background thread that polls VN-100T (VNRRG,8) at a fixed rate
    and keeps the latest sample. Now also accepts linear accelerations
    (if present in the VNRRG,8 message), converts them to the world
    frame, removes gravity (assumes world Z is up) and integrates
    to estimate velocity and position (simple trapezoidal integration).

    Assumptions:
    - parse_vn_vnrrg_08(line) may return either:
        (yaw_deg, pitch_deg, roll_deg)
      or
        (yaw_deg, pitch_deg, roll_deg, ax, ay, az)
      where ax,ay,az are accelerometer specific force in m/s^2 in body frame.
    - Angles are in degrees. World frame is Z-up. Gravity = +9.80665 m/s^2 along +Z.
    """

    def __init__(self, port: str = "/dev/ttyUSB0", baud: int = 115200, rate_hz: float = 50.0):
        self.port = port
        self.baud = baud
        self.period = 1.0 / rate_hz

        self._ser: Optional[serial.Serial] = None
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

        self._lock = threading.Lock()
        # latest sample as dict or None
        self._latest_sample: Optional[Dict[str, Any]] = None
        self._t0 = None

        # integration state
        self._last_sample_time: Optional[float] = None  # relative time in seconds
        self._last_acc_world: Optional[Tuple[float, float, float]] = None
        self._vel = (0.0, 0.0, 0.0)  # vx, vy, vz in world frame (m/s)
        self._pos = (0.0, 0.0, 0.0)  # x, y, z in world frame (m)

        # gravity (z-up)
        self._g = (0.0, 0.0, 9.80665)

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
        self._last_sample_time = None
        self._last_acc_world = None
        self._vel = (0.0, 0.0, 0.0)
        self._pos = (0.0, 0.0, 0.0)

        self._thread = threading.Thread(target=self._run, name="IMUReader", daemon=True)
        self._thread.start()

    def _body_to_world(self, yaw_deg: float, pitch_deg: float, roll_deg: float, v: Tuple[float, float, float]) -> Tuple[float, float, float]:
        """
        Convert a vector v from body frame to world frame using Z(yaw)-Y(pitch)-X(roll) rotations.
        Angles in degrees.
        """
        yaw = math.radians(yaw_deg)
        pitch = math.radians(pitch_deg)
        roll = math.radians(roll_deg)

        cy = math.cos(yaw); sy = math.sin(yaw)
        cp = math.cos(pitch); sp = math.sin(pitch)
        cr = math.cos(roll); sr = math.sin(roll)

        # Rotation matrix R = Rz(yaw) * Ry(pitch) * Rx(roll)
        # R * v_body = v_world
        R11 = cy * cp
        R12 = cy * sp * sr - sy * cr
        R13 = cy * sp * cr + sy * sr

        R21 = sy * cp
        R22 = sy * sp * sr + cy * cr
        R23 = sy * sp * cr - cy * sr

        R31 = -sp
        R32 = cp * sr
        R33 = cp * cr

        bx, by, bz = v
        wx = R11 * bx + R12 * by + R13 * bz
        wy = R21 * bx + R22 * by + R23 * bz
        wz = R31 * bx + R32 * by + R33 * bz
        return (wx, wy, wz)

    def _add_vec(self, a: Tuple[float, float, float], b: Tuple[float, float, float]) -> Tuple[float, float, float]:
        return (a[0]+b[0], a[1]+b[1], a[2]+b[2])

    def _scale_vec(self, a: Tuple[float, float, float], s: float) -> Tuple[float, float, float]:
        return (a[0]*s, a[1]*s, a[2]*s)

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
                    parsed = parse_vn_vnrrg_08(line)
                    if parsed is not None:
                        # handle either (yaw,pitch,roll) or (yaw,pitch,roll,ax,ay,az,...)
                        if isinstance(parsed, (list, tuple)):
                            if len(parsed) >= 6:
                                yaw, pitch, roll, ax, ay, az = parsed[:6]
                                acc_body = (float(ax), float(ay), float(az))
                            else:
                                yaw, pitch, roll = parsed[:3]
                                acc_body = None
                        else:
                            # unexpected type; skip
                            continue

                        t_rel = time.time() - self._t0

                        # integration if accelerations exist
                        if acc_body is not None:
                            # convert specific force (body) to world and add gravity to get linear acceleration
                            acc_world = self._body_to_world(yaw, pitch, roll, acc_body)
                            # sensors typically measure specific force f = a - g -> a = f + g
                            acc_world = self._add_vec(acc_world, self._g)  # a_world = R*f_body + g_world

                            # integrate velocity and position (trapezoidal)
                            if self._last_sample_time is not None:
                                dt = t_rel - self._last_sample_time
                                if dt > 0:
                                    if self._last_acc_world is None:
                                        # first accel -> simple Euler
                                        vx = self._vel[0] + acc_world[0] * dt
                                        vy = self._vel[1] + acc_world[1] * dt
                                        vz = self._vel[2] + acc_world[2] * dt
                                        # position update
                                        x = self._pos[0] + vx * dt
                                        y = self._pos[1] + vy * dt
                                        z = self._pos[2] + vz * dt
                                    else:
                                        # trapezoidal integration for velocity
                                        avg_ax = 0.5 * (self._last_acc_world[0] + acc_world[0])
                                        avg_ay = 0.5 * (self._last_acc_world[1] + acc_world[1])
                                        avg_az = 0.5 * (self._last_acc_world[2] + acc_world[2])
                                        vx = self._vel[0] + avg_ax * dt
                                        vy = self._vel[1] + avg_ay * dt
                                        vz = self._vel[2] + avg_az * dt
                                        # trapezoidal position update using avg velocity
                                        avg_vx = 0.5 * (self._vel[0] + vx)
                                        avg_vy = 0.5 * (self._vel[1] + vy)
                                        avg_vz = 0.5 * (self._vel[2] + vz)
                                        x = self._pos[0] + avg_vx * dt
                                        y = self._pos[1] + avg_vy * dt
                                        z = self._pos[2] + avg_vz * dt
                                    self._vel = (vx, vy, vz)
                                    self._pos = (x, y, z)
                            # update last accel and time
                            self._last_acc_world = acc_world
                        else:
                            # no accel available for this sample
                            acc_world = None

                        # update times and latest sample
                        self._last_sample_time = t_rel
                        sample = {
                            "t_sec": t_rel,
                            "yaw_deg": float(yaw),
                            "pitch_deg": float(pitch),
                            "roll_deg": float(roll),
                            "acc_world_m_s2": acc_world,  # None or (ax,ay,az)
                            "vel_m_s": self._vel,
                            "pos_m": self._pos,
                        }
                        with self._lock:
                            self._latest_sample = sample
                except Exception:
                    # ignore parse / conversion errors; could log
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
            "acc_world_m_s2": (ax,ay,az) or None,
            "vel_m_s": (vx,vy,vz),
            "pos_m": (x,y,z)
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

def process_imu_sample(imu_sample):
    # python
    if isinstance(imu_sample, dict):
        imu_t = imu_sample.get("t_sec")
        imu_yaw = imu_sample.get("yaw_deg")
        imu_pitch = imu_sample.get("pitch_deg")
        imu_roll = imu_sample.get("roll_deg")
        # optional accelerations if present:
        acc = imu_sample.get("acc_world_m_s2")
        if acc is not None:
            ax, ay, az = acc
    else:
        # tuple/list: accept extra fields
        if len(imu_sample) >= 4:
            imu_t, imu_yaw, imu_pitch, imu_roll, *rest = imu_sample
            if len(rest) >= 3:
                ax, ay, az = rest[:3]
        else:
            raise ValueError("imu_sample has fewer than 4 elements")
