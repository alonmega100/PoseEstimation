from imu_reader import IMUReader
import time
import math

imu = IMUReader(port="/dev/ttyUSB0", rate_hz=50.0)
imu.start()

# Wait until we actually have a sample (with a timeout)
timeout_s = 2.0
t0 = time.time()
while True:
    sample = imu.get_latest()
    if sample is not None:
        break
    if time.time() - t0 > timeout_s:
        print("No IMU data after 2s")
        break
    time.sleep(0.01)

for i in range(100):
    imu_sample = imu.get_latest()
    if imu_sample is not None and isinstance(imu_sample, dict):
        imu_t = imu_sample.get("t_sec", float("nan"))
        imu_yaw = imu_sample.get("yaw_deg", float("nan"))
        imu_pitch = imu_sample.get("pitch_deg", float("nan"))
        imu_roll = imu_sample.get("roll_deg", float("nan"))
        acc = imu_sample.get("acc_world_m_s2", None)            # (ax,ay,az) in world frame or None
        vel = imu_sample.get("vel_m_s", (float("nan"),)*3)     # (vx,vy,vz)
        pos = imu_sample.get("pos_m", (float("nan"),)*3)       # (x,y,z)
    else:
        imu_t = imu_yaw = imu_pitch = imu_roll = float("nan")
        acc = None
        vel = pos = (float("nan"),)*3

    row = {
        "imu_t": imu_t,
        "imu_yaw_deg": imu_yaw,
        "imu_pitch_deg": imu_pitch,
        "imu_roll_deg": imu_roll,
        "acc_world_m_s2": acc,
        "vel_m_s": vel,
        "pos_m": pos,
    }
    print(row)
    time.sleep(0.02)  # ~50 Hz read, matches IMU poll rate

imu.stop()
