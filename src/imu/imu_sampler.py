from src.imu.imu_reader import IMUReader
import time

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

print("Recording 100 samples (Raw YPR + Accel)...")

for i in range(100):
    imu_sample = imu.get_latest()

    if imu_sample is not None and isinstance(imu_sample, dict):
        imu_t = imu_sample.get("t_sec", float("nan"))
        imu_yaw = imu_sample.get("yaw_deg", float("nan"))
        imu_pitch = imu_sample.get("pitch_deg", float("nan"))
        imu_roll = imu_sample.get("roll_deg", float("nan"))

        # 'accel' contains the raw body acceleration (ax, ay, az)
        accel = imu_sample.get("accel", None)
    else:
        imu_t = imu_yaw = imu_pitch = imu_roll = float("nan")
        accel = None

    row = {
        "imu_t": imu_t,
        "imu_yaw_deg": imu_yaw,
        "imu_pitch_deg": imu_pitch,
        "imu_roll_deg": imu_roll,
        "accel": accel,
    }
    print(row)
    time.sleep(0.02)  # ~50 Hz read

imu.stop()