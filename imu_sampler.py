from imu_reader import IMUReader
import time
import math  # for math.isnan if you want

print("boop")
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
    if imu_sample is not None:
        imu_t, imu_yaw, imu_pitch, imu_roll = imu_sample
    else:
        imu_t = imu_yaw = imu_pitch = imu_roll = float("nan")

    row = {
        "imu_t": imu_t,
        "imu_yaw_deg": imu_yaw,
        "imu_pitch_deg": imu_pitch,
        "imu_roll_deg": imu_roll,
    }
    print(row)
    time.sleep(0.02)  # ~50 Hz read, matches IMU poll rate

imu.stop()
