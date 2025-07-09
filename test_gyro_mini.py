import numpy as np
from mpu6050 import mpu6050
import time


DT = 20


sensor = mpu6050(0x68)

next_time = time.time() + DT / 1000.0
counter = 0

while True:
	if time.time() >= next_time:
		next_time += DT / 1000.0

		# Read gyro data
		current_accel = sensor.get_accel_data()
		ax =+ current_accel['x']
		ay =+ current_accel['y']
		az =+ current_accel['z']
		counter += 1
		if counter == 10:
			print(f"Gyro: gx={ax}, gy={ay}, gz={az}")
			ax, ay, az = 0, 0, 0
			counter = 0


