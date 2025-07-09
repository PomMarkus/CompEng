import numpy as np
from mpu6050 import mpu6050
import time


DT = 100


sensor = mpu6050(0x68)

next_time = time.time() + DT / 1000.0
counter = 0
g_arr=[[], [], []]  # Initialize arrays for gx, gy, gz

while True:
	if time.time() >= next_time:
		next_time += DT / 1000.0

		# Read gyro data
		current_gyro = sensor.get_gyro_data()
		gx =+ current_gyro['x']
		gy =+ current_gyro['y']
		gz =+ current_gyro['z']
		counter += 1
		if counter == 10:
			print(f"Gyro: gx={gx}, gy={gy}, gz={gz}")
			gx, gy, gz = 0, 0, 0
			counter = 0


