import tkinter as tk
import numpy as np
from mpu6050 import mpu6050
import time

RADIUS = 10
HEIGHT = 480
WIDTH = 800
DT = 50
ACC_SCALE = 10

sensor = mpu6050(0x68)
pos = [WIDTH/2, HEIGHT/2]
vel = [0, 0]

window = tk.Tk()
window.title("Game map")
window.geometry("800x480")
window.attributes('-fullscreen', True)
window.focus_force()

canvas = tk.Canvas(window, width=window.winfo_screenwidth(), height=window.winfo_screenheight(), bg="white")
canvas.pack(fill=tk.BOTH, expand=True)

ball = canvas.create_oval(pos[0] - RADIUS, pos[1] - RADIUS, pos[0] + RADIUS, pos[1] + RADIUS, fill="blue", outline="blue")
#text = canvas.create_text(5, HEIGHT - 5, text = f"{pos[0]}, {pos[1]}", anchor = "sw", font = ("Arial", 12))

def go_fullscreen():
    window.attributes('-fullscreen', True)


def end_fullscreen(event=None):
    window.attributes('-fullscreen', False)    

window.bind("<Escape>", end_fullscreen)


def update_pos():
	global pos
	current_acc = sensor.get_accel_data()
	ax = current_acc['x']
	ay = current_acc['y']
	vel[0] += ACC_SCALE * ax * DT / 1000
	vel[1] += ACC_SCALE * ay * DT / 1000
	pos[0] += vel[0] * DT / 1000
	pos[1] += vel[1] * DT / 1000
	if (pos[0] < RADIUS):
		pos[0] = RADIUS
		vel[0] = - vel[0] * 0.8
	elif (pos[0] > WIDTH - RADIUS):
		pos[0] = WIDTH - RADIUS
		vel[0] = - vel[0] * 0.8
		
	if (pos[1] < RADIUS):
		pos[1] = RADIUS
		vel[1] = - vel[1] * 0.8
	elif (pos[1] > HEIGHT - RADIUS):
		pos[1] = HEIGHT - RADIUS
		vel[1] = - vel[1] * 0.8
	
	canvas.coords(ball, pos[0] - RADIUS, pos[1] - RADIUS, pos[0] + RADIUS, pos[1] + RADIUS)
	#canvas.itemconfigure(text, text = f"{pos[0]}, {pos[1]}")
	window.after(DT, update_pos)
	
window.after(DT, update_pos)
window.after(100, go_fullscreen)

window.mainloop()
	
	# print(sensor.get_accel_data(), end="\r")
	# time.sleep(0.2)
