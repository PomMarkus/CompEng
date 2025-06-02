import tkinter as tk
import numpy as np
from mpu6050 import mpu6050
import time

RADIUS = 10
HEIGHT = 480
WIDTH = 800
DT = 50
DP = 0.1
STARTX = 60
STARTY = 100

ACC_SCALE = 10
DAMPING = 0.8

# =============== Import and process objects from file ===============

FILENAME = "objects.dat"
RADIUS = 10
HEIGHT = 480
WIDTH = 800

objects = []

with open(FILENAME, "r") as f:
    for line in f:
        objects.append(line.strip().split("\t"))

# 2D np array
val_data = np.zeros((HEIGHT, WIDTH, 3))

# Fill area of objects
for obj in objects:
    if obj[0] == 'r':
        # Rectangle coordinates: (x1, y1, x2, y2)
        x1, y1, x2, y2 = int(obj[1]), int(obj[2]), int(obj[3]), int(obj[4])
        val_data[y1:y2, x1:x2, 0] = 2 # pixle occupied

    elif obj[0] == 'o':
        # Oval "corner" coordinates (x1, y1, x2, y2)
        x1, y1, x2, y2 = int(obj[1]), int(obj[2]), int(obj[3]), int(obj[4])
        # Create a mask for the oval
        Y, X = np.ogrid[:HEIGHT, :WIDTH]
        mask = ((X - (x1 + x2) / 2) ** 2) / ((x2 - x1) / 2) ** 2 + ((Y - (y1 + y2) / 2) ** 2) / ((y2 - y1) / 2) ** 2 <= 1
        # Set the pixels inside the oval to 1
        val_data[mask, 0] = -1 # pixle marked as hole

# Fill area of shifted rectangles
for obj in objects:
    if obj[0] == 'r':
        # Rectangle coordinates: (x1, y1, x2, y2)
        x1, y1, x2, y2 = int(obj[1]), int(obj[2]), int(obj[3]), int(obj[4])
        Y, X = np.ogrid[:HEIGHT, :WIDTH]
        # Create rectangle above
        subdata = val_data[y1-RADIUS:y1, x1:x2]
        subdata[subdata[:,:, 0] == 0] = np.array([1, 0, 0])
        subdata[subdata[:,:, 0] != 2] += np.array([0, -1, 0])
        val_data[y1-RADIUS:y1, x1:x2] = subdata
        # Create rectangle below
        subdata = val_data[y2:y2+RADIUS, x1:x2]
        subdata[subdata[:,:, 0] == 0] = np.array([1, 0, 0])
        subdata[subdata[:,:, 0] != 2] += np.array([0, 1, 0])
        val_data[y2:y2+RADIUS, x1:x2] = subdata
        # Create rectangle left
        subdata = val_data[y1:y2, x1-RADIUS:x1]
        subdata[subdata[:,:, 0] == 0] = np.array([1, 0, 0])
        subdata[subdata[:,:, 0] != 2] += np.array([0, 0, -1])
        val_data[y1:y2, x1-RADIUS:x1] = subdata
        # Create rectangle right
        subdata = val_data[y1:y2, x2:x2+RADIUS]
        subdata[subdata[:,:, 0] == 0] = np.array([1, 0, 0])
        subdata[subdata[:,:, 0] != 2] += np.array([0, 0, 1])
        val_data[y1:y2, x2:x2+RADIUS] = subdata

# Create circle 
circle = np.zeros((2*RADIUS + 1, 2*RADIUS + 1))
Y, X = np.ogrid[:2*RADIUS + 1, :2*RADIUS + 1]
mask = ((X - RADIUS) ** 2) + ((Y - RADIUS) ** 2) <= RADIUS ** 2
circle[mask] = 1 # pixle occupied
idx = np.indices((2*RADIUS + 1, 2*RADIUS + 1))
circle = np.stack((circle, (idx[0] - RADIUS) * circle / RADIUS, (idx[1]- RADIUS) * circle / RADIUS), axis=-1)

# Fill cirle areas at the corners
for obj in objects:
    if obj[0] == 'r':
        # Rectangle coordinates: (x1, y1, x2, y2)
        x1, y1, x2, y2 = int(obj[1]), int(obj[2]), int(obj[3]), int(obj[4])

        # Insert circle segments
        # Top left corner
        mask = val_data[y1-RADIUS:y1, x1-RADIUS:x1, 0] == 0
        val_data[y1-RADIUS:y1, x1-RADIUS:x1][mask] = circle[:RADIUS, :RADIUS][mask]
        # Top right corner
        mask = val_data[y1-RADIUS:y1, x2:x2+RADIUS, 0] == 0
        val_data[y1-RADIUS:y1, x2:x2+RADIUS][mask] = circle[:RADIUS, RADIUS + 1:][mask]
        # Bottom left corner
        mask = val_data[y2:y2+RADIUS, x1-RADIUS:x1, 0] == 0
        val_data[y2:y2+RADIUS, x1-RADIUS:x1][mask] = circle[RADIUS + 1:, :RADIUS][mask]
        # Bottom right corner
        mask = val_data[y2:y2+RADIUS, x2:x2+RADIUS, 0] == 0
        val_data[y2:y2+RADIUS, x2:x2+RADIUS][mask] = circle[RADIUS + 1:, RADIUS + 1:][mask]

# ====================================================================

sensor = mpu6050(0x68)

window = tk.Tk()
window.title("Game map")
window.geometry("800x480")
window.attributes('-fullscreen', True)
window.focus_force()

canvas = tk.Canvas(window, width=window.winfo_screenwidth(), height=window.winfo_screenheight(), bg="white")
canvas.pack(fill=tk.BOTH, expand=True)

for obj in objects:
    if obj[0] == 'r':
        # Draw a rectangle (x1, y1, x2, y2)
        canvas.create_rectangle(int(obj[1]), int(obj[2]), int(obj[3]), int(obj[4]), fill="black", outline="black")
    elif obj[0] == 'o':
        # Draw an oval (x1, y1, x2, y2)
        canvas.create_oval(int(obj[1]), int(obj[2]), int(obj[3]), int(obj[4]), fill="black", outline="black")


pos = np.array([STARTX, STARTY], dtype=float)
vel = [0, 0]

ball = canvas.create_oval(int(pos[0]) - RADIUS, int(pos[1]) - RADIUS, int(pos[0]) + RADIUS, int(pos[1]) + RADIUS, fill="blue", outline="blue")

def update_pos():
    window.after(DT, update_pos)
    global pos, vel

    current_acc = sensor.get_accel_data()
    ax, ay = current_acc['x'], current_acc['y']
    vel[0] += ACC_SCALE * ax * DT / 1000
    vel[1] += ACC_SCALE * ay * DT / 1000
    Dpos = np.array(vel) * DT / 1000
    dist = np.linalg.norm(Dpos)
    steps = int(dist / DP) if dist > DP else 1
    
    dstep = Dpos / steps
    counter = 0

    while counter < steps:
        temp_pos = pos + dstep
            
        if (val_data[int(temp_pos[1]), int(temp_pos[0]), 0] < 0):
            #hole - pull ball to center - vector to center in val_data
            pass
        elif (val_data[int(temp_pos[1]), int(temp_pos[0]), 0] > 0):
            vec_norm = val_data[int(temp_pos[1]), int(temp_pos[0]), 1:3]
            vec_proj = np.dot(vec_norm, Dpos) / np.dot(vec_norm, vec_norm) * vec_norm
            Dpos = 2 * vec_proj - Dpos
            Dpos *= (steps - counter) / (steps)
            Dpos *= (1 - DAMPING)
            dist = np.linalg.norm(Dpos)
            steps = int(dist / DP) if dist > DP else 1
            dstep = Dpos / steps
            counter = 0
            continue
        
        pos += dstep
        counter += 1
            

    canvas.coords(ball, int(pos[0]) - RADIUS, int(pos[1]) - RADIUS, int(pos[0]) + RADIUS, int(pos[1]) + RADIUS)

def go_fullscreen():
    window.attributes('-fullscreen', True)


def end_fullscreen(event=None):
    window.attributes('-fullscreen', False)    

window.bind("<Escape>", end_fullscreen)

window.after(DT, update_pos)
window.after(100, go_fullscreen)

window.mainloop()