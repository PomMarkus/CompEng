#  TODO:
#  - define Radien in file

import json
import tkinter as tk
import numpy as np

with open("config.json") as f:
    config = json.load(f)

control_mode = config.get("control", "keyboard")  # Default to keyboard if not specified

if control_mode not in ["keyboard", "mpu6050"]:
    raise ValueError(f"Invalid control mode: {control_mode}. Choose 'keyboard' or 'mpu6050'.")

elif control_mode == "keyboard":
    def on_key_press(event):
        pressed_keys.add(event.keysym)

    def on_key_release(event):
        pressed_keys.discard(event.keysym)

    def get_acceleration():
        if "Left" in pressed_keys:
            ax= -10
        elif "Right" in pressed_keys:
            ax= 10
        else:
            ax = 0
            
        if "Down" in pressed_keys:
            ay= 10
        elif "Up" in pressed_keys:
            ay=-10
        else:
            ay = 0
        return ax, ay
    
elif control_mode == "mpu6050":
    from mpu6050 import mpu6050

    sensor = mpu6050(0x68)

    def get_acceleration():
        current_acc = sensor.get_accel_data()
        return - current_acc['x'], current_acc['y']
        

# import matplotlib.pyplot as plt	
# 
# import time

HEIGHT = 480
WIDTH = 800
DT = 20
DP = 0.1
ACC_SCALE = 60
DAMPING = 0.8
# VELTHRESHOLD = 1

# =============== Import and process objects from file ===============

FILENAME = "map_v1.dat"
RADIUS = 10
HOLERADIUS = 12
HEIGHT = 480
WIDTH = 800

objects = []
start_point = np.array([0, 0], dtype=float)
pressed_keys = set()


with open(FILENAME, "r") as f:
    for line in f:
        line = line.strip()
        if not line or line.startswith("#"):
            pass
        elif line.startswith("s"):
            start_point = np.array([int(line.split("\t")[1]), int(line.split("\t")[2])], dtype=float)
        else: 
            objects.append(line.split("\t"))


# 2D np array
val_data = np.zeros((HEIGHT, WIDTH, 3))

# Create template mask for circle for the holes
circle_hole = np.zeros((HOLERADIUS * 2, HOLERADIUS * 2))
Y, X = np.ogrid[:HOLERADIUS * 2, :HOLERADIUS * 2]
mask_hole = ((X - HOLERADIUS + 0.5) ** 2) + ((Y - HOLERADIUS + 0.5) ** 2) <= HOLERADIUS ** 2
circle_hole[mask_hole] = -1 # corner surroundings of walls are also marked as such

# generating a 2D array e.g. (-2, -1, 0, 1, 2) for x and y
idx = np.indices((HOLERADIUS * 2, HOLERADIUS * 2)) - HOLERADIUS
idx[idx >=0] += 1
circle_hole = np.stack((circle_hole, (idx[0]) * circle_hole / RADIUS, (idx[1]) * circle_hole / RADIUS), axis=-1)

norm_squared = np.sum(circle_hole[:, :, 1:3] ** 2, axis=-1, keepdims=True)
norm_squared[norm_squared == 0] = 1e-1
circle_hole[:, :, 1:3] /= norm_squared**2

inner_mask = ((X - HOLERADIUS + 0.5) ** 2) + ((Y - HOLERADIUS + 0.5) ** 2) <= (HOLERADIUS - RADIUS) ** 2
circle_hole[inner_mask, 0] = -2


# Fill area of circles
for obj in objects:
    # Wall pxl to 2
    if obj[0] == 'w':
        x1, y1, x2, y2 = int(obj[1]), int(obj[2]), int(obj[3]), int(obj[4])
        val_data[y1:y2, x1:x2, 0] = 2 # pixle occupied: 2

    # hole pxl to -1 
    elif obj[0] == 'h': # maybe remove this because of redundancy
        x, y = int(obj[1]), int(obj[2])
        
        subdata = val_data[y - HOLERADIUS:y + HOLERADIUS, x - HOLERADIUS:x + HOLERADIUS]
        subdata[mask_hole, :] = circle_hole[mask_hole, :]
        val_data[y - HOLERADIUS:y + HOLERADIUS, x - HOLERADIUS:x + HOLERADIUS] = subdata # fill the hole with the circle_hole template
    
    # Checkpoint pxl to -2 
    # elif obj[0] == 'c':
    #     x, y = int(obj[1]), int(obj[2])
    #     # Create a mask for defining the pxl inside the circle
    #     Y, X = np.ogrid[:HEIGHT, :WIDTH]
    #     mask_checkpoint = ((X - x ** 2) / x ** 2) + ((Y - y) ** 2) / (y) ** 2 <= 1
    #     # Set the pixels inside the checkoint to -3
    #     val_data[mask_checkpoint, 0] = -3
        

# Fill area of shifted rectangles with 1 and add the normalvector for the rebouncing calculation
for obj in objects:

    if obj[0] == 'w':
        x1, y1, x2, y2 = int(obj[1]), int(obj[2]), int(obj[3]), int(obj[4])
        Y, X = np.ogrid[:HEIGHT, :WIDTH]

        # Create rectangle above - normalvector points upwards
        subdata = val_data[y1-RADIUS:y1, x1:x2]
        subdata[subdata[:,:, 0] <= 0] = np.array([1, 0, 0]) # making sure, that the area around a wall element is marked as surrounding area only once
        subdata[subdata[:,:, 0] != 2] += np.array([0, -1, 0]) # add the normalvector
        val_data[y1-RADIUS:y1, x1:x2] = subdata

        # Create rectangle below - normalvector points downwards
        subdata = val_data[y2:y2+RADIUS, x1:x2]
        subdata[subdata[:,:, 0] <= 0] = np.array([1, 0, 0]) # making sure, that the area around a wall element is marked as surrounding area only once
        subdata[subdata[:,:, 0] != 2] += np.array([0, 1, 0]) # add the normalvector
        val_data[y2:y2+RADIUS, x1:x2] = subdata

        # Create rectangle left - normalvector points left
        subdata = val_data[y1:y2, x1-RADIUS:x1]
        subdata[subdata[:,:, 0] <= 0] = np.array([1, 0, 0]) # making sure, that the area around a wall element is marked as surrounding area only once
        subdata[subdata[:,:, 0] != 2] += np.array([0, 0, -1]) # add the normalvector
        val_data[y1:y2, x1-RADIUS:x1] = subdata

        # Create rectangle right - normalvector poibts right
        subdata = val_data[y1:y2, x2:x2+RADIUS]
        subdata[subdata[:,:, 0] <= 0] = np.array([1, 0, 0]) # making sure, that the area around a wall element is marked as surrounding area only once
        subdata[subdata[:,:, 0] != 2] += np.array([0, 0, 1]) # add the normalvector
        val_data[y1:y2, x2:x2+RADIUS] = subdata


# Create template mask for whole circle for wallcorners
circle_wall = np.zeros((2*RADIUS + 1, 2*RADIUS + 1))
Y, X = np.ogrid[:2*RADIUS + 1, :2*RADIUS + 1]
mask_corner = ((X - RADIUS) ** 2) + ((Y - RADIUS) ** 2) <= RADIUS ** 2
circle_wall[mask_corner] = 1 # corner surroundings of walls are also marked as such

# generating a 2D array e.g. (-2, -1, 0, 1, 2) for x and y
idx = np.indices((2*RADIUS + 1, 2*RADIUS + 1))
circle_wall = np.stack((circle_wall, (idx[0] - RADIUS) * circle_wall / RADIUS, (idx[1]- RADIUS) * circle_wall / RADIUS), axis=-1)


for obj in objects:
    if obj[0] == 'w':
        x1, y1, x2, y2 = int(obj[1]), int(obj[2]), int(obj[3]), int(obj[4])
        # Insert circle sector
        # Top left corner
        mask = val_data[y1-RADIUS:y1, x1-RADIUS:x1, 0] <= 0 # mask for all points within the circle sector at the top left corner

        val_data[y1-RADIUS:y1, x1-RADIUS:x1][mask] = circle_wall[:RADIUS, :RADIUS][mask]
        # Top right corner
        mask = val_data[y1-RADIUS:y1, x2:x2+RADIUS, 0] <= 0 # mask for all points within the circle sector at the top right corner
        val_data[y1-RADIUS:y1, x2:x2+RADIUS][mask] = circle_wall[:RADIUS, RADIUS + 1:][mask]
        # Bottom left corner
        mask = val_data[y2:y2+RADIUS, x1-RADIUS:x1, 0] <= 0 # mask for all points within the circle sector at the bottom left corner
        val_data[y2:y2+RADIUS, x1-RADIUS:x1][mask] = circle_wall[RADIUS + 1:, :RADIUS][mask]
        # Bottom right corner
        mask = val_data[y2:y2+RADIUS, x2:x2+RADIUS, 0] <= 0 # mask for all points within the circle sector at the bottom right corner
        val_data[y2:y2+RADIUS, x2:x2+RADIUS][mask] = circle_wall[RADIUS + 1:, RADIUS + 1:][mask]
    

# ====================================================================


window = tk.Tk()
window.title("Game map")
window.geometry("800x480")
# window.attributes('-fullscreen', True)
window.focus_force()

canvas = tk.Canvas(window, width=window.winfo_screenwidth(), height=window.winfo_screenheight(), bg="white")
canvas.pack(fill=tk.BOTH, expand=True)

for obj in objects:
    
    if obj[0] == 'w':
        # Draw a rectangle
        canvas.create_rectangle(int(obj[1]), int(obj[2]), int(obj[3]), int(obj[4]), fill="#471F01", outline="#471F01")
    elif obj[0] == 'h':
        # Draw an oval
        canvas.create_oval(int(obj[1]) - HOLERADIUS, int(obj[2]) - HOLERADIUS, int(obj[1]) + HOLERADIUS, int(obj[2]) + HOLERADIUS, fill="black", outline="lightgray")
    elif obj[0] == 'c':
        # Draw an oval
        canvas.create_oval(int(obj[1]) - HOLERADIUS, int(obj[2]) - HOLERADIUS, int(obj[1]) + HOLERADIUS, int(obj[2]) + HOLERADIUS, fill="green", outline="green")
    # elif obj[0] == 's':
        # Draw an oval
        # canvas.create_oval(int(obj[1]), int(obj[2]), int(obj[3]), int(obj[4]), fill="blue", outline="blue")


pos = start_point.copy()
vel = np.array([0, 0], dtype=float)

ball = canvas.create_oval(int(pos[0]) - RADIUS + 1, int(pos[1]) - RADIUS + 1, int(pos[0]) + RADIUS, int(pos[1]) + RADIUS, fill="blue", outline="blue")

def update_pos():
    window.after(DT, update_pos)
    global pos, vel

    ax, ay = get_acceleration()

    vel[0] += ACC_SCALE * ax * DT / 1000
    vel[1] += ACC_SCALE * ay * DT / 1000
    Dpos = np.array(vel) * DT / 1000
    dist = np.linalg.norm(Dpos)
    steps = int(dist / DP) if dist > DP else 1
    #print(f"Velocity: {vel}\r\b", end="")
    
    dstep = Dpos / steps
    counter = 0
    security = 0

    while counter < steps:
        security += 1
        if security > 1000:
            print("Security limit reached, breaking loop")
            quit()
        temp_pos = pos + dstep
            
        if (val_data[int(temp_pos[1]), int(temp_pos[0]), 0] > 0):
            vec_norm = val_data[int(temp_pos[1]), int(temp_pos[0]), 1:3][::-1]
            pos_dot_product = np.dot(vec_norm, Dpos)
            if (pos_dot_product < 0):
                vec_proj_pos = pos_dot_product / np.dot(vec_norm, vec_norm) * vec_norm
                vec_proj_vel = np.dot(vec_norm, vel) / np.dot(vec_norm, vec_norm) * vec_norm
                Dpos = - 2 * vec_proj_pos + Dpos
                vel = - 2 * vec_proj_vel + vel

                # Dpos *= (steps - counter) / (steps)
                Dpos += vec_proj_pos * DAMPING
                vel += vec_proj_vel * DAMPING
                dist = np.linalg.norm(Dpos)
                steps = int(dist / DP) if dist > DP else 1
                dstep = Dpos / steps
                counter = 0
                continue
                # cancel minimal speed

            else:
                shift = vec_norm / np.linalg.norm(vec_norm) * DP
                pos += shift
                Dpos -= shift
                counter += 1
                continue
        elif (val_data[int(temp_pos[1]), int(temp_pos[0]), 0] == -2):
            pos = start_point.copy() # maybe delay
            vel = np.array([0, 0], dtype=float)
            break
        
        
        pos += dstep
        Dpos -= dstep
            
        counter += 1

    if (val_data[int(pos[1]), int(pos[0]), 0] == -1):
        vec_norm = val_data[int(pos[1]), int(pos[0]), 1:3][::-1]  # Normal vector (y, x) -> (x, y)
        vec_tang = np.array([vec_norm[1], -vec_norm[0]])  # Tangential vector (90 degrees rotation)
        vec_proj_vel_tang = np.dot(vec_tang, vel) / np.dot(vec_tang, vec_tang) * vec_tang
        vel += vec_norm * 10
        vel -= vec_proj_vel_tang * 0.3

        
    # elif (val_data[int(pos[1]), int(pos[0]), 0] == -2):
    #     pos = start_point.copy()
    #     vel = np.array([0, 0], dtype=float)
            

    canvas.coords(ball, int(pos[0]) - RADIUS + 1, int(pos[1]) - RADIUS + 1, int(pos[0]) + RADIUS, int(pos[1]) + RADIUS)

def go_fullscreen():
    window.attributes('-fullscreen', True)


def end_fullscreen(event=None):
    window.attributes('-fullscreen', False)    

window.bind("<Escape>", end_fullscreen)
window.bind("<KeyPress>", on_key_press)
window.bind("<KeyRelease>", on_key_release)

window.after(DT, update_pos)
# window.after(100, go_fullscreen)

window.mainloop()
# plt.imshow(2 - circle_wall[:,:,0], cmap='gray', vmin=-3, vmax=2)
# plt.show()

# # Downsample for clarity (optional, otherwise plot will be very dense)
# plot_data = val_data
# step = 1  # plot every 10th pixel
# Y, X = np.mgrid[0:plot_data.shape[0]:step, 0:plot_data.shape[1]:step]
# U = plot_data[::step, ::step, 2]  # x-component
# V = plot_data[::step, ::step, 1]  # y-component

# plt.figure(figsize=(10, 6))
# plt.quiver(X, Y, U, V, angles='xy', scale_units='xy', scale=1, color='red')
# plt.gca().invert_yaxis()  # To match image coordinates
# plt.title("Gradient Field (from data[:,:,1] and data[:,:,2])")
# plt.axis('equal')
# plt.show()