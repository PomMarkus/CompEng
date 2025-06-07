import tkinter as tk
import numpy as np
# import matplotlib.pyplot as plt	
# from mpu6050 import mpu6050
# import time

RADIUS = 10
HEIGHT = 480
WIDTH = 800
DT = 20
DP = 0.1
ACC_SCALE = 30
DAMPING = 0.8
# VELTHRESHOLD = 1

# =============== Import and process objects from file ===============

FILENAME = "map_v1.dat"
RADIUS = 10
HEIGHT = 480
WIDTH = 800

objects = []
pressed_keys = set()

with open(FILENAME, "r") as f:
    for line in f:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        objects.append(line.split("\t"))


# 2D np array
val_data = np.zeros((HEIGHT, WIDTH, 3))

# Fill area of objects
for obj in objects:
    x1, y1, x2, y2 = int(obj[1]), int(obj[2]), int(obj[3]), int(obj[4])

    # wall pxl to 2
    if obj[0] == 'w':
        val_data[y1:y2, x1:x2, 0] = 2 # pixle occupied

    # hole pxl to -1 
    elif obj[0] == 'h':
        # Create a mask for defining the pxl inside the circle
        Y, X = np.ogrid[:HEIGHT, :WIDTH]
        mask = ((X - (x1 + x2) / 2) ** 2) / ((x2 - x1) / 2) ** 2 + ((Y - (y1 + y2) / 2) ** 2) / ((y2 - y1) / 2) ** 2 <= 1
        # Set the pixels inside the circle to -1
        val_data[mask, 0] = -1
    
    # Checkpoint pxl to -2 
    elif obj[0] == 'c':
        # Create a mask for defining the pxl inside the circle
        Y, X = np.ogrid[:HEIGHT, :WIDTH]
        mask = ((X - (x1 + x2) / 2) ** 2) / ((x2 - x1) / 2) ** 2 + ((Y - (y1 + y2) / 2) ** 2) / ((y2 - y1) / 2) ** 2 <= 1
        # Set the pixels inside the checkoint to -2
        val_data[mask, 0] = -2
    
    # save the startpoint
    elif obj[0] == 's':
        pos = np.array([x1 + 10, y1 + 10], dtype=float)

# Create template mask for whole circle for wallcorners
circle = np.zeros((2*RADIUS + 1, 2*RADIUS + 1))
Y, X = np.ogrid[:2*RADIUS + 1, :2*RADIUS + 1]
mask = ((X - RADIUS) ** 2) + ((Y - RADIUS) ** 2) <= RADIUS ** 2
circle[mask] = 1 # corner surroundings of walls are also marked as such

# generating a 2D array e.g. (-2, -1, 0, 1, 2) for x and y
idx = np.indices((2*RADIUS + 1, 2*RADIUS + 1))
circle = np.stack((circle, (idx[0] - RADIUS) * circle / RADIUS, (idx[1]- RADIUS) * circle / RADIUS), axis=-1)

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


        # Insert circle sector
        # Top left corner
        mask = val_data[y1-RADIUS:y1, x1-RADIUS:x1, 0] == 0 # mask for all points within the circle sector at the top left corner
        val_data[y1-RADIUS:y1, x1-RADIUS:x1][mask] = circle[:RADIUS, :RADIUS][mask]
        # Top right corner
        mask = val_data[y1-RADIUS:y1, x2:x2+RADIUS, 0] == 0 # mask for all points within the circle sector at the top right corner
        val_data[y1-RADIUS:y1, x2:x2+RADIUS][mask] = circle[:RADIUS, RADIUS + 1:][mask]
        # Bottom left corner
        mask = val_data[y2:y2+RADIUS, x1-RADIUS:x1, 0] == 0 # mask for all points within the circle sector at the bottom left corner
        val_data[y2:y2+RADIUS, x1-RADIUS:x1][mask] = circle[RADIUS + 1:, :RADIUS][mask]
        # Bottom right corner
        mask = val_data[y2:y2+RADIUS, x2:x2+RADIUS, 0] == 0 # mask for all points within the circle sector at the bottom right corner
        val_data[y2:y2+RADIUS, x2:x2+RADIUS][mask] = circle[RADIUS + 1:, RADIUS + 1:][mask]

# ====================================================================

# sensor = mpu6050(0x68)

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
        canvas.create_rectangle(int(obj[1]), int(obj[2]), int(obj[3]), int(obj[4]), fill="black", outline="black")
    elif obj[0] == 'h':
        # Draw an oval
        canvas.create_oval(int(obj[1]), int(obj[2]), int(obj[3]), int(obj[4]), fill="red", outline="red")
    elif obj[0] == 'c':
        # Draw an oval
        canvas.create_oval(int(obj[1]), int(obj[2]), int(obj[3]), int(obj[4]), fill="green", outline="green")
    # elif obj[0] == 's':
        # Draw an oval
        # canvas.create_oval(int(obj[1]), int(obj[2]), int(obj[3]), int(obj[4]), fill="blue", outline="blue")



vel = np.array([0, 0], dtype=float)

ball = canvas.create_oval(int(pos[0]) - RADIUS, int(pos[1]) - RADIUS, int(pos[0]) + RADIUS, int(pos[1]) + RADIUS, fill="blue", outline="blue")

def on_key_press(event):
    pressed_keys.add(event.keysym)

def on_key_release(event):
    pressed_keys.discard(event.keysym)

def update_pos():
    window.after(DT, update_pos)
    global pos, vel

    # current_acc = sensor.get_accel_data()
    # ax, ay = - current_acc['x'], current_acc['y']
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

    vel[0] += ACC_SCALE * ax * DT / 1000
    vel[1] += ACC_SCALE * ay * DT / 1000
    Dpos = np.array(vel) * DT / 1000
    dist = np.linalg.norm(Dpos)
    steps = int(dist / DP) if dist > DP else 1
    
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
        
        elif (val_data[int(temp_pos[1]), int(temp_pos[0]), 0] == -1):
            #hole - pull ball to center - vector to center in val_data
            pass
        
        pos += dstep
        Dpos -= dstep
        counter += 1
            

    canvas.coords(ball, int(pos[0]) - RADIUS, int(pos[1]) - RADIUS, int(pos[0]) + RADIUS, int(pos[1]) + RADIUS)

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
# plt.imshow(2 - val_data[:,:,0], cmap='gray', vmin=0, vmax=2)
# plt.show()