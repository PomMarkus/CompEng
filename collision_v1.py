import json
import tkinter as tk
import time
import sys
import numpy as np
import re

import paho.mqtt.client as mqtt

with open("config.json") as f:
    config = json.load(f)

control_mode = config.get("control", "keyboard")  # Default to keyboard if not specified
mpl_Debug = config.get("mpl_debug", False)  # Default to False if not specified
checkpoint_names = config.get("checkpoints", "1H\t9O\t7L\t0E")
digit_code = re.sub(r'\D', '', checkpoint_names)  # Extract digits from checkpoint names
parsed_checkpoint_names = checkpoint_names.split("\t")
parsed_checkpoint_names = [parsed_checkpoint_names[2], parsed_checkpoint_names[1], parsed_checkpoint_names[0], parsed_checkpoint_names[3]]  # Reorder checkpoints

if control_mode not in ["keyboard", "mpu6050"]:
    raise ValueError(f"Invalid control mode: {control_mode}. Choose 'keyboard' or 'mpu6050'.")

elif control_mode == "keyboard":
    pressed_keys = set()

    def on_key_press(event):
        pressed_keys.add(event.keysym)

    def on_key_release(event):
        pressed_keys.discard(event.keysym)

    def get_acceleration():
        if "Left" in pressed_keys:
            ax= -5
        elif "Right" in pressed_keys:
            ax= 5
        else:
            ax = 0
            
        if "Down" in pressed_keys:
            ay= 5
        elif "Up" in pressed_keys:
            ay=-5
        else:
            ay = 0
        return ax, ay
    
elif control_mode == "mpu6050":
    from mpu6050 import mpu6050 # type: ignore

    sensor = mpu6050(0x68)

    def get_acceleration():
        current_acc = sensor.get_accel_data()
        return current_acc['y'], - current_acc['x']
        
if mpl_Debug:
    import matplotlib.pyplot as plt	


HEIGHT = 480
WIDTH = 800
DT = 20
DP = 0.1
ACC_SCALE = 100
DAMPING = 0.8
# VELTHRESHOLD = 1



FILENAME = "map_v1.dat"
RADIUS = 10
HOLERADIUS = 12

FANGPIO = 15
VIBROGPIO = 14

BROKER = "tanzwg.jkub.com"
PORT = 1883
TOPIC = "pr_embedded/puzzle_tilt_maze"
USERNAME = "SETTLE DOWN"
PASSWORD = "NEULAND"

def on_connect(client, userdata, flags, rc):
    print(f"Connected to MQTT broker with result code {rc}")
    client.publish(TOPIC+"/general", "Connected")
    client.subscribe(TOPIC+"/general")


def on_message(client, userdata, msg):
    global is_started
    print(f"Received message on topic {msg.topic}: {msg.payload.decode()}")
    if msg.topic == TOPIC + "/general":
        if msg.payload.decode() == "initialize":
            client.publish(TOPIC + "/general", "initialize_ack")

        elif msg.payload.decode() == "start" and not is_started:
            is_started = True
            client.publish(TOPIC + "/general", "start_ack")
            start_game()

client = mqtt.Client()

client.username_pw_set(USERNAME, PASSWORD)

client.on_connect = on_connect
client.on_message = on_message


print("trying to connect to MQTT broker...")
client.connect(BROKER, PORT, 30)



is_paused = False
is_finished = False
is_started = False
code_overlay_flag = False
fell_into_holes = 0
hole_cool_down = 0
vibrate_cool_down = 0

client.loop_start()


if sys.platform == "linux":
    import RPi.GPIO as GPIO
    GPIO.setwarnings(False)
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(VIBROGPIO, GPIO.OUT)

    def high(pin):
        GPIO.output(pin,GPIO.HIGH)
    def low(pin):
        GPIO.output(pin,GPIO.LOW)


# =============== Import and process objects from file ===============

objects = []
start_point = np.array([0, 0], dtype=float)

checkpoints = np.array(list(zip(np.empty(4), np.zeros(4, dtype=int), np.empty(4), parsed_checkpoint_names)), dtype=object)



with open(FILENAME, "r") as f:
    for line in f:
        line = line.strip()
        if not line or line.startswith("#"):
            pass
        elif line.startswith("s"):
            start_point = np.array([int(line.split("\t")[1]), int(line.split("\t")[2])], dtype=float)
        elif line.startswith("r"):
            HOLERADIUS = int(line.split("\t")[1])
        else: 
            objects.append(line.split("\t"))

start_point_default = start_point.copy()


# 2D np array
val_data = np.zeros((HEIGHT, WIDTH, 4))

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
circle_hole[:, :, 1:3] /= norm_squared

inner_mask = ((X - HOLERADIUS + 0.5) ** 2) + ((Y - HOLERADIUS + 0.5) ** 2) <= (HOLERADIUS - RADIUS) ** 2
circle_hole[inner_mask, 0] = -2

checkpoint_counter = 0

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
        subdata[mask_hole, :3] = circle_hole[mask_hole, :3]
        val_data[y - HOLERADIUS:y + HOLERADIUS, x - HOLERADIUS:x + HOLERADIUS] = subdata # fill the hole with the circle_hole template
    
    # Checkpoint pxl to -3
    elif obj[0] == 'c':
        x, y = int(obj[1]), int(obj[2])
        # Create a mask for defining the pxl inside the circle
        Y, X = np.ogrid[:HEIGHT, :WIDTH]
        mask_checkpoint = (((X - x) ** 2)) + ((Y - y) ** 2)<= HOLERADIUS ** 2
        # Set the pixels inside the checkoint to -3
        val_data[mask_checkpoint, 0] = -3
        val_data[mask_checkpoint, 3] = checkpoint_counter
        checkpoint_counter += 1

        

# Fill area of shifted rectangles with 1 and add the normalvector for the rebouncing calculation
for obj in objects:

    if obj[0] == 'w':
        x1, y1, x2, y2 = int(obj[1]), int(obj[2]), int(obj[3]), int(obj[4])
        Y, X = np.ogrid[:HEIGHT, :WIDTH]

        # Create rectangle above - normalvector points upwards
        subdata = val_data[y1-RADIUS:y1, x1:x2]
        subdata[subdata[:,:, 0] <= 0, 0] = 1 # making sure, that the area around a wall element is marked as surrounding area only once
        subdata[subdata[:,:, 0] != 2] += np.array([0, -1, 0, 0]) # add the normalvector
        val_data[y1-RADIUS:y1, x1:x2] = subdata

        # Create rectangle below - normalvector points downwards
        subdata = val_data[y2:y2+RADIUS, x1:x2]
        subdata[subdata[:,:, 0] <= 0, 0] = 1 # making sure, that the area around a wall element is marked as surrounding area only once
        subdata[subdata[:,:, 0] != 2] += np.array([0, 1, 0, 0]) # add the normalvector
        val_data[y2:y2+RADIUS, x1:x2] = subdata

        # Create rectangle left - normalvector points left
        subdata = val_data[y1:y2, x1-RADIUS:x1]
        subdata[subdata[:,:, 0] <= 0, 0] = 1 # making sure, that the area around a wall element is marked as surrounding area only once
        subdata[subdata[:,:, 0] != 2] += np.array([0, 0, -1, 0]) # add the normalvector
        val_data[y1:y2, x1-RADIUS:x1] = subdata

        # Create rectangle right - normalvector poibts right
        subdata = val_data[y1:y2, x2:x2+RADIUS]
        subdata[subdata[:,:, 0] <= 0, 0] = 1 # making sure, that the area around a wall element is marked as surrounding area only once
        subdata[subdata[:,:, 0] != 2] += np.array([0, 0, 1, 0]) # add the normalvector
        val_data[y1:y2, x2:x2+RADIUS] = subdata


# Create template mask for whole circle for wallcorners
circle_wall = np.zeros((2*RADIUS + 1, 2*RADIUS + 1))
Y, X = np.ogrid[:2*RADIUS + 1, :2*RADIUS + 1]
mask_corner = ((X - RADIUS) ** 2) + ((Y - RADIUS) ** 2) <= RADIUS ** 2
circle_wall[mask_corner] = 1 # corner surroundings of walls are also marked as such

# generating a 2D array e.g. (-2, -1, 0, 1, 2) for x and y
idx = np.indices((2*RADIUS + 1, 2*RADIUS + 1))
circle_wall = np.stack((circle_wall, (idx[0] - RADIUS) * circle_wall / RADIUS, (idx[1]- RADIUS) * circle_wall / RADIUS, np.zeros((2*RADIUS + 1, 2*RADIUS + 1))), axis=-1)


for obj in objects:
    if obj[0] == 'w':
        x1, y1, x2, y2 = int(obj[1]), int(obj[2]), int(obj[3]), int(obj[4])
        # Insert circle sector
        # Top left corner
        mask = np.isin(val_data[y1-RADIUS:y1, x1-RADIUS:x1, 0], [0, -1]) # mask for all points within the circle sector at the top left corner

        val_data[y1-RADIUS:y1, x1-RADIUS:x1][mask] = circle_wall[:RADIUS, :RADIUS][mask]
        # Top right corner
        mask = np.isin(val_data[y1-RADIUS:y1, x2:x2+RADIUS, 0], [0, -1]) # mask for all points within the circle sector at the top right corner
        val_data[y1-RADIUS:y1, x2:x2+RADIUS][mask] = circle_wall[:RADIUS, RADIUS + 1:][mask]
        # Bottom left corner
        mask = np.isin(val_data[y2:y2+RADIUS, x1-RADIUS:x1, 0], [0, -1]) # mask for all points within the circle sector at the bottom left corner
        val_data[y2:y2+RADIUS, x1-RADIUS:x1][mask] = circle_wall[RADIUS + 1:, :RADIUS][mask]
        # Bottom right corner
        mask = np.isin(val_data[y2:y2+RADIUS, x2:x2+RADIUS, 0], [0, -1]) # mask for all points within the circle sector at the bottom right corner
        val_data[y2:y2+RADIUS, x2:x2+RADIUS][mask] = circle_wall[RADIUS + 1:, RADIUS + 1:][mask]
    

# ====================================================================

window = tk.Tk()
window.title("Tilt Maze")
window.geometry("800x480")
if control_mode == "mpu6050":
    window.attributes('-fullscreen', True)
window.focus_force()

checkpoint_counter = 0
canvas = tk.Canvas(window, width=window.winfo_screenwidth(), height=window.winfo_screenheight(), bg="white")
canvas.pack(fill=tk.BOTH, expand=True)
canvas.config(cursor="none")

for obj in objects:
    
    if obj[0] == 'w':
        # Draw a rectangle
        canvas.create_rectangle(int(obj[1]), int(obj[2]), int(obj[3]), int(obj[4]), fill="#471F01", outline="#471F01")
    elif obj[0] == 'h':
        # Draw an oval
        canvas.create_oval(int(obj[1]) - HOLERADIUS, int(obj[2]) - HOLERADIUS, int(obj[1]) + HOLERADIUS, int(obj[2]) + HOLERADIUS, fill="black", outline="lightgray")
    elif obj[0] == 'c':
        # Draw an oval
        checkpoints[checkpoint_counter][0] = canvas.create_oval(int(obj[1]) - HOLERADIUS, int(obj[2]) - HOLERADIUS, int(obj[1]) + HOLERADIUS, int(obj[2]) + HOLERADIUS, fill="orange", outline="lightgray")
        checkpoints[checkpoint_counter][2] = canvas.create_text(int(obj[1]), int(obj[2]), text="", fill="black", font=("Arial", 10, "bold"))
        checkpoint_counter += 1


pos = start_point.copy()

vel = np.array([0, 0], dtype=float)

ball = canvas.create_oval(int(pos[0]) - RADIUS + 1, int(pos[1]) - RADIUS + 1, int(pos[0]) + RADIUS, int(pos[1]) + RADIUS, fill="blue", outline="blue")
checkpoint_counter = 0

def update_pos():
    global pos, vel, start_point, hole_cool_down, vibrate_cool_down, fell_into_holes, checkpoint_counter, checkpoints, ball, val_data, is_paused, is_finished, hole_status_text, client, vibro_ind

    if is_paused:
        return
    else:
        window.after(DT, update_pos)
    
    if vibrate_cool_down > 0:
        vibrate_cool_down -= DT
        if vibrate_cool_down <= 0:
            vibrate_cool_down = 0
            if sys.platform == "linux":
                low(VIBROGPIO)
            else:
                canvas.itemconfig(vibro_ind, fill="gray")
                  

    if hole_cool_down > 0:
        hole_cool_down -= DT
        if hole_cool_down <= 0:
            hole_cool_down = 0
            pos = start_point.copy()
            vel = np.array([0, 0], dtype=float)
        return
    
    last_pos = pos.copy()
    normal_vectors = set()

    if checkpoint_counter >= len(checkpoints) and not is_finished:
    # if not is_finished:
        is_finished = True
        canvas.itemconfig(ball, fill="gold", outline="gold")
        start_point = start_point_default.copy()
        pos = start_point.copy()
        vel = np.array([0, 0], dtype=float)
        pause_game()
        pause_button.config(state="disabled")
        code_button.config(state="disabled")
        show_code_overlay()
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
            normal_vectors.add((val_data[int(temp_pos[1]), int(temp_pos[0]), 2], val_data[int(temp_pos[1]), int(temp_pos[0]), 1]))
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


            else:
                shift = vec_norm / np.linalg.norm(vec_norm) * DP
                pos += shift
                Dpos -= shift
                counter += 1
                continue
        elif (val_data[int(temp_pos[1]), int(temp_pos[0]), 0] == -2):
            vel = np.array([0, 0], dtype=float)
            fell_into_holes += 1
            canvas.itemconfig(hole_status_text, text=f"Fell into holes: {fell_into_holes}")
            hole_cool_down = 500  # Cooldown for falling into a hole
            break
        elif (val_data[int(temp_pos[1]), int(temp_pos[0]), 0] == -3):
            c_number = int(val_data[int(temp_pos[1]), int(temp_pos[0]), 3])
            if checkpoints[c_number][1] == 0:  # If checkpoint is not yet reached
                checkpoints[c_number][1] = 1  # Mark checkpoint as reached
                canvas.itemconfig(checkpoints[c_number][0], fill="#0CFF0B")  # Change color
                canvas.itemconfig(checkpoints[c_number][2], text=checkpoints[c_number][3])  # Update text
                checkpoint_counter += 1
                coords = canvas.coords(checkpoints[c_number][0])
                start_point = np.array([(coords[0] + coords[2]) / 2, (coords[1] + coords[3]) / 2], dtype=float)  # Update start point to the center of the checkpoint
        
        pos += dstep
        Dpos -= dstep
            
        counter += 1

    if (val_data[int(pos[1]), int(pos[0]), 0] == -1): #Hole mechanism
        vec_norm = val_data[int(pos[1]), int(pos[0]), 1:3][::-1]  # Normal vector (y, x) -> (x, y)
        vec_tang = np.array([vec_norm[1], -vec_norm[0]])  # Tangential vector (90 degrees rotation)
        vec_proj_vel_tang = np.dot(vec_tang, vel) / np.dot(vec_tang, vec_tang) * vec_tang
        vel += vec_norm * 10
        vel -= vec_proj_vel_tang * 0.3

    pos_difference = pos - last_pos
    for vector in normal_vectors:
        vec_norm = np.array(vector, dtype=float)
        vec_proj_difference = np.dot(vec_norm, pos_difference) / np.dot(vec_norm, vec_norm) * vec_norm
        if np.linalg.norm(vec_proj_difference) > 0.1:
            vibrate_cool_down = 100
            if sys.platform == "linux":
                high(VIBROGPIO)
            else:
                canvas.itemconfig(vibro_ind, fill="red")

        # print(pos_difference, vec_norm, np.linalg.norm(vec_proj_difference))

        
    # elif (val_data[int(pos[1]), int(pos[0]), 0] == -2):
    #     pos = start_point.copy()
    #     vel = np.array([0, 0], dtype=float)
            

    canvas.coords(ball, int(pos[0]) - RADIUS + 1, int(pos[1]) - RADIUS + 1, int(pos[0]) + RADIUS, int(pos[1]) + RADIUS)

def go_fullscreen():
    window.attributes('-fullscreen', True)


def end_fullscreen(event=None):
    window.attributes('-fullscreen', False)    

def close_app():
    if sys.platform == "linux":
        low(VIBROGPIO)
        GPIO.cleanup()
    window.destroy()


def reset_game():
    global pos, vel, checkpoint_counter, checkpoints, ball
    pos = start_point.copy()
    vel = np.array([0, 0], dtype=float)
    checkpoint_counter = 0
    for i in range(len(checkpoints)):
        checkpoints[i][1] = 0  # Reset checkpoint status
        canvas.itemconfig(checkpoints[i][0], fill="orange")  # Reset color
        canvas.itemconfig(checkpoints[i][2], text="")  # Reset text
    canvas.coords(ball, int(pos[0]) - RADIUS + 1, int(pos[1]) - RADIUS + 1, int(pos[0]) + RADIUS, int(pos[1]) + RADIUS)
    canvas.itemconfig(ball, fill="blue", outline="blue")  # Reset ball color

def pause_game():
    global is_paused
    if is_paused:
        is_paused = False
        window.after(DT, update_pos)
        pause_button.config(bg="green", text="\u23F8")  # Change button to pause icon
    else:
        is_paused = True
        if sys.platform == "linux":
            low(VIBROGPIO)
        window.after_cancel(update_pos)  # Stop the update loop
        pause_button.config(bg="orange", text="\u25B6")  # Change button to play icon
        

def start_game():
    global overlay_button, overlay_label
    reset_game()  # Reset game state
    overlay_label.config(text="Game is unlocked! Press Start to begin.")
    overlay_button.config(state="normal", bg="green")
    pause_button.config(state="normal")  # Enable pause button


def show_code_overlay():
    global client, code_overlay_flag, digit_code, code_overlay
    if code_overlay_flag:
        code_overlay_flag = False
        code_overlay.destroy()
        return
    code_overlay_flag = True

    # Centered overlay frame
    overlay_w, overlay_h = 300, 323
    overlay_x = (WIDTH - overlay_w) // 2
    overlay_y = (HEIGHT - overlay_h) // 2
    code_overlay = tk.Frame(window, bg="#eeeeee", bd=3, relief="ridge")
    code_overlay.place(x=overlay_x, y=overlay_y, width=overlay_w, height=overlay_h)

    # Display area for entered code
    code_var = tk.StringVar()
    code_entry = tk.Entry(code_overlay, textvariable=code_var, font=("Arial", 24), justify="center", state="readonly", readonlybackground="#ffffff")
    code_entry.pack(pady=(20, 10), padx=20, fill="x")

    # Numpad button handler
    def numpad_press(val):
        global client, digit_code
        if val == "Del":
            code_var.set(code_var.get()[:-1])
        elif val == "OK":
            # Handle code confirmation here
            if code_var.get() == digit_code:
                code_overlay.destroy()
                client.publish(TOPIC + "/general", "finished")
                client.publish(TOPIC + "/points", (5 * fell_into_holes) if fell_into_holes < 10 else 45)
                show_finished_overlay()
            else:
                code_var.set("")
        else:
            if len(code_var.get()) < 4:
                code_var.set(code_var.get() + val)

    # Numpad layout
    btns = [
        ["1", "2", "3"],
        ["4", "5", "6"],
        ["7", "8", "9"],
        ["Del", "0", "OK"]
    ]
    btn_frame = tk.Frame(code_overlay, bg="#eeeeee")
    btn_frame.pack(pady=10)
    for r, row in enumerate(btns):
        for c, val in enumerate(row):
            b = tk.Button(
                btn_frame, text=val, width=5, height=1, font=("Arial", 16),
                command=lambda v=val: numpad_press(v)
            )
            b.grid(row=r, column=c, padx=5, pady=5)


def show_finished_overlay():
    semi_transparent_finished_overlay = canvas.create_rectangle(0, 0, WIDTH, HEIGHT, fill="#424242", outline="", stipple="gray50")
    finished_overlay = canvas.create_rectangle(200, 140, 600, 340, fill="#eeeeee", outline="black")
    finished_overlay_frame = tk.Frame(window, bg="#eeeeee", bd=3, relief="ridge")
    finished_overlay_frame.place(x=200, y=140, width=400, height=200)
    finished_overlay_label = tk.Label(finished_overlay_frame, text="Finished!", bg="#eeeeee", fg="green", font=("Arial", 18, "bold"))
    finished_overlay_label.pack(pady=75)


close_button = tk.Button(window, text="✕", command=close_app, font=("Arial", 14, "bold"), bg="red", fg="white", bd=0, relief="flat", cursor="hand2")
close_button.place(x=780, y=0, width=20, height=20)  # Top-left corner (adjust x, y for top-right if needed)

pause_button = tk.Button(window, text="\u23F8", command=pause_game, font=("Symbola"
"", 12), bg="green", fg="white", bd=0, relief="flat", cursor="hand2")
pause_button.place(x=0, y=0, width=20, height=20)  # Top-left corner (adjust x, y for top-right if needed)
pause_button.config(state="disabled")  # Initially disabled until game starts

# reset_button = tk.Button(window, text="\u27F3", command=reset_game, font=("Arial", 14, "bold"), bg="blue", fg="white", bd=0, relief="flat", cursor="hand2")
# reset_button.place(x=0, y=0, width=20, height=20)  # Top-left corner (adjust x, y for top-right if needed)

code_button = tk.Button(window, text="\U0001F511", command=lambda: [pause_game(), show_code_overlay()], font=("Arial", 12, "bold"), bg="#471F01", fg="white", bd=0, relief="flat", cursor="hand2", highlightthickness=0, highlightbackground="#471F01")


hole_status_text = canvas.create_text(400, 3, text=f"Fell into holes: {fell_into_holes}", font=("Arial", 10, "bold"), fill="white", anchor="n")

vibro_ind = None
if sys.platform != "linux":
    vibro_ind = canvas.create_oval(30, 5, 40, 15, fill="gray", outline="black")


semi_transparent_overlay = canvas.create_rectangle(0, 0, WIDTH, HEIGHT, fill="#424242", outline="", stipple="gray50")

overlay = canvas.create_rectangle(200, 140, 600, 340, fill="#eeeeee", outline="black")
overlay_frame = tk.Frame(window, bg="#eeeeee", bd=3, relief="ridge")
overlay_frame.place(x=200, y=140, width=400, height=200)
overlay_label = tk.Label(overlay_frame, text="Tilt Maze is not yet unlocked!", bg="#eeeeee", font=("Arial", 14, "bold"))
overlay_label.pack(pady=30)
overlay_button = tk.Button(overlay_frame, 
                           text="  Start  ", 
                           state="disabled", 
                           command=lambda: [overlay_frame.destroy(), canvas.delete(overlay), canvas.delete(semi_transparent_overlay), window.after(DT, update_pos), code_button.place(x=0, y=460, width=20, height=20) ],
                           bg="#eeeeee", font=("Arial", 16, "bold"))
overlay_button.pack(pady=15)

code_overlay = None

# popup = tk.Toplevel(window)
# popup.title("Tilt Maze")
# popup.geometry("300x200+250+140")
# popup.transient(window)  # Make popup a child of the main window
# popup.grab_set()  # Make popup modal

# popup_label = tk.Label(popup, text="Tilt Maze not yet unlocked!")
# popup_label.pack(pady=10)

# popup_button = tk.Button(popup, text="Start Game", state="disabled", command=lambda: [popup.destroy(), window.after(DT, update_pos)])
# popup_button.pack(pady=10)

window.bind("<Escape>", end_fullscreen)
if control_mode == "keyboard":
    window.bind("<KeyPress>", on_key_press)
    window.bind("<KeyRelease>", on_key_release)


if control_mode == "mpu6050":
    window.after(100, go_fullscreen)
# =====================
if False:
    start_game()
# =====================
window.mainloop()

if mpl_Debug:
    plt.imshow(-val_data[:,:,0], cmap='gray', vmin=-3, vmax=4)
    plt.show()

    # Downsample for clarity (optional, otherwise plot will be very dense)
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