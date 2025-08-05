# import json
# import tkinter as tk
# import numpy as np

from my_source import *

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
HEIGHT = 480
WIDTH = 800

control_mode, mpl_Debug, checkpoint_names = load_config("config.json")
define_controls(control_mode)

if mpl_Debug:
    import matplotlib.pyplot as plt

objects = []
start_point = np.array([0, 0], dtype=float)
checkpoints = np.array(list(zip(np.empty(4), np.zeros(4, dtype=int), np.empty(4), checkpoint_names.split("\t"))), dtype=object)

hole_radius, start_point, objects = load_map_objects(FILENAME, HOLERADIUS, start_point, objects)

# 2D np array
val_data = generate_val_data(HEIGHT, WIDTH, RADIUS, hole_radius, objects)

window = tk.Tk()
window.title("Game map")
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

close_button = tk.Button(window, text="✕", command=close_app, font=("Arial", 14, "bold"), bg="red", fg="white", bd=0, relief="flat", cursor="hand2")
close_button.place(x=780, y=0, width=20, height=20)  # Top-left corner (adjust x, y for top-right if needed)

reset_button = tk.Button(window, text="\u27F3", command=reset_game, font=("Arial", 14, "bold"), bg="blue", fg="white", bd=0, relief="flat", cursor="hand2")
reset_button.place(x=0, y=0, width=20, height=20)  # Top-left corner (adjust x, y for top-right if needed)

window.bind("<Escape>", end_fullscreen)
if control_mode == "keyboard":
    window.bind("<KeyPress>", on_key_press)
    window.bind("<KeyRelease>", on_key_release)

window.after(DT, update_pos)
if control_mode == "mpu6050":
    window.after(100, go_fullscreen)

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