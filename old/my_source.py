import json
import tkinter as tk
import numpy as np


def load_config(file_path):
    with open(file_path) as f:
        config = json.load(f)
    control_mode = config.get("control", "keyboard")  # Default to keyboard if not specified
    mpl_Debug = config.get("mpl_debug", False)  # Default to False if not specified
    checkpoint_names = config.get("checkpoints", "1G\t9A\t7M\t0E")

    return control_mode, mpl_Debug, checkpoint_names


def define_controls(control_mode):
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
        
def load_map_objects(filename, hole_radius, start_point, objects):
    with open(filename, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                pass
            elif line.startswith("s"):
                start_point = np.array([int(line.split("\t")[1]), int(line.split("\t")[2])], dtype=float)
            elif line.startswith("r"):
                hole_radius = int(line.split("\t")[1])
            else: 
                objects.append(line.split("\t"))
    return hole_radius, start_point, objects

def generate_val_data(height, width, radius, hole_radius, objects):
    # 2D np array
    val_data = np.zeros((height, width, 4))

    # Create template mask for circle for the holes
    circle_hole = np.zeros((hole_radius * 2, hole_radius * 2))
    Y, X = np.ogrid[:hole_radius * 2, :hole_radius * 2]
    mask_hole = ((X - hole_radius + 0.5) ** 2) + ((Y - hole_radius + 0.5) ** 2) <= hole_radius ** 2
    circle_hole[mask_hole] = -1 # corner surroundings of walls are also marked as such

    # generating a 2D array e.g. (-2, -1, 0, 1, 2) for x and y
    idx = np.indices((hole_radius * 2, hole_radius * 2)) - hole_radius
    idx[idx >=0] += 1
    circle_hole = np.stack((circle_hole, (idx[0]) * circle_hole / radius, (idx[1]) * circle_hole / radius), axis=-1)

    norm_squared = np.sum(circle_hole[:, :, 1:3] ** 2, axis=-1, keepdims=True)
    norm_squared[norm_squared == 0] = 1e-1
    circle_hole[:, :, 1:3] /= norm_squared

    inner_mask = ((X - hole_radius + 0.5) ** 2) + ((Y - hole_radius + 0.5) ** 2) <= (hole_radius - radius) ** 2
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
            
            subdata = val_data[y - hole_radius:y + hole_radius, x - hole_radius:x + hole_radius]
            subdata[mask_hole, :3] = circle_hole[mask_hole, :3]
            val_data[y - hole_radius:y + hole_radius, x - hole_radius:x + hole_radius] = subdata # fill the hole with the circle_hole template
        
        # Checkpoint pxl to -3
        elif obj[0] == 'c':
            x, y = int(obj[1]), int(obj[2])
            # Create a mask for defining the pxl inside the circle
            Y, X = np.ogrid[:height, :width]
            mask_checkpoint = (((X - x) ** 2)) + ((Y - y) ** 2)<= hole_radius ** 2
            # Set the pixels inside the checkoint to -3
            val_data[mask_checkpoint, 0] = -3
            val_data[mask_checkpoint, 3] = checkpoint_counter
            checkpoint_counter += 1

            

    # Fill area of shifted rectangles with 1 and add the normalvector for the rebouncing calculation
    for obj in objects:

        if obj[0] == 'w':
            x1, y1, x2, y2 = int(obj[1]), int(obj[2]), int(obj[3]), int(obj[4])
            Y, X = np.ogrid[:height, :width]

            # Create rectangle above - normalvector points upwards
            subdata = val_data[y1-radius:y1, x1:x2]
            subdata[subdata[:,:, 0] <= 0, 0] = 1 # making sure, that the area around a wall element is marked as surrounding area only once
            subdata[subdata[:,:, 0] != 2] += np.array([0, -1, 0, 0]) # add the normalvector
            val_data[y1-radius:y1, x1:x2] = subdata

            # Create rectangle below - normalvector points downwards
            subdata = val_data[y2:y2+radius, x1:x2]
            subdata[subdata[:,:, 0] <= 0, 0] = 1 # making sure, that the area around a wall element is marked as surrounding area only once
            subdata[subdata[:,:, 0] != 2] += np.array([0, 1, 0, 0]) # add the normalvector
            val_data[y2:y2+radius, x1:x2] = subdata

            # Create rectangle left - normalvector points left
            subdata = val_data[y1:y2, x1-radius:x1]
            subdata[subdata[:,:, 0] <= 0, 0] = 1 # making sure, that the area around a wall element is marked as surrounding area only once
            subdata[subdata[:,:, 0] != 2] += np.array([0, 0, -1, 0]) # add the normalvector
            val_data[y1:y2, x1-radius:x1] = subdata

            # Create rectangle right - normalvector poibts right
            subdata = val_data[y1:y2, x2:x2+radius]
            subdata[subdata[:,:, 0] <= 0, 0] = 1 # making sure, that the area around a wall element is marked as surrounding area only once
            subdata[subdata[:,:, 0] != 2] += np.array([0, 0, 1, 0]) # add the normalvector
            val_data[y1:y2, x2:x2+radius] = subdata


    # Create template mask for whole circle for wallcorners
    circle_wall = np.zeros((2*radius + 1, 2*radius + 1))
    Y, X = np.ogrid[:2*radius + 1, :2*radius + 1]
    mask_corner = ((X - radius) ** 2) + ((Y - radius) ** 2) <= radius ** 2
    circle_wall[mask_corner] = 1 # corner surroundings of walls are also marked as such

    # generating a 2D array e.g. (-2, -1, 0, 1, 2) for x and y
    idx = np.indices((2*radius + 1, 2*radius + 1))
    circle_wall = np.stack((circle_wall, (idx[0] - radius) * circle_wall / radius, (idx[1]- radius) * circle_wall / radius, np.zeros((2*radius + 1, 2*radius + 1))), axis=-1)


    for obj in objects:
        if obj[0] == 'w':
            x1, y1, x2, y2 = int(obj[1]), int(obj[2]), int(obj[3]), int(obj[4])
            # Insert circle sector
            # Top left corner
            mask = np.isin(val_data[y1-radius:y1, x1-radius:x1, 0], [0, -1]) # mask for all points within the circle sector at the top left corner

            val_data[y1-radius:y1, x1-radius:x1][mask] = circle_wall[:radius, :radius][mask]
            # Top right corner
            mask = np.isin(val_data[y1-radius:y1, x2:x2+radius, 0], [0, -1]) # mask for all points within the circle sector at the top right corner
            val_data[y1-radius:y1, x2:x2+radius][mask] = circle_wall[:radius, radius + 1:][mask]
            # Bottom left corner
            mask = np.isin(val_data[y2:y2+radius, x1-radius:x1, 0], [0, -1]) # mask for all points within the circle sector at the bottom left corner
            val_data[y2:y2+radius, x1-radius:x1][mask] = circle_wall[radius + 1:, :radius][mask]
            # Bottom right corner
            mask = np.isin(val_data[y2:y2+radius, x2:x2+radius, 0], [0, -1]) # mask for all points within the circle sector at the bottom right corner
            val_data[y2:y2+radius, x2:x2+radius][mask] = circle_wall[radius + 1:, radius + 1:][mask]
        
    return val_data


def update_pos():
    window.after(DT, update_pos)
    global pos, vel, checkpoint_counter, checkpoints, ball, val_data

    if checkpoint_counter >= len(checkpoints):
        canvas.itemconfig(ball, fill="gold", outline="gold")
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
        elif (val_data[int(temp_pos[1]), int(temp_pos[0]), 0] == -3):
            c_number = int(val_data[int(temp_pos[1]), int(temp_pos[0]), 3])
            if checkpoints[c_number][1] == 0:  # If checkpoint is not yet reached
                checkpoints[c_number][1] = 1  # Mark checkpoint as reached
                canvas.itemconfig(checkpoints[c_number][0], fill="#0CFF0B")  # Change color
                canvas.itemconfig(checkpoints[c_number][2], text=checkpoints[c_number][3])  # Update text
                checkpoint_counter += 1
        
        
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

def close_app():
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
