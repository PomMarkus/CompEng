import json
import tkinter as tk
import threading
import time
import sys
import numpy as np
from abc import ABC, abstractmethod

# Placeholder for actual RPi.GPIO or mpu6050 imports
if sys.platform == "linux":
    try:
        import RPi.GPIO as GPIO
        GPIO.setmode(GPIO.BCM)
    except ImportError:
        print("RPi.GPIO not found. Running in simulation mode for fan.")
        GPIO = None
    try:
        from mpu6050 import mpu6050 # type: ignore
    except ImportError:
        print("mpu6050 library not found. Running in keyboard control mode only.")
        mpu6050 = None
else: # Not Linux
    GPIO = None
    mpu6050 = None


# --- 1. GameConfig Class ---
class GameConfig:
    def __init__(self, config_file_path="config.json"):
        with open(config_file_path) as f:
            config_data = json.load(f)

        # Game Constants
        self.HEIGHT = 480
        self.WIDTH = 800
        self.DT = 20 # Milliseconds per update
        self.DP = 0.1 # Distance step for collision checks
        self.ACC_SCALE = 100
        self.DAMPING = 0.8
        self.RADIUS = 10 # Ball radius
        self.HOLERADIUS = 12 # Hole and Checkpoint radius
        self.FILENAME = "map_v1.dat" # Map file

        # Config.json specific
        self.control_mode = config_data.get("control", "keyboard")
        self.mpl_Debug = config_data.get("mpl_debug", False)
        # Note: checkpoint_names will be parsed by GameMap, not just stored as string here
        self.checkpoint_names_str = config_data.get("checkpoints", "1G\t9A\t7M\t0E")

        # Fan Control Constants (can be extended from config.json if desired)
        self.FANGPIO = 15
        self.cooling_hyst_time = 10
        self.cooling_average_time = 5
        self.temp_threshold = [0,58,61,64,67,70,73,76,78,100]
        self.fan_speed_levels = [0,30,40,50,60,70,80,90,100]

        # Basic validation
        if self.control_mode not in ["keyboard", "mpu6050"]:
            raise ValueError(f"Invalid control mode: {self.control_mode}. Choose 'keyboard' or 'mpu6050'.")
        if self.control_mode == "mpu6050" and mpu6050 is None:
            print("Warning: MPU6050 selected but library not found. Falling back to keyboard control.")
            self.control_mode = "keyboard" # Fallback if library missing


# --- 2. GameMap Class ---
class GameMap:
    def __init__(self, map_filename: str, config: GameConfig):
        self.config = config # Composition: GameMap has-a GameConfig reference
        self.val_data = np.zeros((self.config.HEIGHT, self.config.WIDTH, 4))
        self.start_point = np.array([0, 0], dtype=float)
        self.map_objects_raw = [] # Store raw parsed objects (walls, holes, checkpoint data)
        self.checkpoint_data = [] # To be used by GameApp to create Checkpoint instances

        self._load_map_from_file(map_filename)
        self._generate_val_data()

    def _load_map_from_file(self, filename: str):
        with open(filename, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split("\t")
                if parts[0] == 's':
                    self.start_point = np.array([int(parts[1]), int(parts[2])], dtype=float)
                elif parts[0] == 'r':
                    # Note: HOLERADIUS is now primarily from config, but map can override if needed
                    # For this design, let's stick to config for simplicity unless explicitly required
                    pass
                else:
                    self.map_objects_raw.append(parts)
        
        # Parse checkpoint names from config
        parsed_checkpoint_names = self.config.checkpoint_names_str.split("\t")
        checkpoint_counter_for_name = 0
        for obj in self.map_objects_raw:
            if obj[0] == 'c':
                # Store checkpoint specific data for GameApp to create instances
                if checkpoint_counter_for_name < len(parsed_checkpoint_names):
                    name = parsed_checkpoint_names[checkpoint_counter_for_name]
                else:
                    name = f"CP{checkpoint_counter_for_name+1}" # Fallback name
                self.checkpoint_data.append({
                    'x': int(obj[1]),
                    'y': int(obj[2]),
                    'name': name,
                    'map_index': checkpoint_counter_for_name # Index used in val_data
                })
                checkpoint_counter_for_name += 1


    def _generate_val_data(self):
        # Reuse existing val_data generation logic here
        # This includes circle_hole, circle_wall templates, and all loops
        # that fill self.val_data based on self.map_objects_raw.

        # Create template mask for circle for the holes
        circle_hole = np.zeros((self.config.HOLERADIUS * 2, self.config.HOLERADIUS * 2))
        Y, X = np.ogrid[:self.config.HOLERADIUS * 2, :self.config.HOLERADIUS * 2]
        mask_hole = ((X - self.config.HOLERADIUS + 0.5) ** 2) + ((Y - self.config.HOLERADIUS + 0.5) ** 2) <= self.config.HOLERADIUS ** 2
        circle_hole[mask_hole] = -1 # corner surroundings of walls are also marked as such

        # generating a 2D array e.g. (-2, -1, 0, 1, 2) for x and y
        idx = np.indices((self.config.HOLERADIUS * 2, self.config.HOLERADIUS * 2)) - self.config.HOLERADIUS
        idx[idx >=0] += 1
        circle_hole = np.stack((circle_hole, (idx[0]) * circle_hole / self.config.RADIUS, (idx[1]) * circle_hole / self.config.RADIUS), axis=-1)

        norm_squared = np.sum(circle_hole[:, :, 1:3] ** 2, axis=-1, keepdims=True)
        norm_squared[norm_squared == 0] = 1e-1
        circle_hole[:, :, 1:3] /= norm_squared

        inner_mask = ((X - self.config.HOLERADIUS + 0.5) ** 2) + ((Y - self.config.HOLERADIUS + 0.5) ** 2) <= (self.config.HOLERADIUS - self.config.RADIUS) ** 2
        circle_hole[inner_mask, 0] = -2

        # Create template mask for whole circle for wallcorners
        circle_wall = np.zeros((2*self.config.RADIUS + 1, 2*self.config.RADIUS + 1))
        Y, X = np.ogrid[:2*self.config.RADIUS + 1, :2*self.config.RADIUS + 1]
        mask_corner = ((X - self.config.RADIUS) ** 2) + ((Y - self.config.RADIUS) ** 2) <= self.config.RADIUS ** 2
        circle_wall[mask_corner] = 1 # corner surroundings of walls are also marked as such

        # generating a 2D array e.g. (-2, -1, 0, 1, 2) for x and y
        idx = np.indices((2*self.config.RADIUS + 1, 2*self.config.RADIUS + 1))
        circle_wall = np.stack((circle_wall, (idx[0] - self.config.RADIUS) * circle_wall / self.config.RADIUS, (idx[1]- self.config.RADIUS) * circle_wall / self.config.RADIUS, np.zeros((2*self.config.RADIUS + 1, 2*self.config.RADIUS + 1))), axis=-1)

        # Fill area of objects
        checkpoint_counter_val_data = 0
        for obj in self.map_objects_raw:
            if obj[0] == 'w':
                x1, y1, x2, y2 = int(obj[1]), int(obj[2]), int(obj[3]), int(obj[4])
                self.val_data[y1:y2, x1:x2, 0] = 2

            elif obj[0] == 'h':
                x, y = int(obj[1]), int(obj[2])
                subdata = self.val_data[y - self.config.HOLERADIUS:y + self.config.HOLERADIUS, x - self.config.HOLERADIUS:x + self.config.HOLERADIUS]
                # Ensure the slice dimensions match the template before assignment
                # This accounts for map edges where the hole might be clipped
                template_slice = circle_hole[:subdata.shape[0], :subdata.shape[1]]
                mask = mask_hole[:subdata.shape[0], :subdata.shape[1]]

                subdata[mask, :3] = template_slice[mask, :3]
                self.val_data[y - self.config.HOLERADIUS:y + self.config.HOLERADIUS, x - self.config.HOLERADIUS:x + self.config.HOLERADIUS] = subdata
            
            elif obj[0] == 'c':
                x, y = int(obj[1]), int(obj[2])
                Y, X = np.ogrid[:self.config.HEIGHT, :self.config.WIDTH]
                mask_checkpoint = (((X - x) ** 2)) + ((Y - y) ** 2) <= self.config.HOLERADIUS ** 2
                self.val_data[mask_checkpoint, 0] = -3
                self.val_data[mask_checkpoint, 3] = checkpoint_counter_val_data
                checkpoint_counter_val_data += 1

        # Fill area of shifted rectangles with 1 and add the normalvector
        for obj in self.map_objects_raw:
            if obj[0] == 'w':
                x1, y1, x2, y2 = int(obj[1]), int(obj[2]), int(obj[3]), int(obj[4])

                # Create rectangle above - normalvector points upwards
                # Ensure slicing stays within bounds and dimensions match
                y_start, y_end = max(0, y1 - self.config.RADIUS), y1
                x_start, x_end = max(0, x1), min(self.config.WIDTH, x2)
                if y_end > y_start and x_end > x_start:
                    subdata = self.val_data[y_start:y_end, x_start:x_end]
                    # Create a mask that matches the current subdata slice size for assignment
                    mask_relevant = subdata[:,:, 0] <= 0
                    subdata[mask_relevant, 0] = 1
                    subdata[mask_relevant] += np.array([0, -1, 0, 0])
                    self.val_data[y_start:y_end, x_start:x_end] = subdata

                # Create rectangle below - normalvector points downwards
                y_start, y_end = y2, min(self.config.HEIGHT, y2 + self.config.RADIUS)
                x_start, x_end = max(0, x1), min(self.config.WIDTH, x2)
                if y_end > y_start and x_end > x_start:
                    subdata = self.val_data[y_start:y_end, x_start:x_end]
                    mask_relevant = subdata[:,:, 0] <= 0
                    subdata[mask_relevant, 0] = 1
                    subdata[mask_relevant] += np.array([0, 1, 0, 0])
                    self.val_data[y_start:y_end, x_start:x_end] = subdata

                # Create rectangle left - normalvector points left
                y_start, y_end = max(0, y1), min(self.config.HEIGHT, y2)
                x_start, x_end = max(0, x1 - self.config.RADIUS), x1
                if y_end > y_start and x_end > x_start:
                    subdata = self.val_data[y_start:y_end, x_start:x_end]
                    mask_relevant = subdata[:,:, 0] <= 0
                    subdata[mask_relevant, 0] = 1
                    subdata[mask_relevant] += np.array([0, 0, -1, 0])
                    self.val_data[y_start:y_end, x_start:x_end] = subdata

                # Create rectangle right - normalvector points right
                y_start, y_end = max(0, y1), min(self.config.HEIGHT, y2)
                x_start, x_end = x2, min(self.config.WIDTH, x2 + self.config.RADIUS)
                if y_end > y_start and x_end > x_start:
                    subdata = self.val_data[y_start:y_end, x_start:x_end]
                    mask_relevant = subdata[:,:, 0] <= 0
                    subdata[mask_relevant, 0] = 1
                    subdata[mask_relevant] += np.array([0, 0, 1, 0])
                    self.val_data[y_start:y_end, x_start:x_end] = subdata

        # Corners
        for obj in self.map_objects_raw:
            if obj[0] == 'w':
                x1, y1, x2, y2 = int(obj[1]), int(obj[2]), int(obj[3]), int(obj[4])
                
                # Top left corner
                y_sub_start, y_sub_end = max(0, y1 - self.config.RADIUS), y1
                x_sub_start, x_sub_end = max(0, x1 - self.config.RADIUS), x1
                template_y_start = self.config.RADIUS - (y1 - y_sub_start)
                template_x_start = self.config.RADIUS - (x1 - x_sub_start)

                if y_sub_end > y_sub_start and x_sub_end > x_sub_start:
                    subdata = self.val_data[y_sub_start:y_sub_end, x_sub_start:x_sub_end]
                    template_slice = circle_wall[template_y_start:template_y_start + subdata.shape[0], template_x_start:template_x_start + subdata.shape[1]]
                    mask = np.isin(subdata[:,:, 0], [0, -1])
                    subdata[mask] = template_slice[mask]
                    self.val_data[y_sub_start:y_sub_end, x_sub_start:x_sub_end] = subdata

                # Top right corner
                y_sub_start, y_sub_end = max(0, y1 - self.config.RADIUS), y1
                x_sub_start, x_sub_end = x2, min(self.config.WIDTH, x2 + self.config.RADIUS)
                template_y_start = self.config.RADIUS - (y1 - y_sub_start)
                template_x_start = 0 # Corresponds to the start of the right slice of template
                
                if y_sub_end > y_sub_start and x_sub_end > x_sub_start:
                    subdata = self.val_data[y_sub_start:y_sub_end, x_sub_start:x_sub_end]
                    template_slice = circle_wall[template_y_start:template_y_start + subdata.shape[0], self.config.RADIUS + 1:self.config.RADIUS + 1 + subdata.shape[1]]
                    mask = np.isin(subdata[:,:, 0], [0, -1])
                    subdata[mask] = template_slice[mask]
                    self.val_data[y_sub_start:y_sub_end, x_sub_start:x_sub_end] = subdata
                
                # Bottom left corner
                y_sub_start, y_sub_end = y2, min(self.config.HEIGHT, y2 + self.config.RADIUS)
                x_sub_start, x_sub_end = max(0, x1 - self.config.RADIUS), x1
                template_y_start = 0 # Corresponds to the start of the bottom slice of template
                template_x_start = self.config.RADIUS - (x1 - x_sub_start)

                if y_sub_end > y_sub_start and x_sub_end > x_sub_start:
                    subdata = self.val_data[y_sub_start:y_sub_end, x_sub_start:x_sub_end]
                    template_slice = circle_wall[self.config.RADIUS + 1:self.config.RADIUS + 1 + subdata.shape[0], template_x_start:template_x_start + subdata.shape[1]]
                    mask = np.isin(subdata[:,:, 0], [0, -1])
                    subdata[mask] = template_slice[mask]
                    self.val_data[y_sub_start:y_sub_end, x_sub_start:x_sub_end] = subdata

                # Bottom right corner
                y_sub_start, y_sub_end = y2, min(self.config.HEIGHT, y2 + self.config.RADIUS)
                x_sub_start, x_sub_end = x2, min(self.config.WIDTH, x2 + self.config.RADIUS)
                template_y_start = 0 # Corresponds to the start of the bottom slice of template
                template_x_start = 0 # Corresponds to the start of the right slice of template

                if y_sub_end > y_sub_start and x_sub_end > x_sub_start:
                    subdata = self.val_data[y_sub_start:y_sub_end, x_sub_start:x_sub_end]
                    template_slice = circle_wall[self.config.RADIUS + 1:self.config.RADIUS + 1 + subdata.shape[0], self.config.RADIUS + 1:self.config.RADIUS + 1 + subdata.shape[1]]
                    mask = np.isin(subdata[:,:, 0], [0, -1])
                    subdata[mask] = template_slice[mask]
                    self.val_data[y_sub_start:y_sub_end, x_sub_start:x_sub_end] = subdata


    def get_collision_info(self, x: int, y: int) -> np.ndarray:
        """
        Public: Provides collision information for a given (x, y) coordinate.
        Called by: Ball.update_position
        """
        # Ensure coordinates are within bounds
        if not (0 <= y < self.config.HEIGHT and 0 <= x < self.config.WIDTH):
            # Out of bounds, treat as a special collision type (e.g., fall off map)
            return np.array([-2, 0, 0, 0]) # -2 can signify falling off the world too
        return self.val_data[y, x, :]

    def get_start_point(self) -> np.ndarray:
        """
        Public: Returns the ball's start coordinates.
        Called by: GameApp (for Ball initialization), Ball (for reset after falling in hole)
        """
        return self.start_point.copy()

    def get_raw_map_objects(self) -> list:
        """
        Public: Returns the raw parsed map objects for drawing static elements.
        Called by: GameApp
        """
        return self.map_objects_raw
    
    def get_checkpoint_init_data(self) -> list:
        """
        Public: Returns data needed to initialize Checkpoint objects.
        Called by: GameApp
        """
        return self.checkpoint_data


# --- 3. Ball Class ---
class Ball:
    def __init__(self, start_pos: np.ndarray, canvas: tk.Canvas, config: GameConfig):
        self.config = config # Composition: Ball has-a GameConfig reference
        self.pos = start_pos.copy() # Composition: Ball has-a position (NumPy array)
        self.vel = np.array([0, 0], dtype=float) # Composition: Ball has-a velocity (NumPy array)
        self.radius = self.config.RADIUS
        self.canvas = canvas # Composition: Ball has-a reference to the canvas it draws on

        self.canvas_id = self.canvas.create_oval(
            int(self.pos[0]) - self.radius + 1, int(self.pos[1]) - self.radius + 1,
            int(self.pos[0]) + self.radius, int(self.pos[1]) + self.radius,
            fill="blue", outline="blue"
        )
        self.current_color = "blue"

    def update_position(self, accel_x: float, accel_y: float, game_map: GameMap, checkpoints: list):
        """
        Public: Updates ball's position, handles physics and collisions.
        Called by: GameApp._game_loop
        Dependencies: game_map (uses-a GameMap), checkpoints (uses-a list of Checkpoints)
        """
        self.vel[0] += self.config.ACC_SCALE * accel_x * self.config.DT / 1000
        self.vel[1] += self.config.ACC_SCALE * accel_y * self.config.DT / 1000
        
        Dpos = self.vel * self.config.DT / 1000
        dist = np.linalg.norm(Dpos)
        steps = int(dist / self.config.DP) if dist > self.config.DP else 1
        
        dstep = Dpos / steps
        counter = 0
        security_counter = 0 # To prevent infinite loops in rare cases

        while counter < steps:
            security_counter += 1
            if security_counter > 10000: # Increased security limit slightly
                # print("Security limit reached in Ball.update_position, breaking loop")
                # This could indicate an issue or extreme situation, consider logging
                break

            temp_pos = self.pos + dstep
            
            # Clamp temp_pos to map boundaries before querying game_map
            clamped_x = int(np.clip(temp_pos[0], 0, self.config.WIDTH - 1))
            clamped_y = int(np.clip(temp_pos[1], 0, self.config.HEIGHT - 1))

            val_info = game_map.get_collision_info(clamped_x, clamped_y)
            px_type = val_info[0]

            if px_type > 0: # Wall or Wall surrounding (type 1 or 2)
                vec_norm = val_info[1:3][::-1] # Normal vector (y, x) -> (x, y)
                pos_dot_product = np.dot(vec_norm, Dpos)
                
                if (pos_dot_product < 0): # Only if moving towards the wall
                    vec_proj_pos = pos_dot_product / np.dot(vec_norm, vec_norm) * vec_norm
                    vec_proj_vel = np.dot(vec_norm, self.vel) / np.dot(vec_norm, vec_norm) * vec_norm
                    
                    Dpos = - 2 * vec_proj_pos + Dpos
                    self.vel = - 2 * vec_proj_vel + self.vel

                    Dpos *= self.config.DAMPING # Apply damping to remaining displacement
                    self.vel *= self.config.DAMPING # Apply damping to velocity

                    # Recalculate steps with new Dpos to handle new direction/magnitude
                    dist = np.linalg.norm(Dpos)
                    steps = int(dist / self.config.DP) if dist > self.config.DP else 1
                    dstep = Dpos / steps
                    counter = 0 # Restart collision checks for the new direction
                    continue # Continue the while loop

                else: # Moving parallel or away from wall, slide along
                    shift = vec_norm / np.linalg.norm(vec_norm) * self.config.DP
                    self.pos += shift
                    Dpos -= shift
                    counter += 1
                    continue
            elif px_type == -2: # Inner hole (fall through)
                self.reset(game_map.get_start_point()) # Teleport to start
                self.set_color("blue") # Reset color if it was gold
                return # Stop processing movement for this update
            elif px_type == -3: # Checkpoint
                c_number = int(val_info[3])
                if c_number < len(checkpoints):
                    checkpoint = checkpoints[c_number]
                    if not checkpoint.is_reached:
                        checkpoint.mark_reached()
                        # GameApp will track overall checkpoint progress
            
            self.pos += dstep
            Dpos -= dstep
            counter += 1

        # Check final position for hole border effect
        # Clamping final position before checking map data
        clamped_x_final = int(np.clip(self.pos[0], 0, self.config.WIDTH - 1))
        clamped_y_final = int(np.clip(self.pos[1], 0, self.config.HEIGHT - 1))

        final_val_info = game_map.get_collision_info(clamped_x_final, clamped_y_final)
        if final_val_info[0] == -1: # Hole border
            vec_norm_final = final_val_info[1:3][::-1]
            vec_tang = np.array([vec_norm_final[1], -vec_norm_final[0]])
            
            # Project velocity onto normal and tangential components
            vec_proj_vel_norm = np.dot(vec_norm_final, self.vel) / np.dot(vec_norm_final, vec_norm_final) * vec_norm_final
            vec_proj_vel_tang = np.dot(vec_tang, self.vel) / np.dot(vec_tang, vec_tang) * vec_tang

            # Apply forces/damping (this is a simplified model, adjust as needed)
            self.vel += vec_norm_final * 10 # Push away from hole center
            self.vel -= vec_proj_vel_tang * 0.3 # Dampen tangential velocity
    
    def draw(self):
        """
        Public: Updates the ball's visual position on the canvas.
        Called by: GameApp._game_loop
        """
        self.canvas.coords(
            self.canvas_id,
            int(self.pos[0]) - self.radius + 1, int(self.pos[1]) - self.radius + 1,
            int(self.pos[0]) + self.radius, int(self.pos[1]) + self.radius
        )

    def set_color(self, color: str):
        """
        Public: Changes the ball's color.
        Called by: GameApp (on win), Ball (on reset)
        """
        if self.current_color != color:
            self.canvas.itemconfig(self.canvas_id, fill=color, outline=color)
            self.current_color = color


    def reset(self, start_pos: np.ndarray):
        """
        Public: Resets the ball's position, velocity, and color.
        Called by: GameApp.reset_game, Ball.update_position (on falling in inner hole)
        """
        self.pos = start_pos.copy()
        self.vel = np.array([0, 0], dtype=float)
        self.set_color("blue")


# --- 4. Checkpoint Class ---
class Checkpoint:
    def __init__(self, x: int, y: int, radius: int, name: str, canvas: tk.Canvas):
        self.position = np.array([x, y], dtype=float)
        self.radius = radius
        self.name = name
        self.is_reached = False
        self.canvas = canvas # Composition: Checkpoint has-a reference to the canvas

        self.canvas_oval_id = self.canvas.create_oval(
            x - radius, y - radius, x + radius, y + radius,
            fill="orange", outline="lightgray"
        )
        self.canvas_text_id = self.canvas.create_text(
            x, y, text="", fill="black", font=("Arial", 10, "bold")
        )

    def mark_reached(self):
        """
        Public: Marks the checkpoint as reached and updates its visual state.
        Called by: Ball.update_position
        """
        if not self.is_reached:
            self.is_reached = True
            self.canvas.itemconfig(self.canvas_oval_id, fill="#0CFF0B") # Green
            self.canvas.itemconfig(self.canvas_text_id, text=self.name)

    def reset(self):
        """
        Public: Resets the checkpoint's status and visual state.
        Called by: GameApp.reset_game
        """
        self.is_reached = False
        self.canvas.itemconfig(self.canvas_oval_id, fill="orange")
        self.canvas.itemconfig(self.canvas_text_id, text="")


# --- 5. InputHandler Classes ---
class InputHandler(ABC):
    @abstractmethod
    def get_acceleration(self) -> tuple[float, float]:
        pass

class KeyboardInputHandler(InputHandler):
    def __init__(self, window: tk.Tk):
        self._pressed_keys = set() # Protected: Internal state
        window.bind("<KeyPress>", self._on_key_press) # Protected methods as callbacks
        window.bind("<KeyRelease>", self._on_key_release)

    def _on_key_press(self, event: tk.Event):
        self._pressed_keys.add(event.keysym)

    def _on_key_release(self, event: tk.Event):
        self._pressed_keys.discard(event.keysym)

    def get_acceleration(self) -> tuple[float, float]:
        """
        Public: Returns acceleration based on pressed keys.
        Called by: GameApp._game_loop
        """
        ax = 0
        ay = 0
        if "Left" in self._pressed_keys:
            ax = -5
        elif "Right" in self._pressed_keys:
            ax = 5
        
        if "Down" in self._pressed_keys:
            ay = 5
        elif "Up" in self._pressed_keys:
            ay = -5
        return ax, ay

class MPU6050InputHandler(InputHandler):
    def __init__(self):
        if mpu6050 is None:
            raise RuntimeError("MPU6050 library not available. Cannot use MPU6050InputHandler.")
        self._sensor = mpu6050(0x68) # Protected: Internal sensor object

    def get_acceleration(self) -> tuple[float, float]:
        """
        Public: Returns acceleration from MPU6050 sensor.
        Called by: GameApp._game_loop
        """
        current_acc = self._sensor.get_accel_data()
        return current_acc['y'], -current_acc['x']


# --- 6. FanController Class ---
class FanController:
    def __init__(self, config: GameConfig):
        self.config = config # Composition: FanController has-a GameConfig reference
        self._is_running = False # Protected: Internal control flag
        self._cpu_temp_list = np.full(self.config.cooling_average_time * 2, np.nan) # Longer buffer
        self._last_fan_speeds = np.full(self.config.cooling_hyst_time, np.nan)
        self.current_fan_speed = 0 # Public, but primarily updated internally

        if GPIO is None and sys.platform == "linux":
            print("Warning: RPi.GPIO not found for FanController. Fan control disabled.")
        elif GPIO:
            GPIO.setup(self.config.FANGPIO, GPIO.OUT)
            self._pwm_thread = threading.Thread(target=self._pwm_worker, daemon=True)
            self._general_thread = threading.Thread(target=self._general_worker, daemon=True)
        else: # Not Linux or GPIO not imported
            self._pwm_thread = None
            self._general_thread = None


    def _set_gpio_high(self, pin):
        if GPIO: GPIO.output(pin, GPIO.HIGH)
    def _set_gpio_low(self, pin):
        if GPIO: GPIO.output(pin, GPIO.LOW)

    def _pwm_worker(self):
        # Private: Runs in a separate thread.
        if self._pwm_thread is None: return # Should not happen if thread exists
        while self._is_running:
            if self.current_fan_speed == 0:
                self._set_gpio_low(self.config.FANGPIO)
                time.sleep(0.05)
            elif self.current_fan_speed == 100:
                self._set_gpio_high(self.config.FANGPIO)
                time.sleep(0.05)
            else:
                delay1 = self.current_fan_speed * 1E-5
                delay2 = (100 - self.current_fan_speed) * 1E-5
                self._set_gpio_high(self.config.FANGPIO)
                time.sleep(delay1)
                self._set_gpio_low(self.config.FANGPIO)
                time.sleep(delay2)
        self._set_gpio_low(self.config.FANGPIO) # Ensure fan off on shutdown

    def _get_cpu_temp(self):
        if sys.platform == "linux":
            try:
                with open("/sys/class/thermal/thermal_zone0/temp", "r") as f:
                    return int(f.read()) / 1000.0
            except FileNotFoundError:
                # print("CPU temperature file not found. Is this a Raspberry Pi?")
                pass
        return np.nan # Return NaN if not on RPi or file not found

    def _general_worker(self):
        # Private: Runs in a separate thread.
        if self._general_thread is None: return # Should not happen if thread exists
        while self._is_running:
            # Update CPU temp list
            self._cpu_temp_list = np.roll(self._cpu_temp_list, -1)
            self._cpu_temp_list[-1] = self._get_cpu_temp()

            # Calculate moving average temp
            valid_temps = self._cpu_temp_list[~np.isnan(self._cpu_temp_list[-self.config.cooling_average_time:])]
            mtemp = np.mean(valid_temps) if valid_temps.size > 0 else 20 # Default if no data

            # Determine new speed based on thresholds
            new_speed = self.config.fan_speed_levels[0] # Default to lowest speed
            for i in range(len(self.config.temp_threshold) - 1):
                if self.config.temp_threshold[i+1] >= mtemp > self.config.temp_threshold[i]:
                    new_speed = self.config.fan_speed_levels[i]
                    break

            # Apply hysteresis (max of recent speeds)
            self._last_fan_speeds = np.roll(self._last_fan_speeds, -1)
            self._last_fan_speeds[-1] = new_speed
            self.current_fan_speed = int(np.nanmax(self._last_fan_speeds)) # Use int for PWM

            time.sleep(1)

    def start(self):
        """
        Public: Starts the fan control threads.
        Called by: GameApp
        """
        if self._pwm_thread and self._general_thread:
            self._is_running = True
            if not self._pwm_thread.is_alive():
                self._pwm_thread.start()
            if not self._general_thread.is_alive():
                self._general_thread.start()
        else:
            print("Fan control threads not initialized (e.g., not on Linux or GPIO missing).")

    def stop(self):
        """
        Public: Stops the fan control threads and cleans up GPIO.
        Called by: GameApp.close_app
        """
        if self._pwm_thread and self._general_thread and self._is_running:
            self._is_running = False
            self._pwm_thread.join(timeout=2)
            self._general_thread.join(timeout=2)
            if GPIO:
                GPIO.cleanup()
            print("Fan controller stopped and GPIO cleaned up.")


# --- 7. GameApp Class (The Orchestrator) ---
class GameApp:
    def __init__(self, config_file_path="config.json", map_file_path="map_v1.dat"):
        self.config = GameConfig(config_file_path) # Composition: GameApp has-a GameConfig

        # Tkinter setup
        self.window = tk.Tk()
        self.window.title("Tilt Maze Game")
        self.window.geometry(f"{self.config.WIDTH}x{self.config.HEIGHT}")
        if self.config.control_mode == "mpu6050":
            self.window.attributes('-fullscreen', True)
        self.window.focus_force()

        self.canvas = tk.Canvas(self.window, width=self.config.WIDTH, height=self.config.HEIGHT, bg="white")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.config(cursor="none")

        # Game Components (Composition)
        self.game_map = GameMap(map_file_path, self.config) # GameApp has-a GameMap
        self.ball = Ball(self.game_map.get_start_point(), self.canvas, self.config) # GameApp has-a Ball

        self.checkpoints = [] # List of Checkpoint instances
        self._init_checkpoints() # Initialize checkpoint objects after map is loaded

        # Input Handler (Composition & Polymorphism)
        if self.config.control_mode == "keyboard":
            self.input_handler = KeyboardInputHandler(self.window)
        elif self.config.control_mode == "mpu6050":
            try:
                self.input_handler = MPU6050InputHandler()
            except RuntimeError as e:
                print(e)
                self.input_handler = KeyboardInputHandler(self.window) # Fallback

        # Fan Controller (Composition)
        self.fan_controller = FanController(self.config) # GameApp has-a FanController

        # UI Elements
        self._init_ui_elements()

        # Bindings
        self.window.bind("<Escape>", self._end_fullscreen)
        if self.config.control_mode == "keyboard":
            # KeyboardInputHandler already binds its own keys
            pass

        self.game_is_running = True # Control flag for the game loop (Tkinter after loop)

    def _init_checkpoints(self):
        """Initializes Checkpoint objects based on GameMap data."""
        for cp_data in self.game_map.get_checkpoint_init_data():
            cp = Checkpoint(
                cp_data['x'], cp_data['y'],
                self.config.HOLERADIUS,
                cp_data['name'],
                self.canvas
            )
            self.checkpoints.append(cp)

    def _init_ui_elements(self):
        # Draw static map elements first (walls, holes)
        for obj in self.game_map.get_raw_map_objects():
            if obj[0] == 'w':
                self.canvas.create_rectangle(int(obj[1]), int(obj[2]), int(obj[3]), int(obj[4]), fill="#471F01", outline="#471F01")
            elif obj[0] == 'h':
                self.canvas.create_oval(int(obj[1]) - self.config.HOLERADIUS, int(obj[2]) - self.config.HOLERADIUS,
                                        int(obj[1]) + self.config.HOLERADIUS, int(obj[2]) + self.config.HOLERADIUS,
                                        fill="black", outline="lightgray")
            # Checkpoints are drawn by their own class via _init_checkpoints

        close_button = tk.Button(self.window, text="✕", command=self.close_app, font=("Arial", 14, "bold"), bg="red", fg="white", bd=0, relief="flat", cursor="hand2")
        close_button.place(x=self.config.WIDTH - 20, y=0, width=20, height=20)

        reset_button = tk.Button(self.window, text="\u27F3", command=self.reset_game, font=("Arial", 14, "bold"), bg="blue", fg="white", bd=0, relief="flat", cursor="hand2")
        reset_button.place(x=0, y=0, width=20, height=20)
    
    def _go_fullscreen(self):
        self.window.attributes('-fullscreen', True)

    def _end_fullscreen(self, event=None):
        self.window.attributes('-fullscreen', False)

    def _game_loop(self):
        """
        Private: The main game update loop, called by Tkinter's after method.
        Orchestrates interactions between game components.
        """
        if not self.game_is_running:
            return # Stop loop if game is shutting down

        # Get acceleration from input handler
        ax, ay = self.input_handler.get_acceleration()

        # Update ball position and handle collisions.
        # Ball.update_position handles interactions with GameMap and Checkpoints
        self.ball.update_position(ax, ay, self.game_map, self.checkpoints)
        self.ball.draw() # Ensure ball is drawn after its position updates

        # Check if all checkpoints are reached
        all_checkpoints_reached = all(cp.is_reached for cp in self.checkpoints)
        if all_checkpoints_reached:
            self.ball.set_color("gold") # Gold signifies completion

        # Schedule next update
        self.window.after(self.config.DT, self._game_loop)

    def run(self):
        """
        Public: Starts the game application.
        """
        self.fan_controller.start() # Start fan threads
        self.window.after(self.config.DT, self._game_loop) # Start game loop
        if self.config.control_mode == "mpu6050":
            self.window.after(100, self._go_fullscreen) # Go fullscreen shortly after start
        self.window.mainloop()

    def reset_game(self):
        """
        Public: Resets the game to its initial state.
        """
        self.ball.reset(self.game_map.get_start_point())
        for cp in self.checkpoints:
            cp.reset()
        # Ball color reset is handled by ball.reset()

    def close_app(self):
        """
        Public: Gracefully shuts down the application.
        """
        self.game_is_running = False # Signal game loop to stop
        self.fan_controller.stop() # Stop fan threads
        self.window.destroy() # Close Tkinter window

# --- Main execution block ---
if __name__ == "__main__":
    app = GameApp()
    app.run()

    # MPL Debug plot (if enabled in config and app ran successfully)
    if app.config.mpl_Debug:
        try:
            import matplotlib.pyplot as plt
            plt.imshow(-app.game_map.val_data[:,:,0], cmap='gray', vmin=-3, vmax=4)
            plt.title("Game Map val_data (Type Layer)")
            plt.show()

            # Optional quiver plot
            # plot_data = app.game_map.val_data
            # step = 10  # plot every 10th pixel
            # Y_q, X_q = np.mgrid[0:plot_data.shape[0]:step, 0:plot_data.shape[1]:step]
            # U_q = plot_data[::step, ::step, 2]  # x-component
            # V_q = plot_data[::step, ::step, 1]  # y-component

            # plt.figure(figsize=(10, 6))
            # plt.quiver(X_q, Y_q, U_q, V_q, angles='xy', scale_units='xy', scale=1, color='red')
            # plt.gca().invert_yaxis()
            # plt.title("Gradient Field (Normal Vectors)")
            # plt.axis('equal')
            # plt.show()
        except ImportError:
            print("Matplotlib not installed. Cannot display debug plots.")