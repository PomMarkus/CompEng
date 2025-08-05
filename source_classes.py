import json
import tkinter as tk
import threading
import time
import sys
import numpy as np
from abc import ABC, abstractmethod

if sys.platform == "linux":
    try:
        import RPi.GPIO as GPIO
        GPIO.setmode(GPIO.BCM)
    except ImportError:
        print("RPi.GPIO module not found")
        GPIO = None  # Set GPIO to None if not available
    try:
        from mpu6050 import mpu6050
    except ImportError:
        print("mpu6050 module not found")
        mpu6050 = None
else:
    GPIO = None
    mpu6050 = None

try:
    import paho.mqtt.client as mqtt
except ImportError:
    print("paho-mqtt library not found. MQTT functionality disabled.")
    mqtt = None

# --- Configuration Class ---

class GameConfig:
    def __init__(self, config_file_name="config.json"):
        with open(config_file_name, "r") as file:
            config_data = json.load(file)
        self.control_mode = config_data.get("control", "keyboard")
        self.mpl_debug = config_data.get("mpl_debug", False)
        self.checkpoint_namestring = config_data.get("checkpoints", "1G\t9A\t7M\t0E")
        self.time_step_size = config_data.get("time_step_size", 20)
        self.position_step_size = config_data.get("position_step_size", 0.1)
        self.acceleration_scale = config_data.get("acceleration_scale", 100)
        self.damping_factor = config_data.get("damping_factor", 0.8)
        self.ball_radius = config_data.get("ball_radius", 10)
        self.hole_radius = config_data.get("hole_radius", 12)
        self.map_file_name = config_data.get("map_file_name", "map_v1.txt")
        self.screen_width = config_data.get("screen_width", 800)
        self.screen_height = config_data.get("screen_height", 480)
        self.fan_gpio = config_data.get("fan_gpio", 15)
        self.vibration_gpio = config_data.get("vibration_gpio", 14)        

        if self.control_mode not in ["keyboard", "mpu6050"]:
            raise ValueError(f"Invalid control mode: {self.control_mode}. Choose 'keyboard' or 'mpu6050'.")

        if self.control_mode == "mpu6050" and mpu6050 is None:
            raise ImportError("mpu6050 module is required for 'mpu6050' control mode.")
        

# --- Control CLasses ---

class InputControl(ABC):
    @abstractmethod
    def get_acceleration(self) -> tuple[float, float]:
        pass

class KeyboardControl(InputControl):
    def __init__(self, window: tk.Tk):
        self.pressend_keys = set()
        window.bind("<KeyPress>", self._on_key_press)
        window.bind("<KeyRelease>", self._on_key_release)

    def _on_key_press(self, event):
        self.pressend_keys.add(event.keysym)
    
    def _on_key_release(self, event):
        self.pressend_keys.discard(event.keysym)

    def get_acceleration(self) -> tuple[float, float]:
        """_summary_

        Returns:
            tuple[float, float]: _description_
        """
        acc_x = 0.0
        acc_y = 0.0

        if "Up" in self.pressend_keys:
            acc_y -= 5
        elif "Down" in self.pressend_keys:
            acc_y += 5
        if "Left" in self.pressend_keys:
            acc_x -= 5
        elif "Right" in self.pressend_keys:
            acc_x += 5

        return (acc_x, acc_y)
    

class MPU6050Control(InputControl):
    def __init__(self):
        if mpu6050 is None:
            raise ImportError("mpu6050 module not found")
        self.sensor = mpu6050(0x68)
    
    def get_acceleration(self) -> tuple[float, float]:
        """_summary_

        Returns:
            tuple[float, float]: _description_
        """
        accel_data = self.sensor.get_accel_data()
        return (accel_data['y'], -accel_data['x']) # Compensates sensor orientation
    

# --- Game Map Class ---
class GameMap:
    def __init__(self, map_filename: str, config: GameConfig):
        self.config = config
        self.val_data = np.zeros((self.config.screen_height, self.config.screen_width, 4))
        self.start_point = np.array([0, 0], dtype=float)
        self.map_objects = []
        self.checkpoints = []

        self._load_map_objects(map_filename)
        self._generate_val_data()
    
    def _load_map_objects(self, map_filename: str):
        with open(map_filename, "r") as file:
            for line in file:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if parts[0] == "s":
                    self.start_point = np.array([int(parts[1]), int(parts[2])], dtype=float)
                elif parts[0] == "r":
                    self.config.hole_radius = int(parts[1])
                else:
                    self.map_objects.append(parts)

        parsed_checkpoint_names = self.config.checkpoint_namestring.split("\t")

        for i, obj in enumerate(self.map_objects):
            if obj[0] == 'c':
                if i < len(parsed_checkpoint_names):
                    name = parsed_checkpoint_names[i]
                else:
                    name = "-"
                self.checkpoints.append({
                    "x": int(obj[1]),
                    "y": int(obj[2]),
                    "name": name
                })
                
    def _generate_val_data(self):
        # 2D np array
        self.val_data = np.zeros((self.config.screen_height, self.config.screen_width, 4))

        # Create template mask for circle for the holes
        circle_hole = np.zeros((self.config.hole_radius * 2, self.config.hole_radius * 2))
        Y, X = np.ogrid[:self.config.hole_radius * 2, :self.config.hole_radius * 2]
        mask_hole = ((X - self.config.hole_radius + 0.5) ** 2) + ((Y - self.config.hole_radius + 0.5) ** 2) <= self.config.hole_radius ** 2
        circle_hole[mask_hole] = -1 # corner surroundings of walls are also marked as such

        # generating a 2D array e.g. (-2, -1, 0, 1, 2) for x and y
        idx = np.indices((self.config.hole_radius * 2, self.config.hole_radius * 2)) - self.config.hole_radius
        idx[idx >=0] += 1
        circle_hole = np.stack((circle_hole, (idx[0]) * circle_hole / self.config.radius, (idx[1]) * circle_hole / self.config.radius), axis=-1)

        norm_squared = np.sum(circle_hole[:, :, 1:3] ** 2, axis=-1, keepdims=True)
        norm_squared[norm_squared == 0] = 1e-1
        circle_hole[:, :, 1:3] /= norm_squared

        inner_mask = ((X - self.config.hole_radius + 0.5) ** 2) + ((Y - self.config.hole_radius + 0.5) ** 2) <= (self.config.hole_radius - self.config.radius) ** 2
        circle_hole[inner_mask, 0] = -2

        checkpoint_counter = 0

        # Fill area of circles
        for obj in self.map_objects:
            # Wall pxl to 2
            if obj[0] == 'w':
                x1, y1, x2, y2 = int(obj[1]), int(obj[2]), int(obj[3]), int(obj[4])
                self.val_data[y1:y2, x1:x2, 0] = 2 # pixle occupied: 2

            # hole pxl to -1 
            elif obj[0] == 'h': # maybe remove this because of redundancy
                x, y = int(obj[1]), int(obj[2])
                
                subdata = self.val_data[y - self.config.hole_radius:y + self.config.hole_radius, x - self.config.hole_radius:x + self.config.hole_radius]
                subdata[mask_hole, :3] = circle_hole[mask_hole, :3]
                self.val_data[y - self.config.hole_radius:y + self.config.hole_radius, x - self.config.hole_radius:x + self.config.hole_radius] = subdata # fill the hole with the circle_hole template
            
            # Checkpoint pxl to -3
            elif obj[0] == 'c':
                x, y = int(obj[1]), int(obj[2])
                # Create a mask for defining the pxl inside the circle
                Y, X = np.ogrid[:self.config.screen_height, :self.config.screen_width]
                mask_checkpoint = (((X - x) ** 2)) + ((Y - y) ** 2)<= self.config.hole_radius ** 2
                # Set the pixels inside the checkoint to -3
                self.val_data[mask_checkpoint, 0] = -3
                self.val_data[mask_checkpoint, 3] = checkpoint_counter
                checkpoint_counter += 1

                

        # Fill area of shifted rectangles with 1 and add the normalvector for the rebouncing calculation
        for obj in self.map_objects:

            if obj[0] == 'w':
                x1, y1, x2, y2 = int(obj[1]), int(obj[2]), int(obj[3]), int(obj[4])
                Y, X = np.ogrid[:self.config.screen_height, :self.config.screen_width]

                # Create rectangle above - normalvector points upwards
                subdata = self.val_data[y1- self.config.radius:y1, x1:x2]
                subdata[subdata[:,:, 0] <= 0, 0] = 1 # making sure, that the area around a wall element is marked as surrounding area only once
                subdata[subdata[:,:, 0] != 2] += np.array([0, -1, 0, 0]) # add the normalvector
                self.val_data[y1- self.config.radius:y1, x1:x2] = subdata

                # Create rectangle below - normalvector points downwards
                subdata = self.val_data[y2:y2+ self.config.radius, x1:x2]
                subdata[subdata[:,:, 0] <= 0, 0] = 1 # making sure, that the area around a wall element is marked as surrounding area only once
                subdata[subdata[:,:, 0] != 2] += np.array([0, 1, 0, 0]) # add the normalvector
                self.val_data[y2:y2+ self.config.radius, x1:x2] = subdata

                # Create rectangle left - normalvector points left
                subdata = self.val_data[y1:y2, x1- self.config.radius:x1]
                subdata[subdata[:,:, 0] <= 0, 0] = 1 # making sure, that the area around a wall element is marked as surrounding area only once
                subdata[subdata[:,:, 0] != 2] += np.array([0, 0, -1, 0]) # add the normalvector
                self.val_data[y1:y2, x1- self.config.radius:x1] = subdata

                # Create rectangle right - normalvector poibts right
                subdata = self.val_data[y1:y2, x2:x2+ self.config.radius]
                subdata[subdata[:,:, 0] <= 0, 0] = 1 # making sure, that the area around a wall element is marked as surrounding area only once
                subdata[subdata[:,:, 0] != 2] += np.array([0, 0, 1, 0]) # add the normalvector
                self.val_data[y1:y2, x2:x2+ self.config.radius] = subdata


        # Create template mask for whole circle for wallcorners
        circle_wall = np.zeros((2* self.config.radius + 1, 2* self.config.radius + 1))
        Y, X = np.ogrid[:2* self.config.radius + 1, :2* self.config.radius + 1]
        mask_corner = ((X - self.config.radius) ** 2) + ((Y - self.config.radius) ** 2) <= self.config.radius ** 2
        circle_wall[mask_corner] = 1 # corner surroundings of walls are also marked as such

        # generating a 2D array e.g. (-2, -1, 0, 1, 2) for x and y
        idx = np.indices((2* self.config.radius + 1, 2* self.config.radius + 1))
        circle_wall = np.stack((circle_wall, (idx[0] - self.config.radius) * circle_wall / self.config.radius, (idx[1]- self.config.radius) * circle_wall / self.config.radius, np.zeros((2*radius + 1, 2*radius + 1))), axis=-1)


        for obj in objects:
            if obj[0] == 'w':
                x1, y1, x2, y2 = int(obj[1]), int(obj[2]), int(obj[3]), int(obj[4])
                # Insert circle sector
                # Top left corner
                mask = np.isin(val_data[y1- self.config.radius:y1, x1- self.config.radius:x1, 0], [0, -1]) # mask for all points within the circle sector at the top left corner

                self.val_data[y1-radius:y1, x1-radius:x1][mask] = circle_wall[: self.config.radius, : self.config.radius][mask]
                # Top right corner
                mask = np.isin(val_data[y1-radius:y1, x2:x2+radius, 0], [0, -1]) # mask for all points within the circle sector at the top right corner
                self.val_data[y1-radius:y1, x2:x2+radius][mask] = circle_wall[: self.config.radius, self.config.radius + 1:][mask]
                # Bottom left corner
                mask = np.isin(val_data[y2:y2+ self.config.radius, x1- self.config.radius:x1, 0], [0, -1]) # mask for all points within the circle sector at the bottom left corner
                self.val_data[y2:y2+radius, x1-radius:x1][mask] = circle_wall[ self.config.radius + 1:, : self.config.radius][mask]
                # Bottom right corner
                mask = np.isin(val_data[y2:y2+ self.config.radius, x2:x2+ self.config.radius, 0], [0, -1]) # mask for all points within the circle sector at the bottom right corner
                self.val_data[y2:y2+ self.config.radius, x2:x2+ self.config.radius][mask] = circle_wall[ self.config.radius + 1:, self.config.radius + 1:][mask]
        