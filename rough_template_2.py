import json
import tkinter as tk
import time
import sys
import numpy as np
import re
from abc import ABC, abstractmethod
import threading

# Use placeholders for RPi.GPIO and MPU6050 if not on Linux to prevent errors.
if sys.platform == "linux":
    try:
        import RPi.GPIO as GPIO
        GPIO.setwarnings(False)
        GPIO.setmode(GPIO.BCM)
    except ImportError:
        print("RPi.GPIO not found. Vibration control disabled.")
        GPIO = None
    try:
        from mpu6050 import mpu6050 # type: ignore
    except ImportError:
        print("mpu6050 library not found. Running in keyboard control mode only.")
        mpu6050 = None
else: # Not Linux
    GPIO = None
    mpu6050 = None

try:
    import paho.mqtt.client as mqtt
except ImportError:
    print("paho-mqtt library not found. MQTT functionality disabled.")
    mqtt = None

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
        self.FILENAME = "map_v1.dat" # Map file
        self.RADIUS = 10 # Ball radius
        self.HOLERADIUS = 12 # Hole and Checkpoint radius
        self.VIBROGPIO = 14

        # Config.json specific
        self.control_mode = config_data.get("control", "keyboard")
        self.mpl_Debug = config_data.get("mpl_debug", False)
        
        # Checkpoint names with reordering logic
        checkpoint_names_str = config_data.get("checkpoints", "1H\t9O\t7L\t0E")
        self.digit_code = re.sub(r'\D', '', checkpoint_names_str)
        
        parsed_names = checkpoint_names_str.split("\t")
        if len(parsed_names) >= 4:
            self.parsed_checkpoint_names = [parsed_names[2], parsed_names[1], parsed_names[0], parsed_names[3]]
        else:
            self.parsed_checkpoint_names = parsed_names

        # MQTT Config
        self.BROKER = config_data.get("broker", "tanzwg.jkub.com")
        self.PORT = config_data.get("port", 1883)
        self.TOPIC = config_data.get("topic", "pr_embedded/puzzle_tilt_maze")
        self.USERNAME = config_data.get("username", "SETTLE DOWN")
        self.PASSWORD = config_data.get("password", "NEULAND")

        # Fallback for missing libraries
        if self.control_mode == "mpu6050" and mpu6050 is None:
            print("Warning: MPU6050 selected but library not found. Falling back to keyboard control.")
            self.control_mode = "keyboard"
        if self.BROKER == None or not mqtt:
            print("Warning: MQTT broker not configured or library not found. MQTT functionality disabled.")
            self.mqtt_enabled = False
        else:
            self.mqtt_enabled = True

# --- 2. GameMap Class ---
class GameMap:
    def __init__(self, map_filename: str, config: GameConfig):
        self.config = config
        self.val_data = np.zeros((self.config.HEIGHT, self.config.WIDTH, 4))
        self.start_point = np.array([0, 0], dtype=float)
        self.start_point_default = np.array([0, 0], dtype=float)
        self.map_objects_raw = []
        self.checkpoint_data = []
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
                    self.start_point_default = self.start_point.copy()
                elif parts[0] == 'r':
                    self.config.HOLERADIUS = int(parts[1])
                else:
                    self.map_objects_raw.append(parts)
        
        checkpoint_counter_for_name = 0
        for obj in self.map_objects_raw:
            if obj[0] == 'c':
                if checkpoint_counter_for_name < len(self.config.parsed_checkpoint_names):
                    name = self.config.parsed_checkpoint_names[checkpoint_counter_for_name]
                else:
                    name = f"CP{checkpoint_counter_for_name+1}"
                self.checkpoint_data.append({
                    'x': int(obj[1]),
                    'y': int(obj[2]),
                    'name': name,
                    'map_index': checkpoint_counter_for_name
                })
                checkpoint_counter_for_name += 1

    def _generate_val_data(self):
        # Your existing val_data generation logic here, using self.config.HOLERADIUS
        circle_hole = np.zeros((self.config.HOLERADIUS * 2, self.config.HOLERADIUS * 2))
        Y, X = np.ogrid[:self.config.HOLERADIUS * 2, :self.config.HOLERADIUS * 2]
        mask_hole = ((X - self.config.HOLERADIUS + 0.5) ** 2) + ((Y - self.config.HOLERADIUS + 0.5) ** 2) <= self.config.HOLERADIUS ** 2
        circle_hole[mask_hole] = -1

        idx = np.indices((self.config.HOLERADIUS * 2, self.config.HOLERADIUS * 2)) - self.config.HOLERADIUS
        idx[idx >=0] += 1
        circle_hole = np.stack((circle_hole, (idx[0]) * circle_hole / self.config.RADIUS, (idx[1]) * circle_hole / self.config.RADIUS), axis=-1)

        norm_squared = np.sum(circle_hole[:, :, 1:3] ** 2, axis=-1, keepdims=True)
        norm_squared[norm_squared == 0] = 1e-1
        circle_hole[:, :, 1:3] /= norm_squared

        inner_mask = ((X - self.config.HOLERADIUS + 0.5) ** 2) + ((Y - self.config.HOLERADIUS + 0.5) ** 2) <= (self.config.HOLERADIUS - self.config.RADIUS) ** 2
        circle_hole[inner_mask, 0] = -2

        checkpoint_counter = 0
        for obj in self.map_objects_raw:
            if obj[0] == 'w':
                x1, y1, x2, y2 = int(obj[1]), int(obj[2]), int(obj[3]), int(obj[4])
                self.val_data[y1:y2, x1:x2, 0] = 2
            elif obj[0] == 'h':
                x, y = int(obj[1]), int(obj[2])
                subdata = self.val_data[y - self.config.HOLERADIUS:y + self.config.HOLERADIUS, x - self.config.HOLERADIUS:x + self.config.HOLERADIUS]
                template_slice = circle_hole[:subdata.shape[0], :subdata.shape[1]]
                mask = mask_hole[:subdata.shape[0], :subdata.shape[1]]
                subdata[mask, :3] = template_slice[mask, :3]
                self.val_data[y - self.config.HOLERADIUS:y + self.config.HOLERADIUS, x - self.config.HOLERADIUS:x + self.config.HOLERADIUS] = subdata
            elif obj[0] == 'c':
                x, y = int(obj[1]), int(obj[2])
                Y, X = np.ogrid[:self.config.HEIGHT, :self.config.WIDTH]
                mask_checkpoint = (((X - x) ** 2)) + ((Y - y) ** 2) <= self.config.HOLERADIUS ** 2
                self.val_data[mask_checkpoint, 0] = -3
                self.val_data[mask_checkpoint, 3] = checkpoint_counter
                checkpoint_counter += 1

        for obj in self.map_objects_raw:
            if obj[0] == 'w':
                x1, y1, x2, y2 = int(obj[1]), int(obj[2]), int(obj[3]), int(obj[4])
                Y, X = np.ogrid[:self.config.HEIGHT, :self.config.WIDTH]

                subdata = self.val_data[y1-self.config.RADIUS:y1, x1:x2]
                subdata[subdata[:,:, 0] <= 0, 0] = 1
                subdata[subdata[:,:, 0] != 2] += np.array([0, -1, 0, 0])
                self.val_data[y1-self.config.RADIUS:y1, x1:x2] = subdata

                subdata = self.val_data[y2:y2+self.config.RADIUS, x1:x2]
                subdata[subdata[:,:, 0] <= 0, 0] = 1
                subdata[subdata[:,:, 0] != 2] += np.array([0, 1, 0, 0])
                self.val_data[y2:y2+self.config.RADIUS, x1:x2] = subdata

                subdata = self.val_data[y1:y2, x1-self.config.RADIUS:x1]
                subdata[subdata[:,:, 0] <= 0, 0] = 1
                subdata[subdata[:,:, 0] != 2] += np.array([0, 0, -1, 0])
                self.val_data[y1:y2, x1-self.config.RADIUS:x1] = subdata

                subdata = self.val_data[y1:y2, x2:x2+self.config.RADIUS]
                subdata[subdata[:,:, 0] <= 0, 0] = 1
                subdata[subdata[:,:, 0] != 2] += np.array([0, 0, 1, 0])
                self.val_data[y1:y2, x2:x2+self.config.RADIUS] = subdata

        circle_wall = np.zeros((2*self.config.RADIUS + 1, 2*self.config.RADIUS + 1))
        Y, X = np.ogrid[:2*self.config.RADIUS + 1, :2*self.config.RADIUS + 1]
        mask_corner = ((X - self.config.RADIUS) ** 2) + ((Y - self.config.RADIUS) ** 2) <= self.config.RADIUS ** 2
        circle_wall[mask_corner] = 1

        idx = np.indices((2*self.config.RADIUS + 1, 2*self.config.RADIUS + 1))
        circle_wall = np.stack((circle_wall, (idx[0] - self.config.RADIUS) * circle_wall / self.config.RADIUS, (idx[1]- self.config.RADIUS) * circle_wall / self.config.RADIUS, np.zeros((2*self.config.RADIUS + 1, 2*self.config.RADIUS + 1))), axis=-1)

        for obj in self.map_objects_raw:
            if obj[0] == 'w':
                x1, y1, x2, y2 = int(obj[1]), int(obj[2]), int(obj[3]), int(obj[4])
                mask = np.isin(self.val_data[y1-self.config.RADIUS:y1, x1-self.config.RADIUS:x1, 0], [0, -1])
                self.val_data[y1-self.config.RADIUS:y1, x1-self.config.RADIUS:x1][mask] = circle_wall[:self.config.RADIUS, :self.config.RADIUS][mask]
                mask = np.isin(self.val_data[y1-self.config.RADIUS:y1, x2:x2+self.config.RADIUS, 0], [0, -1])
                self.val_data[y1-self.config.RADIUS:y1, x2:x2+self.config.RADIUS][mask] = circle_wall[:self.config.RADIUS, self.config.RADIUS + 1:][mask]
                mask = np.isin(self.val_data[y2:y2+self.config.RADIUS, x1-self.config.RADIUS:x1, 0], [0, -1])
                self.val_data[y2:y2+self.config.RADIUS, x1-self.config.RADIUS:x1][mask] = circle_wall[self.config.RADIUS + 1:, :self.config.RADIUS][mask]
                mask = np.isin(self.val_data[y2:y2+self.config.RADIUS, x2:x2+self.config.RADIUS, 0], [0, -1])
                self.val_data[y2:y2+self.config.RADIUS, x2:x2+self.config.RADIUS][mask] = circle_wall[self.config.RADIUS + 1:, self.config.RADIUS + 1:][mask]
                

    def get_collision_info(self, x: int, y: int) -> np.ndarray:
        if not (0 <= y < self.config.HEIGHT and 0 <= x < self.config.WIDTH):
            return np.array([-2, 0, 0, 0])
        return self.val_data[y, x, :]

    def get_start_point(self) -> np.ndarray:
        return self.start_point.copy()
        
    def get_default_start_point(self) -> np.ndarray:
        return self.start_point_default.copy()

    def get_raw_map_objects(self) -> list:
        return self.map_objects_raw
    
    def get_checkpoint_init_data(self) -> list:
        return self.checkpoint_data

# --- 3. Ball Class ---
class Ball:
    def __init__(self, start_pos: np.ndarray, canvas: tk.Canvas, config: GameConfig):
        self.config = config
        self.pos = start_pos.copy()
        self.vel = np.array([0, 0], dtype=float)
        self.radius = self.config.RADIUS
        self.canvas = canvas

        self.canvas_id = self.canvas.create_oval(
            int(self.pos[0]) - self.radius + 1, int(self.pos[1]) - self.radius + 1,
            int(self.pos[0]) + self.radius, int(self.pos[1]) + self.radius,
            fill="blue", outline="blue"
        )
        self.current_color = "blue"

    def update_position(self, accel_x: float, accel_y: float, game_map: GameMap, checkpoints: list, vibro_motor):
        
        last_pos = self.pos.copy()
        normal_vectors = set()

        self.vel[0] += self.config.ACC_SCALE * accel_x * self.config.DT / 1000
        self.vel[1] += self.config.ACC_SCALE * accel_y * self.config.DT / 1000
        
        Dpos = np.array(self.vel) * self.config.DT / 1000
        dist = np.linalg.norm(Dpos)
        steps = int(dist / self.config.DP) if dist > self.config.DP else 1
        
        dstep = Dpos / steps
        counter = 0
        security = 0

        while counter < steps:
            security += 1
            if security > 1000:
                print("Security limit reached, breaking loop")
                break
            temp_pos = self.pos + dstep
            
            clamped_x = int(np.clip(temp_pos[0], 0, self.config.WIDTH - 1))
            clamped_y = int(np.clip(temp_pos[1], 0, self.config.HEIGHT - 1))
            val_info = game_map.get_collision_info(clamped_x, clamped_y)
            px_type = val_info[0]

            if px_type > 0:
                vec_norm = val_info[1:3][::-1]
                pos_dot_product = np.dot(vec_norm, Dpos)

                if (pos_dot_product < 0):
                    vec_proj_pos = pos_dot_product / np.dot(vec_norm, vec_norm) * vec_norm
                    vec_proj_vel = np.dot(vec_norm, self.vel) / np.dot(vec_norm, vec_norm) * vec_norm
                    
                    Dpos = - 2 * vec_proj_pos + Dpos
                    self.vel = - 2 * vec_proj_vel + self.vel

                    Dpos += vec_proj_pos * self.config.DAMPING
                    self.vel += vec_proj_vel * self.config.DAMPING

                    dist = np.linalg.norm(Dpos)
                    steps = int(dist / self.config.DP) if dist > self.config.DP else 1
                    dstep = Dpos / steps
                    counter = 0
                    continue
                else:
                    shift = vec_norm / np.linalg.norm(vec_norm) * self.config.DP
                    self.pos += shift
                    Dpos -= shift
                    counter += 1
                    continue
            elif px_type == -2:
                self.vel = np.array([0, 0], dtype=float)
                return "hole_fall"
            elif px_type == -3:
                c_number = int(val_info[3])
                if c_number < len(checkpoints) and not checkpoints[c_number].is_reached:
                    checkpoints[c_number].mark_reached()
                    start_point = np.array(checkpoints[c_number].get_center_coords(), dtype=float)
                    game_map.start_point = start_point
                    return "checkpoint_reached"
            
            self.pos += dstep
            Dpos -= dstep
            counter += 1

        if (game_map.get_collision_info(int(self.pos[0]), int(self.pos[1]))[0] == -1):
            vec_norm_final = game_map.get_collision_info(int(self.pos[0]), int(self.pos[1]))[1:3][::-1]
            vec_tang = np.array([vec_norm_final[1], -vec_norm_final[0]])
            vec_proj_vel_tang = np.dot(vec_tang, self.vel) / np.dot(vec_tang, vec_tang) * vec_tang
            self.vel += vec_norm_final * 10
            self.vel -= vec_proj_vel_tang * 0.3
            
        pos_difference = self.pos - last_pos
        for vector in normal_vectors:
            vec_norm = np.array(vector, dtype=float)
            vec_proj_difference = np.dot(vec_norm, pos_difference) / np.dot(vec_norm, vec_norm) * vec_norm
            if np.linalg.norm(vec_proj_difference) > 0.1:
                vibro_motor.vibrate()
        
        return "continue"

    def draw(self):
        self.canvas.coords(
            self.canvas_id,
            int(self.pos[0]) - self.radius + 1, int(self.pos[1]) - self.radius + 1,
            int(self.pos[0]) + self.radius, int(self.pos[1]) + self.radius
        )

    def set_color(self, color: str):
        if self.current_color != color:
            self.canvas.itemconfig(self.canvas_id, fill=color, outline=color)
            self.current_color = color

    def reset(self, start_pos: np.ndarray):
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
        self.canvas = canvas

        self.canvas_oval_id = self.canvas.create_oval(
            x - radius, y - radius, x + radius, y + radius,
            fill="orange", outline="lightgray"
        )
        self.canvas_text_id = self.canvas.create_text(
            x, y, text="", fill="black", font=("Arial", 10, "bold")
        )

    def mark_reached(self):
        if not self.is_reached:
            self.is_reached = True
            self.canvas.itemconfig(self.canvas_oval_id, fill="#0CFF0B")
            self.canvas.itemconfig(self.canvas_text_id, text=self.name)

    def reset(self):
        self.is_reached = False
        self.canvas.itemconfig(self.canvas_oval_id, fill="orange")
        self.canvas.itemconfig(self.canvas_text_id, text="")

    def get_center_coords(self) -> tuple[int, int]:
        return (int(self.position[0]), int(self.position[1]))

# --- 5. InputHandler Classes ---
class InputHandler(ABC):
    @abstractmethod
    def get_acceleration(self) -> tuple[float, float]:
        pass

class KeyboardInputHandler(InputHandler):
    def __init__(self, window: tk.Tk):
        self._pressed_keys = set()
        window.bind("<KeyPress>", self._on_key_press)
        window.bind("<KeyRelease>", self._on_key_release)

    def _on_key_press(self, event: tk.Event):
        self._pressed_keys.add(event.keysym)

    def _on_key_release(self, event: tk.Event):
        self._pressed_keys.discard(event.keysym)

    def get_acceleration(self) -> tuple[float, float]:
        ax, ay = 0, 0
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
            raise RuntimeError("MPU6050 library not available.")
        self._sensor = mpu6050(0x68)

    def get_acceleration(self) -> tuple[float, float]:
        current_acc = self._sensor.get_accel_data()
        return current_acc['y'], -current_acc['x']

# --- 6. MQTTClient Class ---
class MQTTClient:
    def __init__(self, config: GameConfig, app):
        if not config.mqtt_enabled:
            self.client = None
            return

        self.config = config
        self.app = app
        self.client = mqtt.Client()
        self.client.username_pw_set(self.config.USERNAME, self.config.PASSWORD)
        self.client.on_connect = self._on_connect
        self.client.on_message = self._on_message
        
    def _on_connect(self, client, userdata, flags, rc):
        print(f"Connected to MQTT broker with result code {rc}")
        if rc == 0:
            client.publish(self.config.TOPIC + "/general", "Connected")
            client.subscribe(self.config.TOPIC + "/general")

    def _on_message(self, client, userdata, msg):
        print(f"Received message on topic {msg.topic}: {msg.payload.decode()}")
        if (msg.topic == self.config.TOPIC + "/general") and (msg.payload.decode() == "initialize"):
            client.publish(self.config.TOPIC + "/general", "initialize_ack")
            self.app.start_game()

    def connect_and_loop(self):
        if self.client:
            print("Trying to connect to MQTT broker...")
            self.client.connect(self.config.BROKER, self.config.PORT, 30)
            self.client.loop_start()

    def publish_points(self, fell_into_holes):
        if self.client:
            points = (5 * fell_into_holes) if fell_into_holes < 10 else 45
            self.client.publish(self.config.TOPIC + "/points", points)

    def publish_general(self, message):
        if self.client:
            self.client.publish(self.config.TOPIC + "/general", message)

    def disconnect(self):
        if self.client:
            self.client.loop_stop()
            self.client.disconnect()

# --- 7. VibroMotor Class ---
class VibroMotor:
    def __init__(self, config, canvas=None):
        self.config = config
        self.canvas = canvas
        self.vibrate_cool_down = 0
        self.vibro_ind = None

        if GPIO:
            GPIO.setup(self.config.VIBROGPIO, GPIO.OUT)
            self.high = lambda pin: GPIO.output(pin, GPIO.HIGH)
            self.low = lambda pin: GPIO.output(pin, GPIO.LOW)
        else:
            self.high = lambda pin: None
            self.low = lambda pin: None

        if not GPIO and self.canvas:
            self.vibro_ind = self.canvas.create_oval(30, 5, 40, 15, fill="gray", outline="black")

    def vibrate(self):
        if self.vibrate_cool_down <= 0:
            self.vibrate_cool_down = 100
            self.high(self.config.VIBROGPIO)
            if self.vibro_ind:
                self.canvas.itemconfig(self.vibro_ind, fill="red")

    def update(self):
        if self.vibrate_cool_down > 0:
            self.vibrate_cool_down -= self.config.DT
            if self.vibrate_cool_down <= 0:
                self.vibrate_cool_down = 0
                self.low(self.config.VIBROGPIO)
                if self.vibro_ind:
                    self.canvas.itemconfig(self.vibro_ind, fill="gray")

    def cleanup(self):
        if GPIO:
            self.low(self.config.VIBROGPIO)
            GPIO.cleanup()

# --- 8. GameApp Class (The Orchestrator) ---
class GameApp:
    def __init__(self, config_file_path="config.json", map_file_path="map_v1.dat"):
        self.config = GameConfig(config_file_path)

        self.window = tk.Tk()
        self.window.title("Tilt Maze Game")
        self.window.geometry(f"{self.config.WIDTH}x{self.config.HEIGHT}")
        if self.config.control_mode == "mpu6050":
            self.window.attributes('-fullscreen', True)
        self.window.focus_force()

        self.canvas = tk.Canvas(self.window, width=self.config.WIDTH, height=self.config.HEIGHT, bg="white", bd=0, highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.config(cursor="none")

        self.game_map = GameMap(map_file_path, self.config)
        self.ball = Ball(self.game_map.get_start_point(), self.canvas, self.config)
        self.checkpoints = []
        self._init_checkpoints()

        self.mqtt_client = MQTTClient(self.config, self)

        if self.config.control_mode == "keyboard":
            self.input_handler = KeyboardInputHandler(self.window)
        elif self.config.control_mode == "mpu6050":
            try:
                self.input_handler = MPU6050InputHandler()
            except RuntimeError as e:
                print(e)
                self.input_handler = KeyboardInputHandler(self.window)

        self.vibro_motor = VibroMotor(self.config, self.canvas)

        self.is_paused = False
        self.is_finished = False
        self.is_started = False
        self.code_overlay_flag = False
        self.fell_into_holes = 0
        self.hole_cool_down = 0
        self.hole_status_text = None
        self.code_overlay = None

        self._init_ui_elements()
        self._show_start_overlay()
        self._bind_events()

    def _init_checkpoints(self):
        for cp_data in self.game_map.get_checkpoint_init_data():
            cp = Checkpoint(
                cp_data['x'], cp_data['y'],
                self.config.HOLERADIUS,
                cp_data['name'],
                self.canvas
            )
            self.checkpoints.append(cp)

    def _init_ui_elements(self):
        for obj in self.game_map.get_raw_map_objects():
            if obj[0] == 'w':
                self.canvas.create_rectangle(int(obj[1]), int(obj[2]), int(obj[3]), int(obj[4]), fill="#471F01", outline="#471F01")
            elif obj[0] == 'h':
                self.canvas.create_oval(int(obj[1]) - self.config.HOLERADIUS, int(obj[2]) - self.config.HOLERADIUS,
                                        int(obj[1]) + self.config.HOLERADIUS, int(obj[2]) + self.config.HOLERADIUS,
                                        fill="black", outline="lightgray")
        
        self.close_button = tk.Button(self.window, text="✕", command=self.close_app, font=("Arial", 12, "bold"), bg="red", fg="white", bd=0, relief="flat", cursor="hand2")
        self.close_button.place(x=self.config.WIDTH - 20, y=0, width=20, height=20)
        
        self.pause_button = tk.Button(self.window, text="\u23F8", command=self.pause_game, font=("Symbola", 10), bg="green", fg="white", bd=0, relief="flat", cursor="hand2", state="disabled")
        self.pause_button.place(x=0, y=0, width=20, height=20)
        
        self.code_button = tk.Button(self.window, text="\U0001F511", command=self.show_code_overlay, font=("Arial", 10, "bold"), bg="#471F01", fg="white", activebackground="#471F01", activeforeground="white", bd=0, relief="flat", cursor="hand2", highlightbackground="#471F01")
        
        self.hole_status_text = self.canvas.create_text(400, 3, text=f"Caught by {self.fell_into_holes} hole{'s' if self.fell_into_holes != 1 else ''}", font=("Arial", 10, "bold"), fill="white", anchor="n")

    def _bind_events(self):
        self.window.bind("<Escape>", self._end_fullscreen)
        if self.config.control_mode == "keyboard":
            self.window.bind("<KeyPress>", self.input_handler._on_key_press)
            self.window.bind("<KeyRelease>", self.input_handler._on_key_release)

    def _go_fullscreen(self):
        self.window.attributes('-fullscreen', True)

    def _end_fullscreen(self, event=None):
        self.window.attributes('-fullscreen', False)

    def _game_loop(self):
        if self.is_paused:
            return
        
        if self.hole_cool_down > 0:
            self.hole_cool_down -= self.config.DT
            if self.hole_cool_down <= 0:
                self.hole_cool_down = 0
                self.ball.pos = self.game_map.get_start_point()
                self.ball.vel = np.array([0, 0], dtype=float)
            self.window.after(self.config.DT, self._game_loop)
            return

        self.vibro_motor.update()

        if self.is_started and not self.is_finished:
            ax, ay = self.input_handler.get_acceleration()
            update_result = self.ball.update_position(ax, ay, self.game_map, self.checkpoints, self.vibro_motor)

            if update_result == "hole_fall":
                self.fell_into_holes += 1
                self.canvas.itemconfig(self.hole_status_text, text=f"Caught by {self.fell_into_holes} hole{'s' if self.fell_into_holes != 1 else ''}")
                self.hole_cool_down = 500
            elif update_result == "checkpoint_reached":
                if all(cp.is_reached for cp in self.checkpoints):
                    self.is_finished = True
                    self.ball.set_color("gold")
                    self.code_button.place(x=0, y=self.config.HEIGHT - 20, width=20, height=20)
                    self.pause_game()
                    self.pause_button.config(state="disabled")

        self.ball.draw()
        self.window.after(self.config.DT, self._game_loop)

    def run(self):
        self.mqtt_client.connect_and_loop()
        if self.config.control_mode == "mpu6050":
            self.window.after(100, self._go_fullscreen)
        self.window.mainloop()

    def close_app(self):
        self.mqtt_client.disconnect()
        self.vibro_motor.cleanup()
        self.window.destroy()

    def reset_game(self):
        self.ball.reset(self.game_map.get_default_start_point())
        self.game_map.start_point = self.game_map.get_default_start_point()
        for cp in self.checkpoints:
            cp.reset()
        self.fell_into_holes = 0
        self.is_finished = False
        self.is_paused = False
        self.canvas.itemconfig(self.hole_status_text, text=f"Caught by {self.fell_into_holes} hole{'s' if self.fell_into_holes != 1 else ''}")
        self.code_button.place_forget()
        self.pause_button.config(text="\u23F8", bg="green", state="normal")
        self.mqtt_client.publish_general("Game reset")

    def pause_game(self):
        self.is_paused = not self.is_paused
        if self.is_paused:
            self.vibro_motor.low(self.vibro_motor.config.VIBROGPIO)
            self.pause_button.config(bg="orange", text="\u25B6")
        else:
            self.pause_button.config(bg="green", text="\u23F8")
            self.window.after(self.config.DT, self._game_loop)

    def start_game(self):
        self.is_started = True
        self.reset_game()
        self.overlay_frame.destroy()
        self.canvas.delete(self.start_overlay_rect)
        self.canvas.delete(self.start_overlay_semi_trans)
        self.pause_button.config(state="normal")
        self.window.after(self.config.DT, self._game_loop)

    def _show_start_overlay(self):
        self.start_overlay_semi_trans = self.canvas.create_rectangle(0, 0, self.config.WIDTH, self.config.HEIGHT, fill="#424242", outline="", stipple="gray50")
        self.start_overlay_rect = self.canvas.create_rectangle(200, 140, 600, 340, fill="#eeeeee", outline="black")
        self.overlay_frame = tk.Frame(self.window, bg="#eeeeee", bd=3, relief="ridge")
        self.overlay_frame.place(x=200, y=140, width=400, height=200)
        self.overlay_label = tk.Label(self.overlay_frame, text="You are not yet qualified!", bg="#eeeeee", font=("Arial", 14, "bold"))
        self.overlay_label.pack(pady=30)
        self.overlay_button = tk.Button(self.overlay_frame, text="   -----   ", state="disabled", command=self.start_game, bg="#eeeeee", font=("Arial", 16, "bold"))
        self.overlay_button.pack(pady=15)

    def show_code_overlay(self):
        if self.code_overlay_flag:
            self.code_overlay_flag = False
            self.code_overlay.destroy()
            self.pause_game()
            self.pause_button.config(state="normal")
            return
        
        self.code_overlay_flag = True
        if not self.is_paused:
            self.pause_game()
        self.pause_button.config(state="disabled")

        overlay_w, overlay_h = 300, 323
        overlay_x = (self.config.WIDTH - overlay_w) // 2
        overlay_y = (self.config.HEIGHT - overlay_h) // 2
        self.code_overlay = tk.Frame(self.window, bg="#eeeeee", bd=3, relief="ridge")
        self.code_overlay.place(x=overlay_x, y=overlay_y, width=overlay_w, height=overlay_h)

        code_var = tk.StringVar()
        code_entry = tk.Entry(self.code_overlay, textvariable=code_var, font=("Arial", 24), justify="center", state="readonly", readonlybackground="#ffffff")
        code_entry.pack(pady=(20, 10), padx=20, fill="x")

        def numpad_press(val):
            if val == "Del":
                code_var.set(code_var.get()[:-1])
            elif val == "OK":
                if code_var.get() == self.config.digit_code:
                    self.code_overlay.destroy()
                    self.code_button.place_forget()
                    self.mqtt_client.publish_points(self.fell_into_holes)
                    self.mqtt_client.publish_general("finished")
                    self._show_finished_overlay()
                else:
                    code_var.set("")
            else:
                if len(code_var.get()) < 4:
                    code_var.set(code_var.get() + val)

        btns = [["1", "2", "3"], ["4", "5", "6"], ["7", "8", "9"], ["Del", "0", "OK"]]
        btn_frame = tk.Frame(self.code_overlay, bg="#eeeeee")
        btn_frame.pack(pady=10)
        for r, row in enumerate(btns):
            for c, val in enumerate(row):
                b = tk.Button(btn_frame, text=val, width=5, height=1, font=("Arial", 16), command=lambda v=val: numpad_press(v))
                b.grid(row=r, column=c, padx=5, pady=5)

    def _show_finished_overlay(self):
        semi_transparent_finished_overlay = self.canvas.create_rectangle(0, 0, self.config.WIDTH, self.config.HEIGHT, fill="#424242", outline="", stipple="gray50")
        finished_overlay = self.canvas.create_rectangle(200, 140, 600, 340, fill="#eeeeee", outline="black")
        finished_overlay_frame = tk.Frame(self.window, bg="#eeeeee", bd=3, relief="ridge")
        finished_overlay_frame.place(x=200, y=140, width=400, height=200)
        finished_overlay_label = tk.Label(finished_overlay_frame, text="Finished!", bg="#eeeeee", fg="green", font=("Arial", 18, "bold"))
        finished_overlay_label.pack(pady=75)

# --- Main execution block ---
if __name__ == "__main__":
    app = GameApp()
    app.run()

    if app.config.mpl_Debug:
        try:
            import matplotlib.pyplot as plt
            plt.imshow(-app.game_map.val_data[:,:,0], cmap='gray', vmin=-3, vmax=4)
            plt.title("Game Map val_data (Type Layer)")
            plt.show()
        except ImportError:
            print("Matplotlib not installed. Cannot display debug plots.")