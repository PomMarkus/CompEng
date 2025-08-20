import json

class GameConfig:
    def __init__(self, config_file_name="config.json"):
        with open(config_file_name, "r") as file:
            config_data = json.load(file)

        self.control_mode = config_data.get("control", "keyboard")
        self.mpl_debug = config_data.get("mpl_debug", False)
        self.checkpoint_namestring = config_data.get("checkpoints", "1H\t9O\t7L\t0E")
        self.time_step_size = config_data.get("time_step_size", 20)
        self.position_step_size = config_data.get("position_step_size", 0.1)
        self.acceleration_factor = config_data.get("acceleration_factor", 100)
        self.damping_factor = config_data.get("damping_factor", 0.8)
        self.ball_radius = config_data.get("ball_radius", 10)
        self.hole_radius = config_data.get("hole_radius", 12)
        self.map_file_name = config_data.get("map_file_name", "map_v1.txt")
        self.screen_width = config_data.get("screen_width", 800)
        self.screen_height = config_data.get("screen_height", 480)
        self.vibration_gpio = config_data.get("vibration_gpio", 14)        

        # MQTT Config
        self.BROKER = config_data.get("broker", "tanzwg.jkub.com")
        self.PORT = config_data.get("port", 1883)
        self.TOPIC = config_data.get("topic", "pr_embedded/puzzle_tilt_maze")
        self.USERNAME = config_data.get("username", "SETTLE DOWN")
        self.PASSWORD = config_data.get("password", "NEULAND")


        if self.control_mode not in ["keyboard", "mpu6050"]:
            raise ValueError(f"Invalid control mode: {self.control_mode}. Choose 'keyboard' or 'mpu6050'.")

        # if self.control_mode == "mpu6050" and mpu6050 is None:
        #     raise ImportError("mpu6050 module is required for 'mpu6050' control mode.")