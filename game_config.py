import json

class GameConfig:
    """
    Loads and stores configuration settings for the game from a JSON file.

    Attributes:
        control_mode (str): Input control mode, either 'keyboard' or 'mpu6050'.
        online_mode (bool): Whether the game runs in online mode using MQTT.
        mpl_debug (bool): Enables debug mode for matplotlib.
        checkpoint_namestring (str): String representing checkpoint names.
        time_step_size (int): Time step size for game updates.
        position_step_size (float): Step size for position updates.
        acceleration_factor (int): Factor for acceleration calculations.
        damping_factor (float): Damping factor for movement.
        ball_radius (int): Radius of the ball.
        hole_radius (int): Radius of the hole.
        map_file_name (str): Name of the map file.
        screen_width (int): Width of the game screen.
        screen_height (int): Height of the game screen.
        vibration_gpio (int): GPIO pin for vibration motor.
        BROKER (str): MQTT broker address.
        PORT (int): MQTT broker port.
        TOPIC (str): MQTT topic.
        USERNAME (str): MQTT username.
        PASSWORD (str): MQTT password.
        mqtt_failed (bool): Indicates if MQTT connection failed.
    """

    def __init__(self, config_file_name="config.json"):
        """
        Initializes the GameConfig object by loading settings from a JSON configuration file.

        Args:
            config_file_name (str): Path to the configuration JSON file. Defaults to 'config.json'.

        Raises:
            ValueError: If the control mode specified in the config file is invalid.
        """
        with open(config_file_name, "r") as file:
            config_data = json.load(file)

        self.control_mode = config_data.get("control", "keyboard")
        self.online_mode = config_data.get("online_mode", True)
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
        self.USERNAME = config_data.get("username", "aMAZEing")
        self.PASSWORD = config_data.get("password", "puzzle")

        self.mqtt_failed = False

        if self.control_mode not in ["keyboard", "mpu6050"]:
            raise ValueError(f"Invalid control mode: {self.control_mode}. Choose 'keyboard' or 'mpu6050'.")