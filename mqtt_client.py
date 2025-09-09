from game_config import GameConfig
try:
    import paho.mqtt.client as mqtt
except ImportError:
    print("paho-mqtt library not found. MQTT functionality disabled.")
    mqtt = None


class MQTTClient:
    """
    Handles MQTT communication for the game, including connecting to the broker,
    subscribing to topics, publishing messages, and handling received messages.
    """

    def __init__(self, config: GameConfig):
        """
        Initialize the MQTTClient with the username and password from the configuration.

        Args:
            config (GameConfig): The game configuration containing MQTT settings.
        """
        if mqtt is None:
            self.client = None
            return
        
        self.config = config
        self.client = mqtt.Client()
        self.client.username_pw_set(self.config.USERNAME, self.config.PASSWORD)
        self.client.on_connect = self._on_connect
        self.client.on_message = self._on_message

        self.connected = False
        self.game_is_ready = False
        self.is_initialized = False

        self._reset_function = None
        self._unlock_function = None

    def _on_connect(self, client, userdata, flags, rc):
        """
        Callback for when the client receives a CONNACK response from the server.

        Args:
            client: The MQTT client instance.
            userdata: The private user data.
            flags: Response flags sent by the broker.
            rc: The connection result.
        """
        if rc == 0:
            print("Connected to MQTT broker")
            client.publish(self.config.TOPIC+"/general", "Connected")
            client.subscribe(self.config.TOPIC+"/general")
            self.connected = True
        else:
            print(f"Failed to connect to MQTT broker, return code: {rc}")
            self.config.online_mode = False
            self.config.mqtt_failed = True

    def _on_message(self, client, userdata, msg):
        """
        Callback for when a PUBLISH message is received from the server.

        Args:
            client: The MQTT client instance.
            userdata: The private user data.
            msg: The received message.
        """
        print(f"Message received on topic {msg.topic}: {msg.payload.decode()}")
        if (msg.topic == self.config.TOPIC+"/general" and 
            msg.payload.decode() == "initialize"):
            client.publish(self.config.TOPIC + "/general", "initialize_ack")
            if self.game_is_ready:
                if self._reset_function:
                    self._reset_function()
                if self._unlock_function:
                    self._unlock_function()
            else:
                self.is_initialized = True

    def game_ready(self):
        """
        Mark the game as ready and perform reset/unlock if initialization was requested.
        """
        if self.game_is_ready:
            return
        self.game_is_ready = True
        if self.is_initialized:
            if self._reset_function:
                self._reset_function()
            if self._unlock_function:
                self._unlock_function()

    def connect_and_loop(self):
        """
        Connect to the MQTT broker and start the network loop.
        """
        if self.client:
            self.client.connect(self.config.BROKER, self.config.PORT, 30)
            self.client.loop_start()

    def publish_result(self, fell_into_holes: int):
        """
        Publish the game result (points) and notify that the game is finished.

        Args:
            fell_into_holes (int): The number of times the player fell into holes.
        """
        if self.client:
            points = (5 * fell_into_holes) if fell_into_holes < 10 else 45
            self.client.publish(self.config.TOPIC + "/points", points)
            self.client.publish(self.config.TOPIC + "/general", "finished")

    def disconnect(self):
        """
        Disconnect from the MQTT broker and stop the network loop.
        """
        if self.client:
            self.client.loop_stop()
            self.client.disconnect()
            self.connected = False