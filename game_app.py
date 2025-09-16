import tkinter as tk
from game_config import GameConfig
from game_map import GameMap
from checkpoint import Checkpoint
from ball import Ball
from mqtt_client import MQTTClient
from input_control import KeyboardControl, MPU6050Control
from vibro_motor import VibroMotor
from overlay import Overlay
import time


class GameApp:
    """
    Main application class for the Tilt Maze Game.

    Handles the initialization and management of the game window, canvas, game objects,
    user input, haptic feedback, overlays, and game loop.

    Attributes:
        config (GameConfig): Game configuration loaded from file.
        window (tk.Tk): Main Tkinter window.
        canvas (tk.Canvas): Canvas for drawing game elements.
        game_map (GameMap): The game map object.
        checkpoints (list[Checkpoint]): List of checkpoint objects.
        mqtt_client (MQTTClient): MQTT client for online mode (if enabled).
        input_handler: Input handler for keyboard or MPU6050.
        is_running (bool): Whether the game loop is running.
        fell_into_holes (int): Number of times the ball fell into a hole.
        hole_count_text: Canvas text item for hole count.
        hole_cool_down (int): Cooldown timer after falling into a hole.
        start_time (float): Start time for offline mode.
        time_passed (float): Elapsed time for offline mode.
        overlay (Overlay): Overlay manager for UI overlays.
        vibro_motor (VibroMotor): Vibro motor controller.
        ball (Ball): The ball object.
        close_button (tk.Button): Button to close the application.
        pause_button (tk.Button): Button to pause/resume the game.
        code_button (tk.Button): Button to show code overlay.
        reset_button (tk.Button): Button to reset the game (offline mode).
        time_text: Canvas text item for elapsed time (offline mode).
    """

    def __init__(self, config_file_path="config.json", map_file_path="map_v1.dat"):
        """
        Initializes the GameApp, sets up the window, canvas, game objects, and UI.

        Args:
            config_file_path (str): Path to the configuration file.
            map_file_path (str): Path to the map data file.
        """
        self.config = GameConfig(config_file_path) # Load configuration from config.json

        # Initialize Tkinter window and canvas
        self.window = tk.Tk()
        self.window.title("Tilt Maze Game")
        self.window.geometry(f"{self.config.screen_width}x{self.config.screen_height}")

        self.canvas = tk.Canvas(
            self.window,
            width=self.config.screen_width,
            height=self.config.screen_height,
            bg="white",
            bd=0,
            highlightthickness=0
        )
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.config(cursor="none")

        # Initialize game map and checkpoints
        self.game_map = GameMap(map_file_path, self.config)
        self.checkpoints: list[Checkpoint] = []

        # Initialize MQTT client if online mode is enabled. If initialization fails, fall back to offline mode by changing the config.
        if self.config.online_mode:
            self.mqtt_client = MQTTClient(self.config)
            if not self.mqtt_client.client == None:
                self.mqtt_client._reset_function = self._reset_game
                self.mqtt_client._unlock_function = self._unlock_game
                self.mqtt_client.connect_and_loop()

        # Initialize input handler based on control mode. If MPU6050 fails, fall back to keyboard control.
        if self.config.control_mode == "keyboard":
            self.input_handler = KeyboardControl(self.window)
        elif self.config.control_mode == "mpu6050":
            self.input_handler = MPU6050Control()
            if not self.input_handler.status():
                self.input_handler = KeyboardControl(self.window)
        
        # Initialize game state variables - timetracking only for offline mode
        self.is_running = False
        self.fell_into_holes = 0
        self.hole_count_text = None
        self.hole_cool_down = 0
        if not self.config.online_mode:
            self.start_time = 0
            self.last_seconds = 0
            self.time_passed = 0

        self.overlay = Overlay(self.canvas) # Initialize overlay instance

        self.game_map.draw_map(self.canvas) # Draw the game map on the canvas
        self._init_checkpoints() # Initialize checkpoints from map data
        self.vibro_motor = VibroMotor(self.config, self.canvas) # Initialize vibration motor for haptic feedback
        self.ball = Ball(self.canvas, self.config, self.game_map, self.vibro_motor, self.checkpoints) # Initialize and draw the ball
        self._init_ui_elements() # Initialize UI elements like buttons and status texts
        self._bind_events() # Key bindings
        self._overlay_handler("start") # Show start overlay
        
        # If online mode, wait for mqtt initialization, else just start the game
        if self.config.online_mode:
            self.mqtt_client.game_ready()
        else:
            self._reset_game()
            self._unlock_game()

    def _init_checkpoints(self):
        """
        Initializes checkpoint objects from map data and stores them. Checkpoints are immediately drawn.
        """
        for cp_data in self.game_map.get_checkpoint_init_data():
            cp = Checkpoint(
                cp_data['x'],
                cp_data['y'],
                self.config.hole_radius,
                cp_data['name'],
                self.canvas
            )
            self.checkpoints.append(cp)
    
    def _init_ui_elements(self):
        """
        Initializes UI elements such as close, pause, code, and reset buttons,
        as well as status texts.
        """

        # Closes the whole game application (Top right)
        self.close_button = tk.Button(
            self.window, 
            text="✕", 
            command=self.close_app, 
            font=("Arial", 12, "bold"), 
            bg="red", 
            fg="white", 
            bd=0, 
            relief="flat", 
            cursor="hand2"
            )
        
        self.close_button.place(
            x=self.config.screen_width - 20, 
            y=0, 
            width=20, 
            height=20
            )
        
        # Pauses the game (Top left)
        self.pause_button = tk.Button(
            self.window, 
            text="\u23F8", 
            command=self._pause_game, 
            font=("Symbola", 10), 
            bg="green", 
            fg="white", 
            bd=0, 
            relief="flat", 
            cursor="hand2", 
            state="disabled"
            )
        
        # Opens the code input overlay (Bottom left) 
        self.code_button = tk.Button(
            self.window, 
            text="\U0001F511", 
            command = lambda: self._overlay_handler("code") if self.overlay.frame is None else self._overlay_handler("none"),
            font=("Arial", 10, "bold"), 
            bg="#471F01", 
            fg="white", 
            activebackground="#471F01", 
            activeforeground="white", 
            bd=0, 
            relief="flat", 
            cursor="hand2", 
            highlightbackground="#471F01"
            )

        # Resets the game (Bottom right) - only visible in offline mode
        self.reset_button = tk.Button(
            self.window, 
            text="\u27F3", 
            command = lambda: [self._reset_game(), self._unlock_game()], 
            font=("DejaVu Sans", 14), 
            bg="blue", 
            fg="white", 
            bd=0, 
            relief="flat", 
            cursor="hand2")
        
        if not self.config.online_mode:
            self.reset_button.place(
                x=self.config.screen_width - 20, 
                y=self.config.screen_height - 20, 
                width=20, 
                height=20)

        # Status text for holes fallen into (Top center)
        self.hole_count_text = self.canvas.create_text(
            400, 
            3, 
            text=f"Caught by {self.fell_into_holes} hole{'s' if self.fell_into_holes != 1 else ''}", 
            font=("Arial", 10, "bold"), 
            fill="white", 
            anchor="n"
            )
        
        # Status text for time passed (Bottom center) - only visible in offline mode
        if not self.config.online_mode and not self.config.control_mode == "mpu6050":
            self.time_text = self.canvas.create_text(
                400,
                460,
                text=f"Time: 00:00",
                font=("Arial", 10, "bold"),
                fill="white",
                anchor="n"
            )

    def _bind_events(self):
        """
        Binds keyboard and window events for user input and fullscreen control.
        """
        self.window.bind("<Escape>", self._end_fullscreen)
        if self.config.control_mode == "keyboard":
            self.window.bind("<KeyPress>", self.input_handler._on_key_press)
            self.window.bind("<KeyRelease>", self.input_handler._on_key_release)

    def _go_fullscreen(self):
        """
        Sets the window to fullscreen mode.
        """
        self.window.attributes('-fullscreen', True)

    def _end_fullscreen(self, event=None):
        """
        Exits fullscreen mode.

        Args:
            event: Optional Tkinter event.
        """
        self.window.attributes('-fullscreen', False)

    def _game_loop(self):
        """
        Main game loop. Updates game state, handles input, ball movement,
        collisions, haptic feedback and UI updates.
        """

        # Cancels the loop and prevents further execution if the game should not run
        if not self.is_running:
            return
        self.window.after(self.config.time_step_size, self._game_loop)

        # Update time passed text only in offline mode
        if not self.config.online_mode:
            self.time_passed = self.get_elapsed_time()
            if not self.config.control_mode == "mpu6050":
                minutes = int(self.time_passed) // 60
                seconds = int(self.time_passed) % 60
                if seconds != self.last_seconds:
                    self.last_seconds = seconds
                    self.canvas.itemconfig(self.time_text, text=f"Time: {minutes:02}:{seconds:02}")

        self.vibro_motor.update() # Update vibration motor

        # If in hole cooldown phase, decrement cooldown timer and reset ball position/velocity when done
        if self.hole_cool_down > 0:
            self.hole_cool_down -= self.config.time_step_size
            if self.hole_cool_down <= 0:
                self.hole_cool_down = 0
                self.ball.reset_position()
                self.ball.reset_velocity()
            return
        
        # Get acceleration input, update ball position and handle returned events
        acc_x, acc_y = self.input_handler.get_acceleration()
        update_result = self.ball.update_position(acc_x, acc_y)

        if "hole" in update_result:
            # Ball fell into a hole, increment counter, update text, start cooldown and vibrate
            self.fell_into_holes += 1
            self.canvas.itemconfig(
                self.hole_count_text, 
                text=f"Caught by {self.fell_into_holes} hole{'s' if self.fell_into_holes != 1 else ''}"
                )
            self.hole_cool_down = 500
            self.vibro_motor.vibrate(self.hole_cool_down)

        elif "checkpoint" in update_result:
            # Ball reached a checkpoint, if all checkpoints are reached, reset game state and show code overlay
            if all(cp.is_reached for cp in self.checkpoints):
                self.game_map.reset_start_point()
                self.ball.reset_position()
                self.ball.reset_velocity()
                self._overlay_handler("code")
                self.code_button.config(state="disabled")
                self.code_button.place_forget()
                self.pause_button.place_forget()
        
        self.ball.draw() # Redraw ball at new position

    def run(self):
        """
        Starts the Tkinter main loop and enters fullscreen if using MPU6050 which indicates a Raspberry Pi environment.
        """
        if self.config.control_mode == "mpu6050": # if on raspi
            self.window.after(100, self._go_fullscreen)
        self.window.mainloop()

    def _unlock_game(self):
        """
        Unlocks the game by enabling the overlay button and updating the overlay label.
        """
        if self.overlay.frame is None:
            raise ValueError("Overlay not initialized.")
        if self.overlay.button is None:
            raise ValueError("Overlay button not initialized.")
        self.overlay.label.config(text="Try your luck if you dare!")
        self.overlay.button.config(state="normal", bg="green", text="Go")

    def _pause_game(self):
        """
        Toggles the game between paused and running states.
        """
        self.is_running = not self.is_running
        if self.is_running:
            self.pause_button.config(bg="green", text="\u23F8")
            # If in offline mode, adjust start_time to account for paused duration
            if not self.config.online_mode:
                self.start_time = time.time() - self.time_passed
            self.window.after(self.config.time_step_size, self._game_loop)
        else:
            self.pause_button.config(bg="orange", text="\u25B6")
            self.vibro_motor.stop()

    def _reset_game(self):
        """
        Resets the game state, ball, checkpoints, UI, and overlays to initial values.
        """
        # code_button place_forget
        self.code_button.place_forget()
        # disable pause_button
        self.pause_button.config(state="disabled")
        # pause_button place_forget
        self.pause_button.place_forget()
        # pause_button set play symbol
        self.pause_button.config(text="\u25B6", bg="orange")
        # set start overlay
        self._overlay_handler("start")
        # reset ball
        self.game_map.reset_start_point()
        self.ball.reset_position()
        self.ball.reset_velocity()
        self.ball.draw()
        # reset checkpoints
        for cp in self.checkpoints:
            cp.reset()
        # reset holes number
        self.fell_into_holes = 0
        # reset holes status text
        self.canvas.itemconfig(
            self.hole_count_text, 
            text=f"Caught by {self.fell_into_holes} hole{'s' if self.fell_into_holes != 1 else ''}"
        )
        # reset time text
        if not self.config.online_mode and not self.config.control_mode == "mpu6050":
            self.canvas.itemconfig(self.time_text, text=f"Time: 00:00")
        # reset hole cooldown
        self.hole_cool_down = 0
        # reset vibro
        self.vibro_motor.stop()
        # reset is_running
        self.is_running = False

    def set_start_time(self):
        """
        Sets the start time for offline mode timing.
        """
        if not self.config.online_mode:
            self.start_time = time.time()
    
    def get_elapsed_time(self):
        """
        Returns the elapsed time since the game started (offline mode).

        Returns:
            float: Elapsed time in seconds.
        """
        return time.time() - self.start_time

    def close_app(self):
        """
        Cleans up resources and closes the application window.
        """
        if self.config.online_mode:
            self.mqtt_client.disconnect()
        self.vibro_motor.cleanup()
        self.window.destroy()

    def _overlay_handler(self, overlay_type: str = "none"):
        """
        Manages the display and state of overlays (start, code, finish) and pauses/starts the game.

        Args:
            overlay_type (str): Type of overlay to display ("none", "start", "code", "finish").
        """
        if self.overlay.frame is not None:
            self.overlay.close()

        if overlay_type == "none":
            self.pause_button.config(state="normal")
            if not self.is_running:
                self._pause_game()
        else:
            self.pause_button.config(state="disabled")
            if self.is_running:
                self._pause_game()

            if overlay_type == "start":
                self._show_start_overlay()
            elif overlay_type == "code":
                self._show_code_overlay()
            elif overlay_type == "finish":
                self._show_finish_overlay()
        
    def _show_start_overlay(self):
        """
        Displays the start overlay with instructions and a button to begin the game.
        If the mqtt initialization failed, it shows a warning message.
        """
        self.overlay.background = self.canvas.create_rectangle(
            0, 
            0, 
            self.config.screen_width, 
            self.config.screen_height, 
            fill="#424242", 
            outline="", 
            stipple="gray50"
            )
        
        self.overlay.frame = tk.Frame(
            self.window, 
            bg="#eeeeee", 
            bd=3, 
            relief="ridge"
            )
        self.overlay.frame.place(x=200, y=140, width=400, height=200)

        self.overlay.label = tk.Label(
            self.overlay.frame, 
            text="You are not yet qualified!", 
            bg="#eeeeee", 
            font=("Arial", 14, "bold")
            )
        self.overlay.label.pack(pady=25)

        self.overlay.button = tk.Button(self.overlay.frame, 
                           text="  -----  ", 
                           state="disabled", 
                           command=lambda: [
                               self._overlay_handler("none"), 
                               self.set_start_time(),
                               self.pause_button.place(x=0, y=0, width=20, height=20),
                               self.code_button.place(x=0, y=460, width=20, height=20)
                           ],
                           bg="#eeeeee", font=("Arial", 16, "bold"))
        self.overlay.button.pack(pady=10)

        if self.config.mqtt_failed:
            mqtt_fail_label = tk.Label(
                self.overlay.frame, 
                text="MQTT Connection Failed!", 
                bg="#eeeeee", 
                fg="red",
                font=("Arial", 8, "bold")
                )
            mqtt_fail_label.pack(pady=10)

    def _show_code_overlay(self):
        """
        Displays the code entry overlay for finishing the game.
        If the correct code is entered, the overlay is closed and the game ends.
        """
        # Define overlay dimensions and position
        overlay_w, overlay_h = 300, 323
        overlay_x = (self.config.screen_width - overlay_w) // 2
        overlay_y = (self.config.screen_height - overlay_h) // 2
        self.overlay.frame = tk.Frame(self.window, bg="#eeeeee", bd=3, relief="ridge")
        self.overlay.frame.place(x=overlay_x, y=overlay_y, width=overlay_w, height=overlay_h)

        # Initialize code entry field
        code_var = tk.StringVar()
        code_entry = tk.Entry(
            self.overlay.frame, 
            textvariable=code_var, 
            font=("Arial", 24), 
            justify="center", 
            state="readonly", 
            readonlybackground="#ffffff"
            )
        code_entry.pack(pady=(20, 10), padx=20, fill="x")

        def numpad_press(val):
            """
            Handles button presses on a numpad interface for code entry.
            Args:
                val (str): The value of the pressed button. Can be a digit, "Del" for delete, or "OK" to submit the code.
            Behavior:
                - If "Del" is pressed, removes the last character from the code entry.
                - If "OK" is pressed, checks if the entered code matches the required code:
                    - If correct, removes overlays, hides buttons, publishes result if in online mode, and triggers finish handler.
                    - If incorrect, clears the code entry.
                - If a digit is pressed and the code is less than 4 digits, appends the digit to the code entry.
            """

            if val == "Del":
                code_var.set(code_var.get()[:-1])
            elif val == "OK":
                if code_var.get() == self.game_map.digit_code:
                    self.overlay.frame.destroy()
                    self.code_button.place_forget()
                    self.pause_button.place_forget()
                    if self.config.online_mode:
                        self.mqtt_client.publish_result(self.fell_into_holes)
                    self._overlay_handler("finish")
                else:
                    code_var.set("")
            else:
                if len(code_var.get()) < 4:
                    code_var.set(code_var.get() + val)

        # Create numpad buttons
        btns = [["1", "2", "3"], ["4", "5", "6"], ["7", "8", "9"], ["Del", "0", "OK"]]
        btn_frame = tk.Frame(self.overlay.frame, bg="#eeeeee")
        btn_frame.pack(pady=10)
        for r, row in enumerate(btns):
            for c, val in enumerate(row):
                b = tk.Button(btn_frame, text=val, width=5, height=1, font=("Arial", 16), command=lambda v=val: numpad_press(v))
                b.grid(row=r, column=c, padx=5, pady=5)

    def _show_finish_overlay(self):
        """
        Displays the finish overlay. Adds results and statistics if in offline mode.
        """
        self.overlay.background = self.canvas.create_rectangle(
            0, 
            0, 
            self.config.screen_width, 
            self.config.screen_height, 
            fill="#424242", 
            outline="", 
            stipple="gray50"
            )
        
        self.overlay.frame = tk.Frame(
            self.window, 
            bg="#eeeeee", 
            bd=3, 
            relief="ridge"
            )
        self.overlay.frame.place(x=200, y=140, width=400, height=200)

        self.overlay.label = tk.Label(
            self.overlay.frame, 
            text="Finished!", 
            bg="#eeeeee", 
            fg="green", 
            font=("Arial", 18, "bold")
            )
        if self.config.online_mode:
            self.overlay.label.pack(pady=75)
        else:
            self.overlay.label.pack(pady=50)
            minutes = int(self.time_passed) // 60
            seconds = self.time_passed % 60
            info_label = tk.Label(
                self.overlay.frame,
                text=f"Caught by {self.fell_into_holes} hole{'s' if self.fell_into_holes != 1 else ''}\t\tTime: {minutes:02}:{seconds:04.1f}",
                bg="#eeeeee",
                fg="black",
                font=("Arial", 12)
            )
            info_label.pack(pady=10)