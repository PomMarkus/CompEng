import tkinter as tk
from game_config import GameConfig
from game_map import GameMap
from checkpoint import Checkpoint
from ball import Ball
from mqtt_client import MQTTClient
from input_control import KeyboardControl, MPU6050Control
from vibro_motor import VibroMotor
from overlay import Overlay

# Check if mpu and mqtt work:  mqtt_enabled
# decide in config if while game should be startet or just maze
# if mqtt is suppressed: show reset button




class GameApp:
    def __init__(self, config_file_path="config.json", map_file_path="map_v1.dat"):
        self.config = GameConfig(config_file_path)
        
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

        self.game_map = GameMap(map_file_path, self.config)
        self.checkpoints: list[Checkpoint] = []

        self.mqtt_client = MQTTClient(self.config) # test if self.mqtt_client.client == None
        self.mqtt_client._reset_function = self._reset_game
        self.mqtt_client._unlock_function = self._unlock_game

        if self.config.control_mode == "keyboard":
            self.input_handler = KeyboardControl(self.window)
        elif self.config.control_mode == "mpu6050":
            try:
                self.input_handler = MPU6050Control()
            except RuntimeError as e:
                print("Failed to initialize MPU6050:", e)
                self.input_handler = KeyboardControl(self.window)
        
        self.is_running = False
        # self.is_finished = False
        self.fell_into_holes = 0
        self.hole_status_text = None
        self.hole_cool_down = 0

        self.overlay = Overlay(self.canvas)

        self.game_map.draw_map(self.canvas)
        self._init_checkpoints()
        self.vibro_motor = VibroMotor(self.config, self.canvas)
        self.ball = Ball(self.canvas, self.config, self.game_map, self.vibro_motor, self.checkpoints)
        self._init_ui_elements()
        self._bind_events()
        self._overlay_handler("start")
        self.mqtt_client.connect_and_loop()

    def _init_checkpoints(self):
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
        
        # self.pause_button.place(
        #     x=0, 
        #     y=0, 
        #     width=20, 
        #     height=20
        #     )
        
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
        
        # self.code_button.place(
        #     x=0,
        #     y=self.config.screen_height - 20,
        #     width=20,
        #     height=20
        #     )

        
        self.hole_status_text = self.canvas.create_text(
            400, 
            3, 
            text=f"Caught by {self.fell_into_holes} hole{'s' if self.fell_into_holes != 1 else ''}", 
            font=("Arial", 10, "bold"), 
            fill="white", 
            anchor="n"
            )
        
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
        if not self.is_running:
            return

        self.vibro_motor.update()

        if self.hole_cool_down > 0:
            self.hole_cool_down -= self.config.time_step_size
            if self.hole_cool_down <= 0:
                self.hole_cool_down = 0
                self.ball.reset_position()
                self.ball.reset_velocity()
            self.window.after(self.config.time_step_size, self._game_loop)
            return
        
        acc_x, acc_y = self.input_handler.get_acceleration()
        update_result = self.ball.update_position(acc_x, acc_y)

        if "hole" in update_result:
            self.fell_into_holes += 1
            self.canvas.itemconfig(
                self.hole_status_text, 
                text=f"Caught by {self.fell_into_holes} hole{'s' if self.fell_into_holes != 1 else ''}"
                )
            self.hole_cool_down = 500
            self.vibro_motor.vibrate(self.hole_cool_down)

        elif "checkpoint" in update_result:
            if all(cp.is_reached for cp in self.checkpoints):
                # self.is_finished = True
                self._overlay_handler("code")
                self.code_button.config(state="disabled")
                self.code_button.place_forget()
                self.pause_button.place_forget()
        
        self.ball.draw()
        self.window.after(self.config.time_step_size, self._game_loop)

    def run(self):
        if self.config.control_mode == "mpu6050": # if on raspi
            self.window.after(100, self._go_fullscreen)
        self.window.mainloop()

    def _unlock_game(self):
        if self.overlay.frame is None:
            raise ValueError("Overlay not initialized.")
        if self.overlay.button is None:
            raise ValueError("Overlay button not initialized.")
        self.overlay.label.config(text="Try your luck if you dare!")
        self.overlay.button.config(state="normal", bg="green", text="Go")

    def _pause_game(self):
        self.is_running = not self.is_running
        if self.is_running:
            self.pause_button.config(bg="green", text="\u23F8")
            self.window.after(self.config.time_step_size, self._game_loop)
        else:
            self.pause_button.config(bg="orange", text="\u25B6")
            self.vibro_motor.stop()

    def _reset_game(self):
        # code_button place_forget
        self.code_button.place_forget()
        # disable pause_button?
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
            self.hole_status_text, 
            text=f"Caught by {self.fell_into_holes} hole{'s' if self.fell_into_holes != 1 else ''}"
        )
        # reset hole cooldown
        self.hole_cool_down = 0
        # reset vibro
        self.vibro_motor.stop()
        # reset is_running
        self.is_running = False
        # reset is_finished
        # self.is_finished = False

    def close_app(self):
        self.mqtt_client.disconnect()
        self.vibro_motor.cleanup()
        self.window.destroy()

    def _overlay_handler(self, overlay_type: str = "none"):
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
        self.overlay.label.pack(pady=30)

        self.overlay.button = tk.Button(self.overlay.frame, 
                           text="  -----  ", 
                           state="disabled", 
                           command=lambda: [
                               self._overlay_handler("none"), 
                               self.pause_button.place(x=0, y=0, width=20, height=20),
                               self.code_button.place(x=0, y=460, width=20, height=20)
                           ],
                           bg="#eeeeee", font=("Arial", 16, "bold"))
        self.overlay.button.pack(pady=15)

    def _show_code_overlay(self):
        overlay_w, overlay_h = 300, 323
        overlay_x = (self.config.screen_width - overlay_w) // 2
        overlay_y = (self.config.screen_height - overlay_h) // 2
        self.overlay.frame = tk.Frame(self.window, bg="#eeeeee", bd=3, relief="ridge")
        self.overlay.frame.place(x=overlay_x, y=overlay_y, width=overlay_w, height=overlay_h)

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
            if val == "Del":
                code_var.set(code_var.get()[:-1])
            elif val == "OK":
                if code_var.get() == self.game_map.digit_code:
                    self.overlay.frame.destroy()
                    self.code_button.place_forget()
                    self.pause_button.place_forget()
                    self.mqtt_client.publish_result(self.fell_into_holes)
                    self._overlay_handler("finish")
                else:
                    code_var.set("")
            else:
                if len(code_var.get()) < 4:
                    code_var.set(code_var.get() + val)

        btns = [["1", "2", "3"], ["4", "5", "6"], ["7", "8", "9"], ["Del", "0", "OK"]]
        btn_frame = tk.Frame(self.overlay.frame, bg="#eeeeee")
        btn_frame.pack(pady=10)
        for r, row in enumerate(btns):
            for c, val in enumerate(row):
                b = tk.Button(btn_frame, text=val, width=5, height=1, font=("Arial", 16), command=lambda v=val: numpad_press(v))
                b.grid(row=r, column=c, padx=5, pady=5)

    def _show_finish_overlay(self):
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
        self.overlay.label.pack(pady=75)