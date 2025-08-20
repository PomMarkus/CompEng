import sys
from game_config import GameConfig
import tkinter as tk

if sys.platform == "linux":
    try:
        import RPi.GPIO as GPIO
        GPIO.setmode(GPIO.BCM)
    except ImportError:
        print("RPi.GPIO module not found")
        GPIO = None  # Set GPIO to None if not available

else:
    GPIO = None


class VibroMotor:
    def __init__(self, config: GameConfig, canvas: tk.Canvas=None):
        self.config = config
        self.canvas = canvas
        self.vibrate_cool_down = 0
        self.vibro_ind = None

        if GPIO:
            GPIO.setup(self.config.vibration_gpio, GPIO.OUT)
            self.high = lambda: GPIO.output(self.config.vibration_gpio, GPIO.HIGH)
            self.low = lambda: GPIO.output(self.config.vibration_gpio, GPIO.LOW)
        else:
            self.high = lambda: None
            self.low = lambda: None
        
        if not GPIO and self.canvas:
            self.vibro_ind = self.canvas.create_oval(30, 5, 40, 15, fill="gray", outline="black")
    
    def vibrate(self, cool_down):
        self.vibrate_cool_down = cool_down
        if GPIO:
            self.high()
        elif self.canvas:
            self.canvas.itemconfig(self.vibro_ind, fill="red")
        
    def update(self):
        if self.vibrate_cool_down > 0:
            self.vibrate_cool_down -= self.config.time_step_size
            if self.vibrate_cool_down <= 0:
                self.vibrate_cool_down = 0
                if GPIO:
                    self.low()
                elif self.canvas:
                    self.canvas.itemconfig(self.vibro_ind, fill="gray")

    def stop(self):
        if GPIO:
            self.low()
        elif self.canvas and self.vibro_ind:
            self.canvas.itemconfig(self.vibro_ind, fill="gray")
        
        self.vibrate_cool_down = 0

    def cleanup(self):
        if GPIO:
            GPIO.cleanup()
