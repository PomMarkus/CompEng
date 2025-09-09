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
    """
    Controls a vibration motor using GPIO on a Raspberry Pi or visualizes active vibration in the Tkinter canvas.

    Provides methods to vibrate, update, stop, and clean up the motor or the visualization.
    """

    def __init__(self, config: GameConfig, canvas: tk.Canvas=None):
        """
        Initialize the VibroMotor with configuration and optional Tkinter canvas.

        Args:
            config (GameConfig): The game configuration containing GPIO and timing settings.
            canvas (tk.Canvas, optional): Canvas for visualizing the vibration if GPIO is unavailable.
        """
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
        """
        Activate the vibration motor or visualize for a specified cooldown period.

        Args:
            cool_down (float): Duration to keep the motor vibrating.
        """
        self.vibrate_cool_down = cool_down
        if GPIO:
            self.high()
        elif self.canvas:
            self.canvas.itemconfig(self.vibro_ind, fill="red")
        
    def update(self):
        """
        Update the vibration motor state, decreasing the cooldown timer and
        deactivating the motor or visualization if the cooldown has expired.
        """
        if self.vibrate_cool_down > 0:
            self.vibrate_cool_down -= self.config.time_step_size
            if self.vibrate_cool_down <= 0:
                self.vibrate_cool_down = 0
                if GPIO:
                    self.low()
                elif self.canvas:
                    self.canvas.itemconfig(self.vibro_ind, fill="gray")

    def stop(self):
        """
        Immediately stop the vibration motor or visualization and reset the cooldown timer.
        """
        if GPIO:
            self.low()
        elif self.canvas and self.vibro_ind:
            self.canvas.itemconfig(self.vibro_ind, fill="gray")
        
        self.vibrate_cool_down = 0

    def cleanup(self):
        """
        Clean up the GPIO settings or any other resources used by the VibroMotor.
        """
        if GPIO:
            GPIO.cleanup()
