import tkinter as tk
import sys
from abc import ABC, abstractmethod

if sys.platform == "linux":
    try:
        from mpu6050 import mpu6050
    except ImportError:
        print("mpu6050 module not found")
        mpu6050 = None
else:
    mpu6050 = None

class InputControl(ABC):
    """
    Abstract base class for input control devices.

    Defines the interface for status checking and acceleration retrieval.
    """

    @abstractmethod
    def status(self) -> bool:
        """
        Check if the input control is available and functioning.

        Returns:
            bool: True if the control is available, False otherwise.
        """
        pass

    @abstractmethod
    def get_acceleration(self) -> tuple[float, float]:
        """
        Get the current acceleration values from the input control.

        Returns:
            tuple[float, float]: The (x, y) acceleration values.
        """
        pass


class KeyboardControl(InputControl):
    """
    Input control implementation using keyboard arrow keys.

    Listens for key press and release events to simulate acceleration.
    """

    def __init__(self, window: tk.Tk):
        """
        Initialize the KeyboardControl.

        Args:
            window (tk.Tk): The Tkinter window to bind key events to.
        """
        self.pressend_keys = set()
        window.bind("<KeyPress>", self._on_key_press)
        window.bind("<KeyRelease>", self._on_key_release)

    def _on_key_press(self, event):
        """
        Handle key press events.

        Args:
            event: The Tkinter event object.
        """
        self.pressend_keys.add(event.keysym)
    
    def _on_key_release(self, event):
        """
        Handle key release events.

        Args:
            event: The Tkinter event object.
        """
        self.pressend_keys.discard(event.keysym)

    def status(self) -> bool:
        """
        Always returns True for keyboard input.

        Returns:
            bool: True
        """
        return True

    def get_acceleration(self) -> tuple[float, float]:
        """
        Calculate acceleration based on pressed arrow keys.

        Returns:
            tuple[float, float]: The (x, y) acceleration values.
        """
        acc_x = 0.0
        acc_y = 0.0

        if "Up" in self.pressend_keys:
            acc_y -= 7
        elif "Down" in self.pressend_keys:
            acc_y += 7
        if "Left" in self.pressend_keys:
            acc_x -= 7
        elif "Right" in self.pressend_keys:
            acc_x += 7

        return (acc_x, acc_y)
    

class MPU6050Control(InputControl):
    """
    Input control implementation using the MPU6050 accelerometer sensor.

    Reads acceleration data from the sensor if available.
    """

    def __init__(self):
        """
        Initialize the MPU6050Control.

        Attempts to connect to the MPU6050 sensor if available.
        """
        if mpu6050 is None:
            self.sensor = None
            return
        self.sensor = mpu6050(0x68)

    def status(self) -> bool:
        """
        Check if the MPU6050 sensor is available.

        Returns:
            bool: True if the sensor is available, False otherwise.
        """
        if mpu6050 is not None:
            return True
        else:
            return False

    def get_acceleration(self) -> tuple[float, float]:
        """
        Retrieve acceleration data from the MPU6050 sensor.

        Tries up to 10 times to read data from the sensor. If unsuccessful, returns (0.0, 0.0).

        Returns:
            tuple[float, float]: The (y, -x) acceleration values, compensating for sensor orientation.
        """
        for i in range(10):
            try:
                accel_data = self.sensor.get_accel_data()
                return (accel_data['y'], -accel_data['x']) # Compensates sensor orientation
            except:
                print("MPU Error count: ", i + 1)

        return (0.0, 0.0)  # Return zero if sensor fails to read after retries
