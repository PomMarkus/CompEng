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
            acc_y -= 7
        elif "Down" in self.pressend_keys:
            acc_y += 7
        if "Left" in self.pressend_keys:
            acc_x -= 7
        elif "Right" in self.pressend_keys:
            acc_x += 7

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
        for i in range(10):
            try:
                accel_data = self.sensor.get_accel_data()
                return (accel_data['y'], -accel_data['x']) # Compensates sensor orientation
            except:
                print("MPU Error count: ", i + 1)

        return (0.0, 0.0)  # Return zero if sensor fails to read after retries
        