import tkinter as tk
import numpy as np

class Checkpoint:
    def __init__(self, x: int, y: int, radius: int, name: str, canvas: tk.Canvas):
        self.position = np.array([x, y], dtype=float)
        self.radius = radius
        self.name = name
        self.canvas = canvas
        self.is_reached = False

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
            self.canvas.itemconfig(self.canvas_oval_id, fill="green")
            self.canvas.itemconfig(self.canvas_text_id, text=self.name)
            
    def reset(self):
        self.is_reached = False
        self.canvas.itemconfig(self.canvas_oval_id, fill="orange")
        self.canvas.itemconfig(self.canvas_text_id, text="")

    def get_center_coords(self) -> np.ndarray:
        return self.position
    