import tkinter as tk
import numpy as np

class Checkpoint:
    """
    Represents and handles a checkpoint on the game map.

    Attributes:
        position (np.ndarray): The (x, y) coordinates of the checkpoint center.
        radius (int): The radius of the checkpoint circle.
        name (str): The name of the checkpoint.
        canvas (tk.Canvas): The Tkinter canvas where the checkpoint is drawn.
        is_reached (bool): Whether the checkpoint has been reached.
        canvas_oval_id (int): The canvas item ID for the oval.
        canvas_text_id (int): The canvas item ID for the text.
    """

    def __init__(self, x: int, y: int, radius: int, name: str, canvas: tk.Canvas):
        """
        Initializes a Checkpoint instance and draws it on the canvas.

        Args:
            x (int): X-coordinate of the checkpoint center.
            y (int): Y-coordinate of the checkpoint center.
            radius (int): Radius of the checkpoint circle.
            name (str): Name of the checkpoint.
            canvas (tk.Canvas): Canvas to draw the checkpoint on.
        """
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
        """
        Marks the checkpoint as reached, updates its color to green,
        and displays its name on the canvas.
        """
        if not self.is_reached:
            self.is_reached = True
            self.canvas.itemconfig(self.canvas_oval_id, fill="green")
            self.canvas.itemconfig(self.canvas_text_id, text=self.name)
            
    def reset(self):
        """
        Resets the checkpoint to its initial state, changing its color to orange
        and removing its name from the canvas.
        """
        self.is_reached = False
        self.canvas.itemconfig(self.canvas_oval_id, fill="orange")
        self.canvas.itemconfig(self.canvas_text_id, text="")

    def get_center_coords(self) -> np.ndarray:
        """
        Returns the center coordinates of the checkpoint.

        Returns:
            np.ndarray: The (x, y) coordinates of the checkpoint center.
        """
        return self.position
    