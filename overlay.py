import tkinter as tk

class Overlay:
    """
    Overlay class for storing UI elements over a Tkinter canvas.

    Provides a method to remove the overlay frame and background.
    """

    def __init__(self, canvas: tk.Canvas):
        """
        Initialize the Overlay with a Tkinter canvas.

        Args:
            canvas (tk.Canvas): The canvas on which to display the overlay.
        """
        self.canvas = canvas
        self.frame: tk.Frame = None
        self.background = None
        self.label: tk.Label = None
        self.button: tk.Button = None

    def close(self):
        """
        Remove the overlay frame and background from the canvas.
        """
        if self.frame is not None:
            self.frame.destroy()
            self.frame = None
        if self.background is not None:
            self.canvas.delete(self.background)
            self.background = None