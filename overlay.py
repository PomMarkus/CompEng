import tkinter as tk

class Overlay:
    def __init__(self, canvas: tk.Canvas):
        self.canvas = canvas
        self.frame: tk.Frame = None
        self.background = None
        self.label: tk.Label = None
        self.button: tk.Button = None

    def close(self):
        if self.frame is not None:
            self.frame.destroy()
            self.frame = None
        if self.background is not None:
            self.canvas.delete(self.background)
            self.background = None