import tkinter as tk
import numpy as np
from game_config import GameConfig
import re

class GameMap:
    def __init__(self, map_filename: str, config: GameConfig):
        self.config = config
        self.val_data = np.zeros((self.config.screen_height, self.config.screen_width, 4))
        self.start_point = np.array([0, 0], dtype=float)
        self.start_point_default = np.array([0, 0], dtype=float)
        self.map_objects = []
        self.checkpoints = []

        self._load_map_objects(map_filename)
        self._generate_val_data()
    
    def _load_map_objects(self, map_filename: str):
        with open(map_filename, "r") as file:
            for line in file:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if parts[0] == "s":
                    self.start_point = np.array([int(parts[1]), int(parts[2])], dtype=float)
                    self.start_point_default = self.start_point.copy()
                elif parts[0] == "r":
                    self.config.hole_radius = int(parts[1])
                else:
                    self.map_objects.append(parts)

        parsed_checkpoint_names = self.config.checkpoint_namestring.split("\t")
        self.digit_code = re.sub(r'\D', '', self.config.checkpoint_namestring)

        checkpoint_counter = 0
        for obj in self.map_objects:
            if obj[0] == 'c':
                if checkpoint_counter < len(parsed_checkpoint_names):
                    name = parsed_checkpoint_names[checkpoint_counter]
                else:
                    name = "--"
                self.checkpoints.append({
                    "x": int(obj[1]),
                    "y": int(obj[2]),
                    "name": name
                })
                checkpoint_counter += 1
                
    def _generate_val_data(self):
        # Create template mask for circle for the holes
        circle_hole = np.zeros((self.config.hole_radius * 2, self.config.hole_radius * 2))
        Y, X = np.ogrid[:self.config.hole_radius * 2, :self.config.hole_radius * 2]
        mask_hole = ((X - self.config.hole_radius + 0.5) ** 2) + ((Y - self.config.hole_radius + 0.5) ** 2) <= self.config.hole_radius ** 2
        circle_hole[mask_hole] = -1 # corner surroundings of walls are also marked as such

        # generating a 2D array e.g. (-2, -1, 0, 1, 2) for x and y
        idx = np.indices((self.config.hole_radius * 2, self.config.hole_radius * 2)) - self.config.hole_radius
        idx[idx >=0] += 1
        circle_hole = np.stack((circle_hole, (idx[0]) * circle_hole / self.config.ball_radius, (idx[1]) * circle_hole / self.config.ball_radius), axis=-1)

        norm_squared = np.sum(circle_hole[:, :, 1:3] ** 2, axis=-1, keepdims=True)
        norm_squared[norm_squared == 0] = 1e-1
        circle_hole[:, :, 1:3] /= norm_squared

        inner_mask = ((X - self.config.hole_radius + 0.5) ** 2) + ((Y - self.config.hole_radius + 0.5) ** 2) <= (self.config.hole_radius - self.config.ball_radius) ** 2
        circle_hole[inner_mask, 0] = -2

        checkpoint_counter = 0

        # Fill area of circles
        for obj in self.map_objects:
            # Wall pxl to 2
            if obj[0] == 'w':
                x1, y1, x2, y2 = int(obj[1]), int(obj[2]), int(obj[3]), int(obj[4])
                self.val_data[y1:y2, x1:x2, 0] = 2 # pixle occupied: 2

            # hole pxl to -1 
            elif obj[0] == 'h': # maybe remove this because of redundancy
                x, y = int(obj[1]), int(obj[2])
                
                subdata = self.val_data[y - self.config.hole_radius:y + self.config.hole_radius, x - self.config.hole_radius:x + self.config.hole_radius]
                subdata[mask_hole, :3] = circle_hole[mask_hole, :3]
                self.val_data[y - self.config.hole_radius:y + self.config.hole_radius, x - self.config.hole_radius:x + self.config.hole_radius] = subdata # fill the hole with the circle_hole template
            
            # Checkpoint pxl to -3
            elif obj[0] == 'c':
                x, y = int(obj[1]), int(obj[2])
                # Create a mask for defining the pxl inside the circle
                Y, X = np.ogrid[:self.config.screen_height, :self.config.screen_width]
                mask_checkpoint = (((X - x) ** 2)) + ((Y - y) ** 2)<= self.config.hole_radius ** 2
                # Set the pixels inside the checkoint to -3
                self.val_data[mask_checkpoint, 0] = -3
                self.val_data[mask_checkpoint, 3] = checkpoint_counter
                checkpoint_counter += 1

                

        # Fill area of shifted rectangles with 1 and add the normalvector for the rebouncing calculation
        for obj in self.map_objects:

            if obj[0] == 'w':
                x1, y1, x2, y2 = int(obj[1]), int(obj[2]), int(obj[3]), int(obj[4])
                Y, X = np.ogrid[:self.config.screen_height, :self.config.screen_width]

                # Create rectangle above - normalvector points upwards
                subdata = self.val_data[y1- self.config.ball_radius:y1, x1:x2]
                subdata[subdata[:,:, 0] <= 0, 0] = 1 # making sure, that the area around a wall element is marked as surrounding area only once
                subdata[subdata[:,:, 0] != 2] += np.array([0, -1, 0, 0]) # add the normalvector
                self.val_data[y1- self.config.ball_radius:y1, x1:x2] = subdata

                # Create rectangle below - normalvector points downwards
                subdata = self.val_data[y2:y2+ self.config.ball_radius, x1:x2]
                subdata[subdata[:,:, 0] <= 0, 0] = 1 # making sure, that the area around a wall element is marked as surrounding area only once
                subdata[subdata[:,:, 0] != 2] += np.array([0, 1, 0, 0]) # add the normalvector
                self.val_data[y2:y2+ self.config.ball_radius, x1:x2] = subdata

                # Create rectangle left - normalvector points left
                subdata = self.val_data[y1:y2, x1- self.config.ball_radius:x1]
                subdata[subdata[:,:, 0] <= 0, 0] = 1 # making sure, that the area around a wall element is marked as surrounding area only once
                subdata[subdata[:,:, 0] != 2] += np.array([0, 0, -1, 0]) # add the normalvector
                self.val_data[y1:y2, x1- self.config.ball_radius:x1] = subdata

                # Create rectangle right - normalvector points right
                subdata = self.val_data[y1:y2, x2:x2+ self.config.ball_radius]
                subdata[subdata[:,:, 0] <= 0, 0] = 1 # making sure, that the area around a wall element is marked as surrounding area only once
                subdata[subdata[:,:, 0] != 2] += np.array([0, 0, 1, 0]) # add the normalvector
                self.val_data[y1:y2, x2:x2+ self.config.ball_radius] = subdata


        # Create template mask for whole circle for wallcorners
        circle_wall = np.zeros((2* self.config.ball_radius + 1, 2* self.config.ball_radius + 1))
        Y, X = np.ogrid[:2* self.config.ball_radius + 1, :2* self.config.ball_radius + 1]
        mask_corner = ((X - self.config.ball_radius) ** 2) + ((Y - self.config.ball_radius) ** 2) <= self.config.ball_radius ** 2
        circle_wall[mask_corner] = 1 # corner surroundings of walls are also marked as such

        # generating a 2D array e.g. (-2, -1, 0, 1, 2) for x and y
        idx = np.indices((2* self.config.ball_radius + 1, 2* self.config.ball_radius + 1))
        circle_wall = np.stack((circle_wall, (idx[0] - self.config.ball_radius) * circle_wall / self.config.ball_radius, (idx[1]- self.config.ball_radius) * circle_wall / self.config.ball_radius, np.zeros((2*self.config.ball_radius + 1, 2*self.config.ball_radius + 1))), axis=-1)


        for obj in self.map_objects:
            if obj[0] == 'w':
                x1, y1, x2, y2 = int(obj[1]), int(obj[2]), int(obj[3]), int(obj[4])
                # Insert circle sector
                # Top left corner
                mask = np.isin(self.val_data[y1- self.config.ball_radius:y1, x1- self.config.ball_radius:x1, 0], [0, -1]) # mask for all points within the circle sector at the top left corner

                self.val_data[y1-self.config.ball_radius:y1, x1-self.config.ball_radius:x1][mask] = circle_wall[: self.config.ball_radius, : self.config.ball_radius][mask]
                # Top right corner
                mask = np.isin(self.val_data[y1-self.config.ball_radius:y1, x2:x2+self.config.ball_radius, 0], [0, -1]) # mask for all points within the circle sector at the top right corner
                self.val_data[y1-self.config.ball_radius:y1, x2:x2+self.config.ball_radius][mask] = circle_wall[: self.config.ball_radius, self.config.ball_radius + 1:][mask]
                # Bottom left corner
                mask = np.isin(self.val_data[y2:y2+ self.config.ball_radius, x1- self.config.ball_radius:x1, 0], [0, -1]) # mask for all points within the circle sector at the bottom left corner
                self.val_data[y2:y2+self.config.ball_radius, x1-self.config.ball_radius:x1][mask] = circle_wall[ self.config.ball_radius + 1:, : self.config.ball_radius][mask]
                # Bottom right corner
                mask = np.isin(self.val_data[y2:y2+ self.config.ball_radius, x2:x2+ self.config.ball_radius, 0], [0, -1]) # mask for all points within the circle sector at the bottom right corner
                self.val_data[y2:y2+ self.config.ball_radius, x2:x2+ self.config.ball_radius][mask] = circle_wall[ self.config.ball_radius + 1:, self.config.ball_radius + 1:][mask]

    def draw_map(self, canvas: tk.Canvas):
        for obj in self.map_objects:
            if obj[0] == 'w':
                x1, y1, x2, y2 = int(obj[1]), int(obj[2]), int(obj[3]), int(obj[4])
                canvas.create_rectangle(x1, y1, x2, y2, fill="#471F01", outline="#471F01")
            elif obj[0] == 'h':
                x, y = int(obj[1]), int(obj[2])
                canvas.create_oval(
                    x - self.config.hole_radius,
                    y - self.config.hole_radius,
                    x + self.config.hole_radius,
                    y + self.config.hole_radius,
                    fill="black", outline="lightgray"
                    )

    def get_val_info(self, x: int, y: int) -> np.ndarray:
        if not (0 <= y < self.config.screen_height and 0 <= x < self.config.screen_width):
            return np.array([-2, 0, 0, 0])
        return self.val_data[y, x, :]

    def get_start_point(self) -> np.ndarray:
        return self.start_point.copy()
    
    def set_start_point(self, x: int, y: int):
        self.start_point = np.array([x, y], dtype=float)
                
    def reset_start_point(self) -> np.ndarray:
        self.start_point = self.start_point_default.copy()

    def get_checkpoint_init_data(self) -> list:
        return self.checkpoints

