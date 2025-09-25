import tkinter as tk
import numpy as np
from game_config import GameConfig
import re

# Value definitions for map pixel types
WALL = 2             # Wall
WALL_PERIPHERY = 1   # forbidden position around wall
VALID = 0            # Valid position
HOLE_AREA = -1  # Hole periphery
HOLE_CENTER = -2     # Hole center
CHECKPOINT = -3      # Checkpoint

PXTYPE = 0 # Indicates index of pixel definition in val_data array

class GameMap:
    """
    Represents the game map, including walls, holes, checkpoints, and start position.

    This class loads map objects from a file, generates a pixel value array for collision and game logic,
    and provides methods to access map information and draws the map.

    Attributes:
        config (GameConfig): Game configuration object.
        val_data (np.ndarray): 3D array storing the pixel type for each screen pixel (index 0)
          as well as a normal vector perpendicular to the surface of a obstacle if there is one (index 1,2) and the index of a checkpoint if there is one (index 3).
        start_point (np.ndarray): Current start position of the ball.
        start_point_default (np.ndarray): Default start position of the ball.
        map_objects (list): List of map objects parsed from the map file.
        checkpoints (list): List of checkpoint dictionaries with coordinates and names.
        digit_code (str): Extracted digit code from checkpoint namestring.
    """

    def __init__(self, map_filename: str, config: GameConfig):
        """
        Initializes the GameMap by loading map objects and generating the data array defining if pixels are valid positions on the map.

        Args:
            map_filename (str): Path to the map file.
            config (GameConfig): Game configuration object.
        """
        self.config = config
        self.val_data = np.zeros((self.config.screen_height, self.config.screen_width, 4)) # pxl type, normal vector x, normal vector y, checkpoint number - per default all pixels are valid
        self.start_point = np.array([0, 0], dtype=float)
        self.start_point_default = np.array([0, 0], dtype=float)
        self.map_objects = []
        self.checkpoints = []

        self._load_map_objects(map_filename)
        self._generate_val_data()
    
    def _load_map_objects(self, map_filename: str):
        """
        Loads map objects (walls, holes, checkpoints, start point) from the map file.

        Args:
            map_filename (str): Path to the map file.
        """

        # Scans the map file for the start point and holeradius and stores further map objects in a list
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

        # Parse checkpoint names from config and determine the map code
        parsed_checkpoint_names = self.config.checkpoint_namestring.split("\t")
        self.digit_code = re.sub(r'\D', '', self.config.checkpoint_namestring)

        # Create list of checkpoints with coordinates and names
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
        """
        Generates the val_data array, marking walls, holes, WALL_PERIPHERY zones, and checkpoints.

        This method sets up the collision and logic map for the game, including normal vectors
        for rebouncing and special areas for holes and checkpoints.
        """
        # Create circle template mask for the holes
        circle_hole = np.zeros((self.config.hole_radius * 2, self.config.hole_radius * 2))
        Y, X = np.ogrid[:self.config.hole_radius * 2, :self.config.hole_radius * 2] # create two arrays with y and x coordinates
        mask_hole = ((X - self.config.hole_radius + 0.5) ** 2) + ((Y - self.config.hole_radius + 0.5) ** 2) <= self.config.hole_radius ** 2 # mask for all points within the circle
        circle_hole[mask_hole] = HOLE_AREA # pixel type in periphery of holes is set to HOLE_AREA

        # generating a 2D array e.g. (-2, -1, 1, 2) for x and y
        idx = np.indices((self.config.hole_radius * 2, self.config.hole_radius * 2)) - self.config.hole_radius
        idx[idx >=0] += 1 # Removes 0 from the "center" by incrementing all positive values by one in order to obtain a symetrical distribution
        # combine pixel type and normal vector: the vectors are normalized by the hole radius and the pixeltype is used to point them to the center and suppress the outside of the hole
        circle_hole = np.stack((circle_hole, (idx[0]) * circle_hole / self.config.hole_radius, (idx[1]) * circle_hole / self.config.hole_radius), axis=-1) # combine pixel type and normal vector

        # Vectors are further scaled in order to simulate gravitational pull into the hole
        norm_squared = np.sum(circle_hole[:, :, 1:3] ** 2, axis=-1, keepdims=True)
        norm_squared[norm_squared == 0] = 1e-1
        circle_hole[:, :, 1:3] /= norm_squared

        # Set the center pixels of the hole to HOLE_CENTER
        inner_mask = ((X - self.config.hole_radius + 0.5) ** 2) + ((Y - self.config.hole_radius + 0.5) ** 2) <= (self.config.hole_radius - self.config.ball_radius) ** 2
        circle_hole[inner_mask, PXTYPE] = HOLE_CENTER

        checkpoint_counter = 0 # Used to assign unique indices to checkpoints in val_data

        # Mark pixeltypes of walls, holes and checkpoints listed in map_objects in val_data
        for obj in self.map_objects:
            # Set wall pixels to WALL
            if obj[0] == 'w':
                x1, y1, x2, y2 = int(obj[1]), int(obj[2]), int(obj[3]), int(obj[4])
                self.val_data[y1:y2, x1:x2, PXTYPE] = WALL # pixle occupied: 2

            # Set hole pixels to HOLE
            elif obj[0] == 'h':
                # Select the area in val_data corresponding to the hole position and fill it with the circle_hole template
                x, y = int(obj[1]), int(obj[2])
                subdata = self.val_data[y - self.config.hole_radius:y + self.config.hole_radius, x - self.config.hole_radius:x + self.config.hole_radius]
                subdata[mask_hole, :3] = circle_hole[mask_hole, :3] 
                self.val_data[y - self.config.hole_radius:y + self.config.hole_radius, x - self.config.hole_radius:x + self.config.hole_radius] = subdata # fill the hole with the circle_hole template
            
            # Checkpoint pxl to CHECKPOINT
            elif obj[0] == 'c':
                # Create a mask for defining the pxl inside the circle
                x, y = int(obj[1]), int(obj[2])
                Y, X = np.ogrid[:self.config.screen_height, :self.config.screen_width]
                mask_checkpoint = (((X - x) ** 2)) + ((Y - y) ** 2)<= self.config.hole_radius ** 2
                # Set the pixels inside the checkpoint to CHECKPOINT and set unique index
                self.val_data[mask_checkpoint, PXTYPE] = CHECKPOINT
                self.val_data[mask_checkpoint, 3] = checkpoint_counter
                checkpoint_counter += 1

                
        # Fill area of shifted rectangles with WALL_PERIPHERY and add the normalvector for the rebouncing calculation
        for obj in self.map_objects:

            if obj[0] == 'w':
                # Defining wall corners
                x1, y1, x2, y2 = int(obj[1]), int(obj[2]), int(obj[3]), int(obj[4])

                # Create rectangle above - normalvector points upwards
                shifted = y1- self.config.ball_radius
                if shifted < 0:
                    shifted = 0
                subdata = self.val_data[shifted:y1, x1:x2]
                subdata[subdata[:,:, PXTYPE] != WALL] += np.array([0, -1, 0, 0]) # add the normalvector
                subdata[subdata[:,:, PXTYPE] <= VALID, PXTYPE] = WALL_PERIPHERY # Set pixel type to WALL_PERIPHERY - done separately to avoid summing up pixel types
                self.val_data[shifted:y1, x1:x2] = subdata

                # Create rectangle below - normalvector points downwards
                shifted = y2 + self.config.ball_radius
                if shifted > self.config.screen_height:
                    shifted = self.config.screen_height
                subdata = self.val_data[y2:shifted, x1:x2]
                subdata[subdata[:,:, PXTYPE] != WALL] += np.array([0, 1, 0, 0]) # add the normalvector
                subdata[subdata[:,:, PXTYPE] <= VALID, PXTYPE] = WALL_PERIPHERY # Set pixel type to WALL_PERIPHERY - done separately to avoid summing up pixel types
                self.val_data[y2:shifted, x1:x2] = subdata

                # Create rectangle left - normalvector points left
                shifted = x1 - self.config.ball_radius
                if shifted < 0:
                    shifted = 0
                subdata = self.val_data[y1:y2, shifted:x1]
                subdata[subdata[:,:, PXTYPE] != WALL] += np.array([0, 0, -1, 0]) # add the normalvector
                subdata[subdata[:,:, PXTYPE] <= VALID, PXTYPE] = WALL_PERIPHERY # Set pixel type to WALL_PERIPHERY - done separately to avoid summing up pixel types
                self.val_data[y1:y2, shifted:x1] = subdata

                # Create rectangle right - normalvector points right
                shifted = x2 + self.config.ball_radius
                if shifted > self.config.screen_width:
                    shifted = self.config.screen_width
                subdata = self.val_data[y1:y2, x2:shifted]
                subdata[subdata[:,:, PXTYPE] != WALL] += np.array([0, 0, 1, 0]) # add the normalvector
                subdata[subdata[:,:, PXTYPE] <= VALID, PXTYPE] = WALL_PERIPHERY # Set pixel type to WALL_PERIPHERY - done separately to avoid summing up pixel types
                self.val_data[y1:y2, x2:shifted] = subdata

        
        # Create template mask of whole circle for wallcorners including normal vectors
        circle_wall = np.zeros((2 * self.config.ball_radius + 1, 2 * self.config.ball_radius + 1))
        Y, X = np.ogrid[:2 * self.config.ball_radius + 1, :2 * self.config.ball_radius + 1]
        mask_circle = ((X - self.config.ball_radius) ** 2) + ((Y - self.config.ball_radius) ** 2) <= self.config.ball_radius ** 2
        circle_wall[mask_circle] = WALL_PERIPHERY # corner surroundings of walls are also marked as such
        idx = np.indices((2 * self.config.ball_radius + 1, 2 * self.config.ball_radius + 1)) # generating a 2D array e.g. (-2, -1, 0, 1, 2) for x and y
        circle_wall = np.stack(
            (
             circle_wall, 
             (idx[0] - self.config.ball_radius) * circle_wall / self.config.ball_radius, 
             (idx[1]- self.config.ball_radius) * circle_wall / self.config.ball_radius, 
             np.zeros((2 * self.config.ball_radius + 1, 2 * self.config.ball_radius + 1))
            ), 
            axis=-1)

        # Define circular areas at the corners of walls as WALL_PERIPHERY zones
        for obj in self.map_objects:
            if obj[0] == 'w':
                # Define wall corners
                x1, y1, x2, y2 = int(obj[1]), int(obj[2]), int(obj[3]), int(obj[4])

                # Create masks for the respective corners and insert circle sector at each corner
                shifted_x1 = x1 - self.config.ball_radius
                shifted_y1 = y1 - self.config.ball_radius
                shifted_x2 = x2 + self.config.ball_radius
                shifted_y2 = y2 + self.config.ball_radius
                if shifted_x1 < 0:
                    shifted_x1 = 0
                if shifted_y1 < 0:
                    shifted_y1 = 0
                if shifted_x2 > self.config.screen_width:
                    shifted_x2 = self.config.screen_width
                if shifted_y2 > self.config.screen_height:
                    shifted_y2 = self.config.screen_height

                diff_x1 = x1 - shifted_x1
                offset_x1 = self.config.ball_radius - diff_x1
                diff_y1 = y1 - shifted_y1
                offset_y1 = self.config.ball_radius - diff_y1
                diff_x2 = shifted_x2 - x2
                offset_x2 = self.config.ball_radius + diff_x2 + 1
                diff_y2 = shifted_y2 - y2
                offset_y2 = self.config.ball_radius + diff_y2 + 1

                # Top left corner
                mask = np.isin(self.val_data[shifted_y1:y1, shifted_x1:x1, PXTYPE], [VALID, HOLE_AREA])
                mask[mask_circle[offset_y1: self.config.ball_radius, offset_x1: self.config.ball_radius] != 1] = False
                self.val_data[shifted_y1:y1, shifted_x1:x1][mask] = circle_wall[offset_y1: self.config.ball_radius, offset_x1: self.config.ball_radius][mask]

                # Top right corner
                mask = np.isin(self.val_data[shifted_y1:y1, x2:shifted_x2, PXTYPE], [VALID, HOLE_AREA])
                mask[mask_circle[offset_y1: self.config.ball_radius, self.config.ball_radius + 1:offset_x2] != 1] = False
                self.val_data[shifted_y1:y1, x2:shifted_x2][mask] = circle_wall[offset_y1: self.config.ball_radius, self.config.ball_radius + 1:offset_x2][mask]
                
                # Bottom left corner
                mask = np.isin(self.val_data[y2:shifted_y2, shifted_x1:x1, PXTYPE], [VALID, HOLE_AREA])
                mask[mask_circle[self.config.ball_radius + 1:offset_y2, offset_x1: self.config.ball_radius] != 1] = False
                self.val_data[y2:shifted_y2, shifted_x1:x1][mask] = circle_wall[ self.config.ball_radius + 1:offset_y2, offset_x1: self.config.ball_radius][mask]
                
                # Bottom right corner
                mask = np.isin(self.val_data[y2:shifted_y2, x2:shifted_x2, PXTYPE], [VALID, HOLE_AREA])
                mask[mask_circle[self.config.ball_radius + 1:offset_y2, self.config.ball_radius + 1:offset_x2] != 1] = False
                self.val_data[y2:shifted_y2, x2:shifted_x2][mask] = circle_wall[self.config.ball_radius + 1:offset_y2, self.config.ball_radius + 1:offset_x2][mask]

    def draw_map(self, canvas: tk.Canvas):
        """
        Draws the map objects (walls and holes) onto a Tkinter canvas.

        Args:
            canvas (tk.Canvas): The canvas to draw the map on.
        """
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
        """
        Returns the value information for a given pixel coordinate.

        Args:
            x (int): X coordinate.
            y (int): Y coordinate.

        Returns:
            np.ndarray: Array containing pixel type and additional data.
        """

        # Return HOLE_CENTER for out-of-bounds coordinates to reset ball to start point
        if not (0 <= y < self.config.screen_height and 0 <= x < self.config.screen_width):
            return np.array([HOLE_CENTER, 0, 0, 0])
        return self.val_data[y, x, :]

    def get_start_point(self) -> np.ndarray:
        """
        Returns a copy of the current start point.

        Returns:
            np.ndarray: Start point coordinates.
        """
        return self.start_point.copy()
    
    def set_start_point(self, x: int, y: int):
        """
        Sets the start point to the given coordinates.

        Args:
            x (int): X coordinate.
            y (int): Y coordinate.
        """
        self.start_point = np.array([x, y], dtype=float)
                
    def reset_start_point(self) -> np.ndarray:
        """
        Resets the start point to its default value.

        Returns:
            np.ndarray: Reset start point coordinates.
        """
        self.start_point = self.start_point_default.copy()

    def get_checkpoint_init_data(self) -> list:
        """
        Returns the list of checkpoint initialization data.

        Returns:
            list: List of checkpoint dictionaries.
        """
        return self.checkpoints

