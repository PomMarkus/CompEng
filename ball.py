import tkinter as tk
import numpy as np
from game_config import GameConfig
from game_map import GameMap
from game_map import WALL, WALL_PERIPHERY, VALID, HOLE_AREA, HOLE_CENTER, CHECKPOINT, PXTYPE
from vibro_motor import VibroMotor
from checkpoint import Checkpoint

class Ball:
    """
    Represents the ball object in the 2D game environment, handling its physics, collisions, and interactions with the game map.
    Attributes:
        canvas (tk.Canvas): The canvas on which the ball is drawn.
        config (GameConfig): Configuration parameters for the game and ball physics.
        gamemap (GameMap): The game map providing collision and environment information.
        vibro_motor (VibroMotor): Motor used to provide vibration feedback on collisions.
        checkpoints (list[Checkpoint]): List of checkpoints in the game.
        position (np.ndarray): Current position of the ball as a 2D vector.
        velocity (np.ndarray): Current velocity of the ball as a 2D vector.
        canvas_id (int): Canvas object ID for the ball's oval representation.
    """

    def __init__(self, canvas: tk.Canvas, config: GameConfig, gamemap: GameMap, vibro_motor: VibroMotor, checkpoints: list[Checkpoint]):
        """
        Initializes the Ball object with the given canvas, configuration, game map, vibro motor, and checkpoints.
        Args:
            canvas (tk.Canvas): The Tkinter canvas where the ball will be drawn.
            config (GameConfig): Configuration object containing game settings such as ball radius.
            gamemap (GameMap): The game map object providing the start point and map-related logic.
            vibro_motor (VibroMotor): The vibro motor controller for haptic feedback.
            checkpoints (list[Checkpoint]): List of checkpoint objects for the game.
        """

        self.canvas = canvas
        self.config = config
        self.gamemap = gamemap
        self.vibro_motor = vibro_motor
        self.checkpoints = checkpoints
        self.position = self.gamemap.get_start_point()
        self.velocity = np.array([0, 0], dtype=float)

        self.canvas_id = self.canvas.create_oval(
            int(self.position[0]) - self.config.ball_radius + 1, int(self.position[1]) - self.config.ball_radius + 1,
            int(self.position[0]) + self.config.ball_radius, int(self.position[1]) + self.config.ball_radius,
            fill="blue", outline="blue"
        ) 

    def update_position(self, acc_x: float, acc_y: float):
        """
        Updates the position and velocity of the ball based on the provided acceleration values.
        This method simulates the movement of the ball in discrete steps, taking into account acceleration,
        velocity, collisions with obstacles, holes, and checkpoints on the game map. It also handles
        collision response, velocity damping, and triggers vibration feedback when significant collisions occur.
        Args:
            acc_x (float): Acceleration in the x-direction.
            acc_y (float): Acceleration in the y-direction.
        Returns:
            list: A list of strings indicating special events that occurred during the update, such as
                  "hole" if the ball falls into a hole, or "checkpoint" if a checkpoint is reached.
        """
        
        return_statement = [] # In certain cases, statements like "hole" or "checkpoint" are appended to this list
        last_position = self.position.copy() # Is used to determine path travelled in one time step in order to prevent vibration in case of constant contact
        normal_vectors = set() # At each interaction with a wall, the normal vectors are stored here to manage the vibration

        self.velocity[0] += acc_x * self.config.acceleration_factor * self.config.time_step_size / 1000
        self.velocity[1] += acc_y * self.config.acceleration_factor * self.config.time_step_size / 1000

        Dpos = np.array(self.velocity) * self.config.time_step_size / 1000 # Trajectory the ball should travel in this time step based on the velocity
        dist = np.linalg.norm(Dpos)
        steps = int(dist / self.config.position_step_size) if dist > self.config.position_step_size else 1 # Number of steps the trajectory is divided into to ensure collision detection

        dstep = Dpos / steps # Next step vector
        counter = 0
        security_counter = 0 # Prevents infinite loops

        while counter < steps:
            security_counter += 1
            if security_counter > 1000:
                print("Security limit reached, breaking loop")
                break #quit()?
            temp_pos = self.position + dstep # New position to be checked for collisions

            val_info = self.gamemap.get_val_info(int(temp_pos[0]), int(temp_pos[1])) # Get information about the pixel at the new position (if pixel is valid, hole, checkpoint, obstacle)
            px_type = val_info[PXTYPE]

            if px_type > VALID: # Collision with wall or WALL_PERIPHERY area
                vec_norm = val_info[2:0:-1] # Normal vector stored for this pixel (reversed order to get [nx, ny])
                normal_vectors.add(tuple(vec_norm))
                pos_dot_product = np.dot(vec_norm, Dpos) # Projection of current trajectory on the normal vector - used to determine consequent direction of movement
                
                if (pos_dot_product < 0): # Negative value means, that the ball is moving towards the wall - physically correct scenario
                    
                    # Current trajectory and velocity are mirrored with respect to the normal vector of the obstacle
                    vec_proj_pos = pos_dot_product / np.dot(vec_norm, vec_norm) * vec_norm
                    vec_proj_vel = np.dot(vec_norm, self.velocity) / np.dot(vec_norm, vec_norm) * vec_norm
                    Dpos = - 2 * vec_proj_pos + Dpos
                    self.velocity = - 2 * vec_proj_vel + self.velocity

                    # Damping of the mirrored components to simulate energy loss during the collision
                    Dpos += vec_proj_pos * self.config.damping_factor
                    self.velocity += vec_proj_vel * self.config.damping_factor

                    # Recalculation of the remaining trajectory after the collision
                    dist = np.linalg.norm(Dpos)
                    steps = int(dist / self.config.position_step_size) if dist > self.config.position_step_size else 1
                    dstep = Dpos / steps

                    counter = 0
                    continue

                else: # Should physically not happen, but due to pixel based representation of arcs, such results are possible and would pull the ball towards corners -> push away
                    shift = vec_norm / np.linalg.norm(vec_norm) * self.config.position_step_size # Direction of the normal vector with length of one position step  
                    self.position += shift # Small shift away from the wall
                    Dpos -= shift # Consider the shift in the remaining trajectory
                    counter += 1
                    continue
            
            elif px_type == HOLE_CENTER: # Ball falls into the hole
                self.reset_velocity()
                return_statement.append("hole")
                break

            elif px_type == CHECKPOINT: # Ball reaches a checkpoint
                c_number = int(val_info[3]) # Checkpoint number
                if c_number < len(self.checkpoints) and not self.checkpoints[c_number].is_reached:
                    self.checkpoints[c_number].mark_reached()
                    start_point = np.array(self.checkpoints[c_number].get_center_coords(), dtype=float)
                    self.gamemap.set_start_point(*start_point) # New start point is the center of the checkpoint
                    return_statement.append("checkpoint")
            
            # If no wall or hole is hit, the step is executed as planned
            self.position += dstep
            Dpos -= dstep
            counter += 1

        # Mechanism to pull the ball into holes
        val_info = self.gamemap.get_val_info(int(self.position[0]), int(self.position[1]))
        if (val_info[0] == HOLE_AREA): 
            vec_norm = val_info[2:0:-1] # Normal vector stored for this pixel (reversed order to get [nx, ny])
            vec_tang = np.array([vec_norm[1], -vec_norm[0]])
            vec_proj_vel_tang = np.dot(vec_tang, self.velocity) / np.dot(vec_tang, vec_tang) * vec_tang # Projection of the velocity onto the tangential vector
            self.velocity += vec_norm * 10 # Pull into the hole
            self.velocity -= vec_proj_vel_tang * 0.3 # Damping of the tangential component to avoid circling around the hole

        # Check for each collision, if it is "new" or if the ball is in constant contact with the wall - only vibrate in the first case
        pos_difference = self.position - last_position
        for vector in normal_vectors:
            vec_norm = np.array(vector, dtype=float)
            vec_proj_difference = np.dot(vec_norm, pos_difference) / np.dot(vec_norm, vec_norm) * vec_norm # Projection of the path travelled in this time step onto the normal vector
            if np.linalg.norm(vec_proj_difference) > 0.1:
                self.vibro_motor.vibrate(100)
        
        return return_statement


    def draw(self):
        """
        Updates the position of the ball on the canvas by setting the coordinates
        of the canvas object associated with this ball.
        The coordinates are calculated based on the current position of the ball
        and its radius, ensuring the ball is drawn at the correct location.
        Returns:
            None
        """

        self.canvas.coords(
            self.canvas_id,
            int(self.position[0]) - self.config.ball_radius + 1, 
            int(self.position[1]) - self.config.ball_radius + 1,
            int(self.position[0]) + self.config.ball_radius, 
            int(self.position[1]) + self.config.ball_radius
        )
    

    def reset_position(self):
        """
        Resets the ball's position to the starting point defined by the game map.
        This method updates the `position` attribute of the ball to the coordinates
        returned by `self.gamemap.get_start_point()`, effectively moving the ball
        back to its initial location.
        Returns:
            None
        """

        self.position = self.gamemap.get_start_point()


    def reset_velocity(self):
        """
        Resets the velocity of the ball to zero.
        This method sets the velocity attribute to a NumPy array [0, 0] of type float,
        effectively stopping any movement of the ball.
        Returns:
            None
        """

        self.velocity = np.array([0, 0], dtype=float)