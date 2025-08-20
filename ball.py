import tkinter as tk
import numpy as np
from game_config import GameConfig
from game_map import GameMap
from vibro_motor import VibroMotor
from checkpoint import Checkpoint

class Ball:
    def __init__(self, canvas: tk.Canvas, config: GameConfig, gamemap: GameMap, vibro_motor: VibroMotor, checkpoints: list[Checkpoint]):
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
        return_statement = []
        last_position = self.position.copy()
        normal_vectors = set()

        self.velocity[0] += acc_x * self.config.acceleration_factor * self.config.time_step_size / 1000
        self.velocity[1] += acc_y * self.config.acceleration_factor * self.config.time_step_size / 1000

        Dpos = np.array(self.velocity) * self.config.time_step_size / 1000
        dist = np.linalg.norm(Dpos)
        steps = int(dist / self.config.position_step_size) if dist > self.config.position_step_size else 1

        dstep = Dpos / steps
        counter = 0
        security_counter = 0

        while counter < steps:
            security_counter += 1
            if security_counter > 1000:
                print("Security limit reached, breaking loop")
                break #quit()?
            temp_pos = self.position + dstep

            val_info = self.gamemap.get_val_info(int(temp_pos[0]), int(temp_pos[1]))
            px_type = val_info[0]

            if px_type > 0:
                vec_norm = val_info[2:0:-1]
                normal_vectors.add(tuple(vec_norm))
                pos_dot_product = np.dot(vec_norm, Dpos)
                if (pos_dot_product < 0):
                    vec_proj_pos = pos_dot_product / np.dot(vec_norm, vec_norm) * vec_norm
                    vec_proj_vel = np.dot(vec_norm, self.velocity) / np.dot(vec_norm, vec_norm) * vec_norm

                    Dpos = - 2 * vec_proj_pos + Dpos
                    self.velocity = - 2 * vec_proj_vel + self.velocity

                    # Dpos *= (steps - counter) / (steps)
                    Dpos += vec_proj_pos * self.config.damping_factor
                    self.velocity += vec_proj_vel * self.config.damping_factor
                    dist = np.linalg.norm(Dpos)
                    steps = int(dist / self.config.position_step_size) if dist > self.config.position_step_size else 1
                    dstep = Dpos / steps
                    counter = 0
                    continue

                else:
                    shift = vec_norm / np.linalg.norm(vec_norm) * self.config.position_step_size
                    self.position += shift
                    Dpos -= shift
                    counter += 1
                    continue
            
            elif px_type == -2:
                self.reset_velocity()
                return_statement.append("hole")
                break

            elif px_type == -3:
                c_number = int(val_info[3])
                if c_number < len(self.checkpoints) and not self.checkpoints[c_number].is_reached:
                    self.checkpoints[c_number].mark_reached()
                    start_point = np.array(self.checkpoints[c_number].get_center_coords(), dtype=float)
                    self.gamemap.set_start_point(*start_point)
                    return_statement.append("checkpoint")
            
            self.position += dstep
            Dpos -= dstep
            counter += 1

        #Hole mechanism
        val_info = self.gamemap.get_val_info(int(self.position[0]), int(self.position[1]))
        if (val_info[0] == -1): 
            vec_norm = val_info[2:0:-1]
            vec_tang = np.array([vec_norm[1], -vec_norm[0]])
            vec_proj_vel_tang = np.dot(vec_tang, self.velocity) / np.dot(vec_tang, vec_tang) * vec_tang
            self.velocity += vec_norm * 10
            self.velocity -= vec_proj_vel_tang * 0.3

        pos_difference = self.position - last_position
        for vector in normal_vectors:
            vec_norm = np.array(vector, dtype=float)
            vec_proj_difference = np.dot(vec_norm, pos_difference) / np.dot(vec_norm, vec_norm) * vec_norm
            if np.linalg.norm(vec_proj_difference) > 0.1:
                self.vibro_motor.vibrate(100)
        
        return return_statement


    def draw(self):
        self.canvas.coords(
            self.canvas_id,
            int(self.position[0]) - self.config.ball_radius + 1, 
            int(self.position[1]) - self.config.ball_radius + 1,
            int(self.position[0]) + self.config.ball_radius, 
            int(self.position[1]) + self.config.ball_radius
        )
    

    def reset_position(self):
        self.position = self.gamemap.get_start_point()


    def reset_velocity(self):
        self.velocity = np.array([0, 0], dtype=float)