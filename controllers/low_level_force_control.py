import mujoco
import cv2
import numpy as np
import yaml
import pickle
import pathlib
import pickle
import mujoco.viewer
import time
from mim_robots.robot_list import MiM_Robots
from mim_robots.robot_loader import load_mujoco_model, get_robot_list

class Low_level_Force_Control:
    def __init__(self):
        self.kp = 10
        self.kv = 5
        self.damping = 0.0
        self.q_des = np.array([0, 0, 0, 0, 0, 0, 0])
        self.qv_des = np.array([0, 0, 0, 0, 0, 0, 0])
        self.u0 = np.array([0, 0, 0, 0, 0, 0, 0])
        self.umax = 200*np.ones(7)
        self.umin = -200*np.ones(7)

    def policy(self, data):
        """ """
        q = data.qpos
        qv = data.qvel
        u = (
            self.kp * (self.q_des[:7] - q[:7])
            + self.kv * (self.qv_des[:7] - qv[:7])
            - self.damping * qv[:7]
        )
        u = np.clip(u, self.umin, self.umax)
        data.ctrl[0] = u[0]
        data.ctrl[1] = u[1]
        data.ctrl[2] = u[2]
        data.ctrl[3] = u[3]
        data.ctrl[4] = u[4]
        data.ctrl[5] = u[5]
        data.ctrl[6] = u[6]

    def control_u0(self, data):
        data.ctrl[0] = self.u0[0]
        data.ctrl[1] = self.u0[1]
        data.ctrl[2] = self.u0[2]
        data.ctrl[3] = self.u0[3]
        data.ctrl[4] = self.u0[4]
        data.ctrl[5] = self.u0[5]
        data.ctrl[6] = self.u0[6]
