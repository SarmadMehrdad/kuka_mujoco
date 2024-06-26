import mujoco
import cv2
import numpy as np
import yaml
import pickle
import pathlib
import pickle
import mujoco.viewer
import time
from controllers.low_level_force_control import Low_level_Force_Control
from mim_robots.robot_list import MiM_Robots
from mim_robots.robot_loader import load_mujoco_model, get_robot_list

class Position_Planner:

    def __init__(self, model, data):
        self.low_level_force_control = Low_level_Force_Control()

        self.low_level_force_control.q_des = np.array(data.qpos)
        self.low_level_force_control.qv_des = np.zeros(7)

        self.plan = []  # sequence of positions and velocities
        self.reference_max_vel = 0.2
        self.p_lb = -1*np.ones(7)
        self.p_ub = np.ones(7)
        self.goal = self.p_lb + np.random.rand(7) * (self.p_ub - self.p_lb)
        self.start = data.qpos

        self.p_lb_small = self.p_lb + 0.1 * (self.p_ub - self.p_lb)
        self.p_ub_small = self.p_ub - 0.1 * (self.p_ub - self.p_lb)
        self.last_call_time = 0

        self.dif = self.goal - self.start
        time = np.linalg.norm(self.dif) / self.reference_max_vel

        self.hl_time = 0.1  # in seconds
        self.ll_time = model.opt.timestep
        times = np.linspace(0, time, int(time / self.hl_time))
        self.plan = [self.start + t * self.dif / time for t in times]
        self.times = times
        self.counter = 0

    def get_data(self):
        return {
            "q_des": self.low_level_force_control.q_des,
            "qv_des": self.low_level_force_control.qv_des,
        }


    def policy(self, data):

        # genereate a xdes and vdes
        if data.time - self.last_call_time < self.hl_time:
            self.low_level_force_control.policy(data)