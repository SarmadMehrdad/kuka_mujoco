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

RobotInfo = MiM_Robots["iiwa"]
model = load_mujoco_model("iiwa")
data = mujoco.MjData(model)
renderer = mujoco.Renderer(model)
mujoco.mj_forward(model, data)
renderer.update_scene(data)

duration = 5  # (seconds)
framerate = 60  # (Hz)

m = model
d = data

print("geoms", [model.geom(i).name for i in range(model.ngeom)])
print("bodies", [model.body(i).name for i in range(model.nbody)])

print("geoms", [model.geom(i).id for i in range(model.ngeom)])
print("bodies", [model.body(i).id for i in range(model.nbody)])

names = [model.body(i).name for i in range(model.nbody)]
for name in names:
    x = data.body(name).xpos # Body xyz
    v = data.body(name).cvel[:3] # Body velocity in cartesian
    q = data.qpos # Joint values
    qv = data.qvel # Joint velocities

print('Joint Values: ', q)
# How to get Jacobian from Mujoco
jacp = np.zeros((3,7))
jacr = np.zeros((3,7))
mujoco.mj_jac(m, d, jacp, jacr, np.array([0, 0, 0]), 6) # This line updates the content of jacp (positional jacobian) and jacr (rotational jacobian)
J = np.concatenate((jacp,jacr),axis=0) # Full jacobian matrix
print("Jacobian:\n",J)

# How to get joint axes
print("Joint Axes: \n",model.jnt_axis)

# Inertia Matrix
M = np.zeros(shape=(7,7))
mujoco.mj_fullM(model, M, data.qM)
print("Inertia Matrix:\n" , M)

# Bias force:  Coriolis/Centrifugal/Gravitational ==> C(q,v)*q_dot
C = np.zeros(shape=(7,1))
mujoco.mj_rne(model, data, 0, C) # Third argument determines if q_dot_dot is considered 0 or not. 0 for Yes, 1 for No.
print("C Vector:\n", C)
print(C.shape)
