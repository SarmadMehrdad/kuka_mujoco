import mujoco
import cv2
import numpy as np
import yaml
import pickle
import pathlib
import pickle
import mujoco.viewer
from operator import index
import pinocchio as pin
import numpy as np
from mim_robots.robot_loader import load_pinocchio_wrapper, get_robot_list
from numpy.linalg import norm, solve
import time
from mim_robots.robot_loader import load_mujoco_model, get_robot_list

# print(get_robot_list())

robot = load_pinocchio_wrapper("iiwa")
pin_model = robot.model
pin_collision_model = robot.collision_model
pin_visual_model = robot.visual_model
pin_data = pin_model.createData()

mj_model = load_mujoco_model("iiwa")
mj_data = mujoco.MjData(mj_model)

q = pin.randomConfiguration(pin_model)
print('q: %s' % q.T)

print("Center of Mass: ", robot.com(q))

print(robot.model)
idx = robot.index('A7')

placement = robot.data.oMi[idx].copy()
print("Placement of joint {}: \n {}".format(idx, placement))

for i in range(7):
    print(robot.visual_model.geometryObjects[i].name)

J = pin.computeFrameJacobian(robot.model, robot.data, q, idx)
print("Jacobian: \n", J)

eps    = 1e-4
IT_MAX = 1000
DT     = 1e0
damp   = 1e-6
q = pin.neutral(pin_model)
i = 1

def go_to(model, data, idx, q0, R, P, DT, eps, damp):
    q = q0
    i = 1
    while True:
        oMdes = pin.SE3(R, P)
        pin.forwardKinematics(model,data,q)
        dMi = oMdes.actInv(data.oMi[idx])
        err = pin.log(dMi).vector
        if norm(err) < eps:
            success = True
            break
        if i >= IT_MAX:
            success = False
            break
        J = pin.computeJointJacobian(model,data,q,idx)
        v = - J.T.dot(solve(J.dot(J.T) + damp * np.eye(6), err))
        q = pin.integrate(model,q,v*DT)
        # if not i % 10:
            # print('%d: error = %s' % (i, err.T))
        # i += 1
    return model,data, q

q0 = pin.neutral(pin_model)
R = np.array([[np.cos(30), 0, -np.sin(30)],[0, 1, 0], [np.sin(30), 0, np.cos(30)]])
P = np.array([0.6, 0.0, 0.7])
pin_model, pin_data, q = go_to(pin_model, pin_data, idx, q0, R, P, DT, eps, damp)

t0 = time.time()
T = 1000
t = time.time() - t0

with mujoco.viewer.launch_passive(mj_model, mj_data, show_left_ui=True, show_right_ui=True) as viewer:
    while viewer.is_running() and t < 10.:
        R = np.array([[np.cos(30), 0, -np.sin(30)],[0, 1, 0], [np.sin(30), 0, np.cos(30)]])
        P = np.array([0.6 + 0.2*np.cos(t*np.pi),  0.2*np.sin(t*np.pi), 0.5 -  0.2*np.sin(t*np.pi)])
        pin_model, pin_data, q = go_to(pin_model, pin_data, idx, q, R, P, DT, eps, damp)
        mj_data.qpos = q
        print(mj_data.qpos)
        mujoco.mj_step(mj_model, mj_data)
        viewer.sync()
        t = time.time() - t0


