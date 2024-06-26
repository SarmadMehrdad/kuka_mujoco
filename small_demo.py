import mujoco
import cv2
import numpy as np
import yaml
import pickle
import pathlib
import pickle
import mujoco.viewer
import time


# load the model.xml
xml_path = "model.xml"

model = mujoco.MjModel.from_xml_path(xml_path)

data = mujoco.MjData(model)
renderer = mujoco.Renderer(model)

mujoco.mj_forward(model, data)
renderer.update_scene(data)

import argparse

argsp = argparse.ArgumentParser()

argsp.add_argument("--simulate", action="store_true")
argsp.add_argument("--time", type=float,default=5)
argsp.add_argument("--data_file", type=str, default="")
argsp.add_argument("--vis", type=int, default=0)

args = argsp.parse_args()
add_capsules = args.vis > 0

if not args.simulate and args.data_file == "":
    raise ValueError("You need to use --simulate or do provide a --data_file")

if args.vis > 1:
    while True:
        # cv2 uses BGR, mujoco uses RGB
        cv2.imshow("image", cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR))
        # close the window when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cv2.destroyAllWindows()

duration = 5  # (seconds)
framerate = 60  # (Hz)

if args.vis > 1:
    # simulate and visualize
    frames = []
    mujoco.mj_resetData(model, data)  # Reset state and time.
    while data.time < duration:
        mujoco.mj_step(model, data)
        if len(frames) < data.time * framerate:
            renderer.update_scene(data)
            pixels = renderer.render()
            frames.append(pixels)

    for frame in frames:
        cv2.imshow("image", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        time.sleep(1 / framerate)

    cv2.destroyAllWindows()


m = model
d = data


print("positions", data.qpos)
import sys


print("geoms", [model.geom(i).name for i in range(model.ngeom)])
print("bodies", [model.body(i).name for i in range(model.nbody)])

print("geoms", [model.geom(i).id for i in range(model.ngeom)])
print("bodies", [model.body(i).id for i in range(model.nbody)])


names = [model.geom(i).name for i in range(model.ngeom)]


for name in names:
    if name != "":
        print(name, data.geom(name).xpos)


def add_visual_capsule(scene, point1, point2, radius, rgba):
    """Adds one capsule to an mjvScene."""
    if scene.ngeom >= scene.maxgeom:
        raise ValueError("scene is full")
    scene.ngeom += 1  # increment ngeom
    # initialise a new capsule, add it to the scene using mjv_makeConnector
    mujoco.mjv_initGeom(
        scene.geoms[scene.ngeom - 1],
        mujoco.mjtGeom.mjGEOM_CAPSULE,
        np.zeros(3),
        np.zeros(3),
        np.zeros(9),
        rgba.astype(np.float32),
    )
    mujoco.mjv_makeConnector(
        scene.geoms[scene.ngeom - 1],
        mujoco.mjtGeom.mjGEOM_CAPSULE,
        radius,
        point1[0],
        point1[1],
        point1[2],
        point2[0],
        point2[1],
        point2[2],
    )


center = np.array([0, 0, 0])


class Low_level_Force_Control:

    def __init__(self):
        self.kp = 10
        # self.kv = 5
        # self.damping = .1
        self.kv = 0  # lets try kv to zero
        self.damping = 5.0
        self.xdes = np.array([0, 0, 0])
        self.vdes = np.array([0, 0, 0])
        self.u0 = np.array([0, 0])
        self.umax = np.array([1, 1])
        self.umin = np.array([-1, -1])

    def policy(self, data):
        """ """
        x = data.body("ball").xpos
        v = data.body("ball").cvel[3:]


        u = (
            self.kp * (self.xdes[:2] - x[:2])
            + self.kv * (self.vdes[:2] - v[:2])
            - self.damping * v[:2]
        )
        u = np.clip(u, self.umin, self.umax)
        data.ctrl[0] = u[0]
        data.ctrl[1] = u[1]

    def control_u0(self, data):
        """ """
        data.ctrl[0] = self.u0[0]
        data.ctrl[1] = self.u0[1]


class Position_Planner:

    def __init__(self, data):
        self.low_level_force_control = Low_level_Force_Control()

        self.low_level_force_control.xdes = np.array(data.body("ball").xpos[:3])
        self.low_level_force_control.vdes = np.zeros(3)

        self.plan = []  # sequence of positions and velocities
        self.reference_max_vel = 0.2
        self.mode = "to_rand"
        self.p_lb = np.array([-0.3, -0.3, 0.05])
        self.p_ub = np.array([0.3, 0.3, 0.05])
        self.goal = self.p_lb + np.random.rand(3) * (self.p_ub - self.p_lb)
        self.start = data.body("ball").xpos
        self.scene = None  # used to add visual capsules

        self.p_lb_small = self.p_lb + 0.1 * (self.p_ub - self.p_lb)
        self.p_ub_small = self.p_ub - 0.1 * (self.p_ub - self.p_lb)
        self.last_call_time = 0

        self.dif = self.goal - self.start
        time = np.linalg.norm(self.dif) / self.reference_max_vel

        self.hl_time = 0.1  # in seconds
        self.ll_time = m.opt.timestep
        times = np.linspace(0, time, int(time / self.hl_time))
        self.plan = [self.start + t * self.dif / time for t in times]
        self.times = times


        self.counter = 0

    def get_data(self):
        return {
            "xdes": self.low_level_force_control.xdes,
            "vdes": self.low_level_force_control.vdes,
            "mode": self.mode,
        }


    def policy(self, data):

        # genereate a xdes and vdes
        if data.time - self.last_call_time < self.hl_time:
            # i don't do anything, only call again the low level force control
            self.low_level_force_control.policy(data)

        else:
            self.last_call_time = data.time
            if self.counter < len(self.plan):
                xdes = self.plan[self.counter]
                vdes = (
                    np.zeros(3)
                    if self.counter == len(self.plan) - 1
                    else (self.plan[self.counter + 1] - self.plan[self.counter])
                    / self.hl_time
                )
                self.counter += 1
                # print("counter is ", self.counter)
                # print("updating xdes and vdes")
                # print("xdes", xdes)
                # print("vdes",vdes)
                if (
                    np.linalg.norm(self.low_level_force_control.xdes[:2] - xdes[:2])
                    > 0.1
                ):
                    print("Warning: very big change in xdes")
                self.low_level_force_control.xdes = np.copy(xdes)
                self.low_level_force_control.vdes = np.copy(vdes)

                self.dist_fail_controller = 10

                if (
                    np.linalg.norm(data.body("ball").xpos - xdes)
                    > self.dist_fail_controller
                ):
                    print("PRINT: failed to reach the desired position!")
                    print("lets do a random motion instead")

                    self.mode = "to_rand"
                    self.goal = self.p_lb + np.random.rand(3) * (self.p_ub - self.p_lb)

                    if add_capsules:
                        add_visual_capsule(
                            self.scene,
                            self.goal,
                            self.goal + np.array([0, 0, 0.1]),
                            0.05,
                            np.array([0, 0, 1, 0.1]),
                        )

                    self.start = data.body("ball").xpos
                    self.dif = self.goal - self.start
                    time = np.linalg.norm(self.dif) / self.reference_max_vel
                    steps = int(time / self.hl_time)
                    self.plan = [
                        self.start + t * self.dif / steps for t in range(steps)
                    ]
                    print(f"current plan ({len(self.plan)}) ) is ", self.plan)
                    self.counter = 0
                    self.mode = "to_rand"

                    if add_capsules:
                        if len(self.plan) > 1000:
                            for i in range(0, len(self.plan), 1000):
                                add_visual_capsule(
                                    self.scene,
                                    self.plan[i],
                                    self.plan[i] + np.array([0, 0, 0.1]),
                                    0.05,
                                    np.array([0, 0, 1, 0.1]),
                                )

                else:
                    self.low_level_force_control.policy(data)
            else:
                self.counter = 0

                xdes = data.body("ball").xpos[:2]
                vdes = np.zeros(2)

                if np.linalg.norm(self.low_level_force_control.xdes[:2] - xdes) > 0.1:
                    print("Warning: very big change in xdes")
                self.low_level_force_control.xdes = np.copy(xdes)
                self.low_level_force_control.vdes = np.copy(vdes)
                # self.low_level_force_control.control_u0(data)
                # generate a new plan
                # change the mode
                # compute distance between cube and ball

                self.threshold_rand_vs_obj = 0.1

                cube = np.array(data.body("cube").xpos)
                if np.any(cube[:2] < self.p_lb_small[:2]) or np.any(
                    cube[:2] > self.p_ub_small[:2]
                ):

                    print("cube is outside!")
                    rel_cube = cube - center

                    # I will generate 3 goal

                    goal_1 = np.zeros(3)

                    goal_1[0] = rel_cube[1]
                    goal_1[1] = -1 * rel_cube[0]

                    a = 2 * np.pi / 3  # lets do "a" degrees
                    R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
                    goal_1[:2] = 0.8 * R @ goal_1[:2]

                    goal_2 = 1.8 * rel_cube
                    goal_3 = center

                    # take an orthogonal_direction:
                    self.big_goals = [goal_1, goal_2, goal_3]

                    if add_capsules:
                        for g in self.big_goals:
                            add_visual_capsule(
                                self.scene,
                                g,
                                g + np.array([0, 0, 0.1]),
                                0.05,
                                np.array([1, 1, 0, 0.1]),
                            )

                    self.plan = []
                    for i in range(len(self.big_goals)):

                        if i == 0:
                            start = data.body("ball").xpos
                        else:
                            start = self.big_goals[i - 1]
                        dif = self.big_goals[i] - start

                        time = np.linalg.norm(self.dif) / self.reference_max_vel

                        num_steps = int(time / self.hl_time)
                        print(f"num_steps {i}", num_steps)
                        self.plan += [
                            start + t * dif / num_steps for t in range(num_steps)
                        ]

                    self.mode = "recovery"
                    print("new high level mode", self.mode)
                    print("len(self.plan)", len(self.plan))

                    if add_capsules:
                        if len(self.plan) > 1000:
                            for i in range(0, len(self.plan), 1000):
                                add_visual_capsule(
                                    self.scene,
                                    self.plan[i],
                                    self.plan[i] + np.array([0, 0, 0.1]),
                                    0.05,
                                    np.array([1, 1, 0, 0.1]),
                                )

                else:
                    if (
                        np.linalg.norm(data.body("cube").xpos - data.body("ball").xpos)
                        < self.threshold_rand_vs_obj
                    ):
                        self.mode = "to_rand"
                    else:
                        self.mode = "to_obj"

                    print("new high level mode", self.mode)
                    if self.mode == "to_rand":
                        self.goal = self.p_lb + np.random.rand(3) * (
                            self.p_ub - self.p_lb
                        )

                        if add_capsules:
                            add_visual_capsule(
                                self.scene,
                                self.goal,
                                self.goal + np.array([0, 0, 0.1]),
                                0.05,
                                np.array([0, 0, 1, 0.1]),
                            )

                    elif self.mode == "to_obj":
                        cube = np.array(data.body("cube").xpos)
                        ball = np.array(data.body("ball").xpos)
                        diff_weights = np.array([1, 1, 0])
                        self.penetration = 0.2
                        cube_to_ball = diff_weights * np.array(cube - ball)
                        self.goal = (
                            cube
                            + self.penetration
                            * cube_to_ball
                            / np.linalg.norm(cube_to_ball)
                        )
                        self.goal = np.clip(self.goal, self.p_lb, self.p_ub)

                        if add_capsules:
                            add_visual_capsule(
                                self.scene,
                                self.goal,
                                self.goal + np.array([0, 0, 0.1]),
                                0.05,
                                np.array([1, 0, 0, 0.1]),
                            )

                    self.start = data.body("ball").xpos

                    self.dif = self.goal - self.start

                    time = np.linalg.norm(self.dif) / self.reference_max_vel
                    times = np.linspace(0, time, int(time / self.hl_time))

                    self.plan = [self.start + t * self.dif / time for t in times]

                    if add_capsules:
                        if len(self.plan) > 1000:
                            for i in range(0, len(self.plan), 1000):
                                add_visual_capsule(
                                    self.scene,
                                    self.plan[i],
                                    self.plan[i] + np.array([0, 0, 0.1]),
                                    0.05,
                                    np.array([1, 0, 0, 0.1]),
                                )

id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "ball")
vel = np.zeros(6)

mujoco.mj_forward(model, data)
data.time = 0

simulation_time = args.time
pos_array = []
times_array = []
controls_array = []
des_array = []



times = []
positions = []
speeds = []
offset = model.jnt_axis[0] / 16  # offset along the joint axis

point1 = np.array([1, 0, 1])
point2 = np.array([0, 1, 1])


def get_state_summary(data):

    geom1_id = data.geom("gcube_0").id
    geom2_id = data.geom("gball_0").id

    case_a = data.geom("gcase_a").id
    case_b = data.geom("gcase_b").id
    case_c = data.geom("gcase_c").id
    case_d = data.geom("gcase_d").id

    case_ids = [case_a, case_b, case_c, case_d]

    # get contact status
    objects_in_contact = False
    contact_wall_cube = False
    for i in range(data.ncon):
        con = data.contact[i]
        # help(con)
        if (con.geom1 == geom1_id and con.geom2 == geom2_id) or (
            con.geom2 == geom1_id and con.geom1 == geom2_id
        ):
            contact_pos = con.pos
            objects_in_contact = True
            # break

        if con.geom1 in case_ids and con.geom2 == geom2_id:
            contact_wall_cube = True
        if con.geom2 in case_ids and con.geom1 == geom2_id:
            contact_wall_cube = True

    # print("objects_in_contact", objects_in_contact)

    D = {
        "q": np.array(data.qpos),
        "qvel": np.array(data.qvel),
        "time": data.time,
        "cube": np.array(data.body("cube").xpos),
        "ball": np.array(data.body("ball").xpos),
        "cube_vel": np.array(data.body("cube").cvel[3:]),
        "ball_vel": np.array(data.body("ball").cvel[3:]),
        "cube_r": np.array(data.body("cube").xquat),
        "ball_r": np.array(data.body("ball").xquat),
        "cube_w": np.array(data.body("cube").cvel[:3]),
        "ball_w": np.array(data.body("ball").cvel[:3]),
        "contact_ball_cube": np.array(objects_in_contact),
        "contact_cube_wall": np.array(contact_wall_cube),
        "u": np.array(data.ctrl),
    }

    return D


all_data = []

import pathlib

if args.vis > 1:
    while True:
        # cv2 uses BGR, mujoco uses RGB
        cv2.imshow("image", cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR))
        # close the window when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cv2.destroyAllWindows()


time_str = time.strftime("%Y-%m-%d--%H-%M-%S", time.localtime())
image_id = 0
if args.simulate:

    controller = Position_Planner(data)
    goal = controller.goal
    start = time.time()

    sim_time = [0]

    def step():

        # controller.policy(data)
        controller.policy(data)
        mujoco.mj_step(model, data)

        _data = get_state_summary(data)
        _data_controller = controller.get_data()
        _data.update(_data_controller)

        # add dicts _data and _data_controller
        all_data.append(_data)

        sim_time[0] += m.opt.timestep
        times_array.append(data.time)

    if args.vis >= 1:

        with mujoco.viewer.launch_passive(
            m, d, show_left_ui=False, show_right_ui=False
        ) as viewer:

            controller.scene = viewer._user_scn
            last_render = time.time()
            while viewer.is_running() and data.time < simulation_time:

                step_start = time.time()
                step()
                viewer.sync()
                time_until_next_step = m.opt.timestep - (time.time() - step_start)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)

                # lets save some images  to the disk
                current_time = time.time()

                # if current_time - last_render > 1/24:
                #     last_render = current_time
                #     renderer.update_scene(data)
                #     frame = renderer.render()
                #     import matplotlib.pyplot as plt
                #
                #     # while True:
                #     #     # cv2 uses BGR, mujoco uses RGB
                #     #     cv2.imshow(
                #     #         "image", cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR)
                #     #     )
                #     #     # close the window when 'q' is pressed
                #     #     if cv2.waitKey(1) & 0xFF == ord("q"):
                #     #         break
                #     # cv2.destroyAllWindows()
                #
                #
                #     # continue here!!
                #     file_out = f"images/frame_{time_str}/{image_id:05d}.png"
                #     image_id += 1
                #     pathlib.Path(file_out).parent.mkdir(parents=True, exist_ok=True)
                #     print("saving image to ", file_out)
                #     frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                #     cv2.imwrite(file_out, frame)

    else:
        while data.time < simulation_time:
            step()

    now = time.time()

    data_out = f"data/main2_data_{time_str}.dat"
    pathlib.Path(data_out).parent.mkdir(parents=True, exist_ok=True)
    print("saving data to ", data_out)
    with open(data_out, "wb") as f:
        pickle.dump(all_data, f)
    # data_out_yml = f"data/main2_data_{time_str}.yaml"
    # print("saving data to ", data_out_yml)
    # with open(data_out_yml, "w") as f:
    #     yaml.dump(all_data, f)

    # lets reply the simulation

    # create another viewer
    replay_visualization = False
    if replay_visualization:
        print("replaying the simulation")

        data.ctrl[0] = 0.0
        data.ctrl[1] = 0.0

        data.qpos = all_data[0]["q"]
        data.qvel = all_data[0]["qvel"]

        mujoco.mj_step(model, data)
        mujoco.mj_forward(model, data)

        with mujoco.viewer.launch_passive(
            model, data, show_left_ui=False, show_right_ui=False
        ) as viewer:
            last_time = 0
            for i in range(len(all_data)):
                if all_data[i]["time"] - last_time > 1 / 30:
                    last_time = all_data[i]["time"]
                    data.qpos = all_data[i]["q"]
                    data.qvel = all_data[i]["qvel"]
                    mujoco.mj_step(model, data)
                    mujoco.mj_forward(model, data)
                    viewer.sync()
                    time.sleep(1 / 30)

