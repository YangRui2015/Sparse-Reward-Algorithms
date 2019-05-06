from tkinter import *
from tkinter import ttk
import time
import copy
import numpy as np
from mujoco_py import load_model_from_path, MjSim, MjViewer
import env_utils as utils


class Environment():

    def __init__(self, env_params, FLAGS):
        self.viewer = None
        self.model_path = env_params["model_path"]

        # 一些常数
        self.has_object = env_params["has_object"]
        self.initial_qpos = env_params["initial_qpos"]
        self.gripper_extra_height = env_params["gripper_extra_height"]
        self.block_gripper = env_params["block_gripper"]

        # Create Mujoco Simulation
        self.model = load_model_from_path(self.model_path)
        self.sim = MjSim(self.model)

        # Set dimensions and ranges of states, actions, and goals in order to configure actor/critic networks
        self._env_setup()
        self.initial_state = copy.deepcopy(self.sim.get_state())    # 得到仿真系统初始状态
        observation = utils.get_observation(self.sim, self.has_object)
        self.state_dim = len(observation)              # 其实是observation的维度
  
        self.action_dim = env_params["action_dim"]
        self.action_bounds = env_params["action_bounds"]
        self.action_offset = env_params["action_offset"]

        # goal的维数
        self.end_goal_dim = env_params["end_goal_dim"]
        self.subgoal_dim = env_params["subgoal_dim"]

        # Projection functions  状态-目标映射函数
        self.project_state_to_end_goal = env_params["project_state_to_end_goal"]
        self.project_state_to_subgoal = env_params["project_state_to_subgoal"]

        # subgoal范围
        self.target_offset = env_params["target_offset"]
        self.target_range = env_params["target_range"]
        self.target_in_the_air = env_params["target_in_the_air"]
        self.obj_range = env_params["obj_range"]
        self.object_place = None

        # End goal/subgoal thresholds
        self.end_goal_thresholds = env_params["end_goal_thresholds"]
        self.subgoal_thresholds = env_params["subgoal_thresholds"]

        self.subgoal_bounds_symmetric = self.initial_gripper_xpos
        self.subgoal_bounds_offset = np.array([self.target_range for _ in range(self.subgoal_dim)])

        self.max_actions = env_params["max_actions"]

        # Implement visualization if necessary
        self.visualize = FLAGS.show  # 是否显示
        if self.visualize:
            self.viewer = MjViewer(self.sim)
        self.num_frames_skip = env_params["timesteps_per_action"]
        # 输出env信息
        self.log_env_info()

    def _env_setup(self):               # 初始设置
        for name, value in self.initial_qpos.items():      # 这是设置几个轴的角度
            self.sim.data.set_joint_qpos(name, value)
        utils.reset_mocap_welds(self.sim)
        self.sim.forward()

        # Move end effector into position.
        gripper_target = np.array([-0.498, 0.005, -0.431 + self.gripper_extra_height]) + self.sim.data.get_site_xpos('robot0:grip')
        gripper_rotation = np.array([1., 0., 1., 0.])
        self.sim.data.set_mocap_pos('robot0:mocap', gripper_target)
        self.sim.data.set_mocap_quat('robot0:mocap', gripper_rotation)
        for _ in range(40):             # 移动杆需要时间
            self.sim.step()

        # Extract information for sampling goals.
        self.initial_gripper_xpos = self.sim.data.get_site_xpos('robot0:grip').copy()   # 到固定位置才是initial位置
        if self.has_object:
            self.height_offset = self.sim.data.get_site_xpos('object0')[2]

    def get_state(self):
        return utils.get_observation(self.sim, has_object=self.has_object)

    def execute_action(self, action):
        assert action.shape[0] == self.action_dim
        action = action.copy()  # ensure that we don't change the action outside of this scope
        pos_ctrl, gripper_ctrl = action[:3], action[3]

        pos_ctrl *= 0.05  # limit maximum change in position
        rot_ctrl = [1., 0., 1., 0.]         # fixed rotation of the end effector, expressed as a quaternion
        gripper_ctrl = np.array([gripper_ctrl, gripper_ctrl])
        assert gripper_ctrl.shape == (2,)
        if self.block_gripper:
            gripper_ctrl = np.zeros_like(gripper_ctrl)
        action = np.concatenate([pos_ctrl, rot_ctrl, gripper_ctrl])

        # Apply action to simulation.
        utils.ctrl_set_action(self.sim, action)
        utils.mocap_set_action(self.sim, action)

        for _ in range(self.num_frames_skip):   # 每隔多少步执行一个动作
            self.sim.step()
            if self.visualize:
                self.viewer.render()

        return self.get_state()

    # 随机选择目标
    def get_next_goal(self):
        if self.has_object:
            if self.object_place is not None:
                flag = 1
                while flag:
                    goal = self.initial_gripper_xpos[:3] + np.random.uniform(-self.target_range, self.target_range, size=3)
                    goal += self.target_offset
                    goal[2] = self.height_offset
                    if self.target_in_the_air and np.random.uniform() < 0.5:
                        goal[2] += np.random.uniform(0, 0.45)

                    flag = 0
                    # for i in range(len(goal)):
                    #     if abs(goal[i] - self.object_place[i]) > self.end_goal_thresholds[i]:
                    #         flag = 0
                    #         break

            else:
                goal = self.initial_gripper_xpos[:3] + np.random.uniform(-self.target_range, self.target_range, size=3)
                goal += self.target_offset
                goal[2] = self.height_offset        # height_offset是object的初始高度
                if self.target_in_the_air and np.random.uniform() < 0.5:
                    goal[2] += np.random.uniform(0, 0.45)
        else:
            goal = self.initial_gripper_xpos + np.random.uniform(-self.target_range, self.target_range, size=3)

        return goal.copy()

    # Reset simulation to state within initial state specified by user
    def reset_sim(self):
        self.sim.set_state(self.initial_state)

        # Randomize start position of object.
        if self.has_object:
            object_xpos = self.initial_gripper_xpos[:2]
            while np.linalg.norm(object_xpos - self.initial_gripper_xpos[:2]) < 0.1:
                object_xpos = self.initial_gripper_xpos[:2] + np.random.uniform(-self.obj_range, self.obj_range, size=2)
            object_qpos = self.sim.data.get_joint_qpos('object0:joint')
            assert object_qpos.shape == (7,)
            object_qpos[:2] = object_xpos
            self.sim.data.set_joint_qpos('object0:joint', object_qpos)
            self.object_place = object_qpos[0:3]

        self.sim.forward()
        self.goal = self.get_next_goal()
        self._render_target()   # 显示goal

        # Return state
        return self.get_state()

    def _render_target(self):
        sites_offset = (self.sim.data.site_xpos - self.sim.model.site_pos).copy()
        site_id = self.sim.model.site_name2id('target0')
        self.sim.model.site_pos[site_id] = self.goal - sites_offset[0]
        self.sim.forward()

    def log_env_info(self):
        print("Environment information: {}\n".format(self.model_path))
        print("Initial state:{}\n".format(self.initial_state))
        print("State dim:{}\n End_goal dim:{}\n Subgoal dim:{}\n".format(self.state_dim, self.end_goal_dim, self.subgoal_dim))
        print("Action dim:{}\n Action bounds:{}\n Action offset:{}\n".format(self.action_dim, self.action_bounds, self.action_offset))
        print("Target range :{}\n".format(self.target_range))
        print("End goal thresholds:{}\n Subgoal_thresholds:{}\n".format(self.end_goal_thresholds, self.subgoal_thresholds))
        print("Max actions:{}\n Visualize:{}\n Num_frames_skip:{}\n\n".format(self.max_actions, self.visualize, self.num_frames_skip))




