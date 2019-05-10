import numpy as np
import gym


class Environment():

    def __init__(self, env_params, FLAGS):
        self.env = gym.make(env_params["env_name"])
        self.set_seed(FLAGS.seed)

        observation = self.reset_sim()
        self.state_dim = len(observation)  # 其实是observation的维度

        self.action_dim = self.env.env.action_space.shape[0]
        self.action_bounds = 1
        self.action_offset = 0

        # goal的维数
        self.end_goal_dim = len(self.env.env._get_obs()["desired_goal"])
        self.subgoal_dim = self.end_goal_dim

        # Projection functions  状态-目标映射函数
        if FLAGS.env == "handreach":
            self.project_state_to_end_goal = lambda state: state[-15:]
            self.project_state_to_subgoal = lambda state: state[-15:]
        else:
            if self.env.env.has_object:
                self.project_state_to_end_goal = lambda state: state[3:6]
                self.project_state_to_subgoal = lambda state: state[3:6]
            else:
                self.project_state_to_end_goal = lambda state: state[0:3]
                self.project_state_to_subgoal = lambda state: state[0:3]

        # End goal/subgoal thresholds
        self.end_goal_thresholds = env_params["end_goal_thresholds"]
        self.subgoal_thresholds = env_params["subgoal_thresholds"]

        if FLAGS.env == "handreach":
            self.subgoal_bounds_symmetric = self.env.env.initial_goal
            self.subgoal_bounds_offset = np.array([0.05 for _ in range(self.subgoal_dim)])

        else:
            self.subgoal_bounds_symmetric = self.env.env.initial_gripper_xpos
            self.subgoal_bounds_offset = np.array([self.env.env.target_range for _ in range(self.subgoal_dim)])

        self.max_actions = env_params["max_actions"]

        self.visualize = FLAGS.show  # 是否显示

    def set_seed(self, seed):
        self.env.seed(seed)

    def get_state(self):
        return self.env.env._get_obs()["observation"]

    def execute_action(self, action):
        self.env.step(action)
        if self.visualize:
            self.env.render()

        return self.get_state()

    # Reset simulation to state within initial state specified by user
    def reset_sim(self):
        observation = self.env.reset()
        self.goal = observation["desired_goal"]

        return self.get_state()


