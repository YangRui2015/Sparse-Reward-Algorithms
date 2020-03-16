import numpy as np
import gym


class Environment():
    def __init__(self, env_params, FLAGS):
        self.env = gym.make(env_params["env_name"])
        self.set_seed(FLAGS.seed)

        # target range / obj range
        self.target_range = self.env.env.target_range = env_params["target_range"]
        self.obj_range = self.env.env.obj_range = env_params["obj_range"]

        # End goal/subgoal thresholds
        self.end_goal_thresholds = env_params["end_goal_thresholds"]
        self.subgoal_thresholds = env_params["subgoal_thresholds"]
        self.subgoal_dim = self.end_goal_dim = len(self.env.env._get_obs()["desired_goal"])

        observation = self.reset_sim()
        self.state_dim = len(observation)  
        self.action_dim = self.env.env.action_space.shape[0]
        self.action_bounds = 1
        self.action_offset = 0
        self.max_actions = env_params["max_actions"]

        # Projection functions, from states to goals  
        if self.env.env.has_object:
            self.project_state_to_end_goal = lambda state: state[3:6]
            self.project_state_to_subgoal = lambda state: state[3:6]
        else:
            self.project_state_to_end_goal = lambda state: state[0:3]
            self.project_state_to_subgoal = lambda state: state[0:3]

        self.subgoal_bounds_symmetric = self.env.env.initial_gripper_xpos
        self.subgoal_bounds_offset = np.array([self.target_range for _ in range(self.subgoal_dim)])
        self.visualize = FLAGS.show  
    
    # for curriculum learning
    def set_threshold(self, thre):  
        self.end_goal_thresholds = thre
        self.subgoal_thresholds = thre
    
    def set_goal_range(self, r):
        self.target_range = r

    def sample_goal(self):
        goal = self.env.env.initial_gripper_xpos + np.random.uniform(-self.target_range, self.target_range, size=3)
        return goal

    def set_seed(self, seed):
        self.env.seed(seed)

    def get_state(self):
        return self.env.env._get_obs()["observation"]

    def execute_action(self, action):
        self.env.step(action)
        if self.visualize:
            self.env.render()
        return self.get_state()

    # Reset simulation to state within initial state 
    def reset_sim(self):
        observation = self.env.reset()
        self.goal = self.sample_goal()
        self.env.env.goal = self.goal
        return self.get_state()


