"""
This file provides the template for designing the agent and environment.  The below hyperparameters must be assigned to a value for the algorithm to work properly.
"""

import numpy as np


def design_agent_and_env(FLAGS):
    # 有关分层网络的超参数设定

    FLAGS.total_steps = 50                                                     # 每个episode最大步数
    x = pow(FLAGS.total_steps, 1/FLAGS.layers)
    if x - int(x) == 0:
        FLAGS.time_scale = int(x)
    else:
        FLAGS.time_scale = int(x) + 1                                      # 下层步数比上层步数

    FLAGS.num_exploration_episodes = 100
    FLAGS.num_test_episodes = 100            # 测试episode数
    FLAGS.num_epochs = FLAGS.episodes // FLAGS.num_exploration_episodes

    # 有关模环境型的参数设定
    env_params = {}
    model_name = FLAGS.env + ".xml"
    env_params["model_path"] = "/Library/Frameworks/Python.framework/Versions/3.6/lib/python3.6/site-packages/gym/envs/robotics/assets/fetch/" + model_name

    env_params["has_object"] = True
    env_params["obj_range"] = 0.15

    env_params["max_actions"] = FLAGS.total_steps  # 一个episode的最大步数（动作数）
    env_params["timesteps_per_action"] = 20  # 动作执行间隔（间隔的是仿真最短时间）

    env_params["action_dim"] = 4  # action都是-1~1
    env_params["action_bounds"] = np.array([1, 1, 1, 1])  # low-level action bounds  1
    env_params["action_offset"] = np.array([0, 0, 0, 0])  # Assumes symmetric low-level action ranges

    env_params["initial_qpos"] = {
            'robot0:slide0': 0.405,
            'robot0:slide1': 0.48,
            'robot0:slide2': 0.0,
            'object0:joint': [1.25, 0.53, 0.4, 1., 0., 0., 0.],
        }
    env_params["block_gripper"] = True
    env_params["gripper_extra_height"] = 0
    env_params["project_state_to_end_goal"] = lambda state: state[3:6]    # 状态到目标goal的映射，只取位置信息

    distance_threshold = 0.05   # 5cm
    env_params["end_goal_thresholds"] = np.array([distance_threshold, distance_threshold, distance_threshold])

    env_params["project_state_to_subgoal"] = lambda state: state[3:6]      # 状态到子目标subgoal的映射，只取位置信息
    env_params["subgoal_thresholds"] = np.array([distance_threshold, distance_threshold, distance_threshold])

    env_params["end_goal_dim"] = 3
    env_params["subgoal_dim"] = 3

    env_params["target_offset"] = 0
    env_params["target_range"] = 0.15    # 随机生成子目标偏移初始位置范围
    env_params["target_in_the_air"] = False



    # agent参数
    agent_params = {}
    # Define percentage of actions that a subgoal level (i.e. level i > 0) will test subgoal actions
    agent_params["subgoal_test_perc"] = 0.3   # 0.3

    # Define subgoal penalty for missing subgoal.
    agent_params["subgoal_penalty"] = -FLAGS.time_scale     # 子目标未实现的惩罚
    agent_params["atomic_noise"] = [0.2 for i in range(4)]   # 0.1
    agent_params["subgoal_noise"] = [0.03 for i in range(3)]

    # Define number of episodes of transitions to be stored by each level of the hierarchy
    agent_params["episodes_to_store"] = 1000
    agent_params["update_times"] = 40
    agent_params["batch_size"] = 12

    return agent_params, env_params
