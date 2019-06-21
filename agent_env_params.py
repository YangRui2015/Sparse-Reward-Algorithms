

def design_agent_and_env(FLAGS):

    # 有关模环境型的参数设定
    env_params = {}
    if FLAGS.env == "reach":
        env_params["env_name"] = "FetchReach-v1"
        env_params["has_object"] = False
        FLAGS.total_steps = 20
    elif FLAGS.env == "push":
        env_params["env_name"] = "FetchPush-v1"
        env_params["has_object"] = True
        FLAGS.total_steps = 50
    elif FLAGS.env == "handreach":
        env_params["env_name"] = "HandReach-v0"
        FLAGS.total_steps = 30
    else:
        raise TypeError("No such environment.")

    x = pow(FLAGS.total_steps, 1/FLAGS.layers)
    if x - int(x) == 0:
        FLAGS.time_scale = int(x)
    else:
        FLAGS.time_scale = int(x) + 1                                      # 下层步数比上层步数

    FLAGS.num_exploration_episodes = 100
    FLAGS.num_test_episodes = 100            # 测试episode数
    FLAGS.num_epochs = FLAGS.episodes // FLAGS.num_exploration_episodes

    env_params["obj_range"] = 0.15
    env_params["max_actions"] = FLAGS.total_steps  # 一个episode的最大步数（动作数）

    if FLAGS.env == "handreach":
        distance_threshold = 0.01
    else:
        distance_threshold = 0.05   # 5cm
    env_params["end_goal_thresholds"] = distance_threshold
    env_params["subgoal_thresholds"] = distance_threshold

    # agent参数
    agent_params = {}
    agent_params["subgoal_test_perc"] = 0.3

    # Define subgoal penalty for missing subgoal.
    agent_params["subgoal_penalty"] = -FLAGS.time_scale     # 子目标未实现的惩罚
    agent_params["atomic_noise"] = 0.1
    agent_params["subgoal_noise"] = 0.03
    agent_params["epsilon"] = 0.1          # choose random action

    # Define number of episodes of transitions to be stored by each level of the hierarchy
    agent_params["episodes_to_store"] = 1000
    if FLAGS.rnd:
        agent_params["update_times"] = 4    # 2
        agent_params["batch_size"] = 512
    else:
        agent_params["update_times"] = 40    #40
        agent_params["batch_size"] = 64     # 32   12

    return agent_params, env_params
