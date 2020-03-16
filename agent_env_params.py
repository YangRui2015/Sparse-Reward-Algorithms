import numpy as np 

def design_agent_and_env(FLAGS):

    # environment params
    env_params = {}
    if FLAGS.env == "reach":
        env_params["env_name"] = "FetchReach-v1"
        env_params["has_object"] = False
        FLAGS.total_steps = 20
    else:
        raise TypeError("No such environment till now")

    # number of actions to achieve subgoals in HDDPG
    x = pow(FLAGS.total_steps, 1/FLAGS.layers)
    if x - int(x) == 0:
        FLAGS.time_scale = int(x)
    else:
        FLAGS.time_scale = int(x) + 1                                

    FLAGS.num_exploration_episodes = 100
    FLAGS.num_test_episodes = 100           
    FLAGS.num_epochs = FLAGS.episodes // FLAGS.num_exploration_episodes

    env_params["obj_range"] = 0.15  
    env_params["target_range"] = 0.15 
    env_params["max_actions"] = FLAGS.total_steps  

    distance_threshold = 0.05  # 5cm
    env_params["end_goal_thresholds"] = distance_threshold
    env_params["subgoal_thresholds"] = distance_threshold

    if FLAGS.curriculum >= 2:
        range_lis = list(np.linspace(0.05, 0.15, FLAGS.curriculum))
        env_params['curriculum_list'] = range_lis

    # agent params
    agent_params = {}
    agent_params["subgoal_test_perc"] = 0.3

    agent_params["subgoal_penalty"] = -FLAGS.time_scale    # Define subgoal penalty for missing subgoal.
    agent_params["atomic_noise"] = 0.1
    agent_params["subgoal_noise"] = 0.03
    agent_params["epsilon"] = 0.1          # rate of choose random action

    agent_params["episodes_to_store"] = 1000
    agent_params["update_times"] = 40   
    agent_params["batch_size"] = 64    
    
    agent_params['imit_batch_size'] = 32
    agent_params['imit_ratio'] = FLAGS.imit_ratio

    return agent_params, env_params
