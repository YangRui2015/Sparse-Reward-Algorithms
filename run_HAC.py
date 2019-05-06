# 主函数
from global_utils import print_summary
from options import parse_options
from global_utils import set_global_seed, save_performance, plot_data
import time
from importlib import import_module
from multiprocessing import Process
import random

from environment import Environment
from agent import Agent


def run_HAC(FLAGS,env,agent, plot_figure=False, num=0):
    from global_utils import save_plot_figure

    NUM_EPOCH = FLAGS.num_epochs  # 训练多少个epoch
    SAVE_FREQ = FLAGS.save_freq  # 每两个epoch存一次

    # Print task summary
    print_summary(FLAGS, env)

    if not FLAGS.test:
        num_episodes = FLAGS.num_exploration_episodes  # 默认为100
    else:
        num_episodes = FLAGS.num_test_episodes
        NUM_EPOCH = 1   # 测试只测试100个episode

    performance_list = []
    for epoch in range(1, NUM_EPOCH + 1):    # 从1开始，使第一个不存，最后一个存
        successful_episodes = 0

        for episode in range(num_episodes):                        # episode这里是序号
            print("\nEpoch %d, Episode %d" % (epoch, episode))

            # Train for an episode
            success = agent.train(env, episode)

            if success:
                print("End Goal Achieved\n")
                successful_episodes += 1

        # Save agent
        if epoch % SAVE_FREQ == 0 and not FLAGS.test:
            agent.save_model(epoch * num_episodes)

        success_rate = successful_episodes / num_episodes * 100
        print("\nEpoch %d, Success Rate %.2f%%" % (epoch, success_rate))
        performance_list.append(success_rate)
        if plot_figure:
            save_plot_figure(performance_list)

    save_performance(performance_list, FLAGS, num)


def worker(agent_params, env_params, FLAGS, i):
    set_global_seed(time.time())
    env = Environment(env_params, FLAGS)
    agent = Agent(FLAGS, env, agent_params, i)
    run_HAC(FLAGS, env, agent, plot_figure=False, num=i)


FLAGS = parse_options()
# set_global_seed(FLAGS.seed)
design_function = import_module("agent_env_params_" + FLAGS.env)
agent_params, env_params = design_function.design_agent_and_env(FLAGS)

assert FLAGS.threadings >= 1, "Threadings should be more than 1!"
if FLAGS.threadings == 1:
    set_global_seed(time.time() + random.randint(0,10))
    env = Environment(env_params, FLAGS)
    agent = Agent(FLAGS, env, agent_params)
    run_HAC(FLAGS, env, agent, plot_figure=True)
else:
    # 并行运行
    thread_list = []
    for i in range(FLAGS.threadings):
        p = Process(target=worker, args=(agent_params, env_params, FLAGS, i))
        p.start()
        thread_list.append(p)

    for p in thread_list:
        p.join()










# 依次运行
# FLAGS = parse_options()
# # set_global_seed(FLAGS.seed)
# print("Start time 1")
# set_global_seed(time.time())
# design_function = import_module("agent_env_params_" + FLAGS.env)
# agent_params, env_params = design_function.design_agent_and_env(FLAGS)
# env = Environment(env_params, FLAGS)
# agent = Agent(FLAGS, env, agent_params)
#
# for i in range(1, FLAGS.times + 1):
#     print("\n\n###################################")
#     print("\nStart times {} ......".format(i))
#     print("\n\n####################################\n")
#     set_global_seed(time.time())
#     assert FLAGS.retrain == True,  "Not Retrain ERROR!"
#     agent.initialize_networks()    # 重新训练
#     agent.clear_buffer()
#     run_HAC(FLAGS, env, agent)


