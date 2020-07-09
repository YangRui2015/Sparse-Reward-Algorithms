# -*- coding: utf-8 -*-
# main.py
# author: yangrui
# description: 
# created: 2019-05-10T19:01:15.155Z+08:00
# last-modified: 2020-03-13T10:00:30.182Z+08:00
# email: yangrui19@mails.tsinghua.edu.cn

from global_utils import print_summary
from options import parse_options
from global_utils import set_global_seed, save_performance, plot_data
import time
from agent_env_params import design_agent_and_env
from multiprocessing import Process
import random

from environment import Environment
from agent import Agent


def run_HAC(FLAGS,env,agent, plot_figure=False, num=0):
    from global_utils import save_plot_figure # import here is for mutilprocessing

    NUM_EPOCH = FLAGS.num_epochs  
    SAVE_FREQ = FLAGS.save_freq  
    # Print task summary
    print_summary(FLAGS, env)

    if not FLAGS.test:
        num_episodes = FLAGS.num_exploration_episodes  
    else:
        num_episodes = FLAGS.num_test_episodes
        NUM_EPOCH = 1   # only test 1 epoch

    performance_list = []
    test_performance_list = []
    if FLAGS.curriculum >= 2:
        curriculum_epoch = NUM_EPOCH / FLAGS.curriculum
        assert curriculum_epoch == int(curriculum_epoch), 'NUM_EPOCH / FLAGS.curriculum should be int' 
        
    for epoch in range(1, NUM_EPOCH + 1):    
        successful_episodes = 0
        if not FLAGS.test and  FLAGS.curriculum >= 2:
            env.set_goal_range(env_params['curriculum_list'][int((epoch - 1) // curriculum_epoch)])

        for episode in range(num_episodes):                        
            print("\nEpoch %d, Episode %d" % (epoch, episode))
            # Train for an epoch
            success = agent.train(env, epoch * num_episodes + episode,test=FLAGS.test)
            if success:
                print("End Goal Achieved\n")
                successful_episodes += 1
        # Save agent
        if epoch % SAVE_FREQ == 0 and not FLAGS.test and FLAGS.threadings == 1:
            agent.save_model(epoch * num_episodes)
        success_rate = successful_episodes / num_episodes * 100
        print("\nEpoch %d, Success Rate %.2f%%" % (epoch, success_rate))
        performance_list.append(success_rate)

        if not FLAGS.test: 
            success_test = 0
            if FLAGS.curriculum >= 2:
                env.set_goal_range(env_params['curriculum_list'][-1])

            print('\ntesting for %d episodes' % (FLAGS.num_test_episodes))
            for episode in range(FLAGS.num_test_episodes):
                success = agent.train(env, episode, test=True)
                success_test += int(success)
            success_rate = success_test / FLAGS.num_test_episodes * 100
            print('testing accuracy: %.2f%%' % (success_rate))
            test_performance_list.append(success_test)

        if plot_figure:
            save_plot_figure(performance_list)
            save_plot_figure(test_performance_list, name='test-performance.jpg')

    save_performance(performance_list, test_performance_list, FLAGS=FLAGS, thread_num=num)
    if FLAGS.save_experience:
        agent.save_experience()
        


def worker(agent_params, env_params, FLAGS, i):
    seed = int(time.time()) + random.randint(0, 100)
    set_global_seed(seed)
    FLAGS.seed = seed
    env = Environment(env_params, FLAGS)
    agent = Agent(FLAGS, env, agent_params)
    run_HAC(FLAGS, env, agent, plot_figure=False, num=i)


FLAGS = parse_options()
agent_params, env_params = design_agent_and_env(FLAGS)

assert FLAGS.threadings >= 1, "Threadings should be more than 1!"
if FLAGS.threadings == 1:
    seed = int(time.time()) + random.randint(0,100)
    set_global_seed(seed)
    FLAGS.seed = seed
    env = Environment(env_params, FLAGS)
    agent = Agent(FLAGS, env, agent_params)
    run_HAC(FLAGS, env, agent, plot_figure=True)
else:
    # parallel run
    thread_list = []
    for i in range(FLAGS.threadings):
        p = Process(target=worker, args=(agent_params, env_params, FLAGS, i))
        p.start()
        thread_list.append(p)

    for p in thread_list:
        p.join()










