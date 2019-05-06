import numpy as np
from actor_critic_layer.layer import Layer
import tensorflow as tf
import os
import time


class Agent():
    def __init__(self,FLAGS, env, agent_params, num=0):
        self.num = num  # 多进程序号
        self.FLAGS = FLAGS
        self.sess = tf.Session()

        # Set subgoal testing ratio each layer will use
        self.subgoal_test_perc = agent_params["subgoal_test_perc"]     # 以多少的概率进行subgoal的测试

        # Create agent with number of levels specified by user       
        self.layers = [Layer(i,FLAGS,env,self.sess,agent_params) for i in range(FLAGS.layers)]        

        # Below attributes will be used help save network parameters
        self.saver = None
        self.model_dir = None
        self.model_loc = None

        # Initialize actor/critic networks.  Load saved parameters if not retraining
        self.initialize_networks()   
        
        # goal_array will store goal for each layer of agent.
        self.goal_array = [None for i in range(FLAGS.layers)]

        self.current_state = None

        # Track number of low-level actions executed, 最低级动作的数量
        self.steps_taken = 0

        # Below hyperparameter specifies number of Q-value updates made after each episode， 每个episode后Q值更新次数？？
        self.num_updates = agent_params["update_times"]


        self.other_params = agent_params


    def check_goals(self,env):
        # goal_status is vector showing status of whether a layer's goal has been achieved
        goal_status = [False for i in range(self.FLAGS.layers)]

        max_lay_achieved = None

        # Project current state onto the subgoal and end goal spaces
        proj_subgoal = env.project_state_to_subgoal(self.current_state)
        proj_end_goal = env.project_state_to_end_goal(self.current_state)

        for i in range(self.FLAGS.layers):

            goal_achieved = True
            
            # If at latest layer, compare to end goal thresholds
            if i == self.FLAGS.layers - 1:

                # Check dimensions are appropriate         
                assert len(proj_end_goal) == len(self.goal_array[i]) == len(env.end_goal_thresholds), "Projected end goal, actual end goal, and end goal thresholds should have same dimensions"

                # Check whether layer i's goal was achieved by checking whether projected state is within the goal achievement threshold
                for j in range(len(proj_end_goal)):
                    if np.absolute(self.goal_array[i][j] - proj_end_goal[j]) > env.end_goal_thresholds[j]:
                        goal_achieved = False
                        break

            # If not highest layer, compare to subgoal thresholds
            else:
                # Check that dimensions are appropriate
                assert len(proj_subgoal) == len(self.goal_array[i]) == len(env.subgoal_thresholds), "Projected subgoal, actual subgoal, and subgoal thresholds should have same dimensions"           

                # Check whether layer i's goal was achieved by checking whether projected state is within the goal achievement threshold
                for j in range(len(proj_subgoal)):
                    if np.absolute(self.goal_array[i][j] - proj_subgoal[j]) > env.subgoal_thresholds[j]:
                        goal_achieved = False
                        break

            # If projected state within threshold of goal, mark as achieved
            if goal_achieved:
                goal_status[i] = True
                max_lay_achieved = i
            else:
                goal_status[i] = False

        return goal_status, max_lay_achieved

    def clear_buffer(self):
        for i in range(len(self.layers)):
            self.layers[i].replay_buffer.clear()



    def initialize_networks(self):
        model_vars = tf.trainable_variables()
        self.saver = tf.train.Saver(model_vars)

        # Set up directory for saving models
        self.model_dir = os.getcwd() + '/models'
        self.model_loc = self.model_dir + '/HAC_' + str(self.num) + '.ckpt'

        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

        self.sess.run(tf.global_variables_initializer())

        if not self.FLAGS.retrain:
            self.saver.restore(self.sess, tf.train.latest_checkpoint(self.model_dir))


    # Save neural network parameters
    def save_model(self, episode):
        self.saver.save(self.sess, self.model_loc, global_step=episode)


    # Update actor and critic networks for each layer
    def learn(self):

        for i in range(len(self.layers)):     # 每个layer一次学习一个episode的次数是相同的，默认是40
            self.layers[i].learn(self.num_updates)

       
    # Train agent for an episode， train函数只持续一个episode, episode_num是episode的序号
    def train(self,env, episode_num):

        # Select initial state from in initial state space, defined in environment.py
        self.current_state = env.reset_sim()

        # Select final goal from final goal space, defined in "design_agent_and_env.py" 得到每次更新的目标，先reset然后得到新目标
        self.goal_array[self.FLAGS.layers - 1] = env.goal
        print("Next End Goal: ", self.goal_array[self.FLAGS.layers - 1])

        # Reset step counter
        self.steps_taken = 0

        # Train for an episode，似乎是执行/存储数据，而不是训练
        goal_status, max_lay_achieved = self.layers[self.FLAGS.layers-1].train(self,env, episode_num=episode_num)

        # Update actor/critic networks if not testing
        if not self.FLAGS.test:
            self.learn()

        # Return whether end goal was achieved 返回目标是否达到
        return goal_status[self.FLAGS.layers-1]

    




        

        

        
        
        


