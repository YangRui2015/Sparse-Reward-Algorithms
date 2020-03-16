import numpy as np
from actor_critic_layer.layer import Layer
import tensorflow as tf
import os
import time
from global_utils import save_pkl, load_pkl


class Agent():
    def __init__(self,FLAGS, env, agent_params, num=0):
        self.num = num  # number of process, for mutilprocessing
        self.FLAGS = FLAGS
        self.sess = tf.Session()

        # Set subgoal testing ratio each layer will use
        self.subgoal_test_perc = agent_params["subgoal_test_perc"]        
        self.layers = [Layer(i,FLAGS,env,self.sess,agent_params) for i in range(FLAGS.layers)]        
        self.saver = None
        self.model_dir = None
        self.model_loc = None
        self.initialize_networks()   
        self.goal_array = [None for i in range(FLAGS.layers)]
        self.current_state = None
        self.steps_taken = 0
        self.num_updates = agent_params["update_times"]
        self.other_params = agent_params
        if FLAGS.imitation:
            self.load_experience()

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
                assert len(proj_end_goal) == len(self.goal_array[i]), "Projected end goal, actual end goal, and end goal thresholds should have same dimensions"
                for j in range(len(proj_end_goal)):
                    if np.absolute(self.goal_array[i][j] - proj_end_goal[j]) > env.end_goal_thresholds:
                        goal_achieved = False
                        break
            else:
                assert len(proj_subgoal) == len(self.goal_array[i]), "Projected subgoal, actual subgoal, and subgoal thresholds should have same dimensions"
                for j in range(len(proj_subgoal)):
                    if np.absolute(self.goal_array[i][j] - proj_subgoal[j]) > env.subgoal_thresholds:
                        goal_achieved = False
                        break
            
            if goal_achieved:
                goal_status[i] = True
                max_lay_achieved = i
            else:
                goal_status[i] = False
        return goal_status, max_lay_achieved

    def clear_buffer(self):
        for i in range(len(self.layers)):
            self.layers[i].replay_buffer.clear()
    
    # to save demo for imitation learning
    def save_experience(self, path='./data/demo_data_'):
        for i in range(len(self.layers)):
            experience = self.layers[i].replay_buffer.experiences
            save_pkl(experience, path + str(i) + '.pkl')
    
    # for imitation learning
    def load_experience(self, path='./data/demo_data_'):
        for i in range(len(self.layers)):
            experience = load_pkl(path + str(i) + '.pkl')
            self.layers[i].imitation_buffer._init_experience(experience)

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
    def learn(self, episode_num):
        for i in range(len(self.layers)):     
            self.layers[i].learn(self.num_updates, episode_num)

    def train(self,env, episode_num, test=False):
        self.current_state = env.reset_sim()
        self.goal_array[self.FLAGS.layers - 1] = env.goal
        self.steps_taken = 0
        goal_status, max_lay_achieved = self.layers[self.FLAGS.layers-1].train(self, env,test=test, episode_num=episode_num)
        if not test:
            self.learn(episode_num)
        return goal_status[self.FLAGS.layers-1]

    




        

        

        
        
        


