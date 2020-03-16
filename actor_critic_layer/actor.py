import tensorflow as tf
import numpy as np
from .utils import layer
import time

class Actor():
    def __init__(self, 
            sess,
            env,
            batch_size,
            layer_number,
            FLAGS,
            learning_rate=0.001,
            tau=0.05,
            imit_batch_size=32,
            imit_learning_rate=0.001,
            imit_ratio=1
            ):
        self.sess = sess
        self.imit_batch_size = imit_batch_size
        self.imit_init_ratio = imit_ratio
        self.imit_ratio = imit_ratio
        self.FLAGS = FLAGS
        if layer_number == 0:             # the last layer produce actions
            self.action_space_bounds = env.action_bounds
            self.action_offset = env.action_offset
        else:
            self.action_space_bounds = env.subgoal_bounds_symmetric
            self.action_offset = env.subgoal_bounds_offset
      
        if layer_number == 0:
            self.action_space_size = env.action_dim
        else:
            self.action_space_size = env.subgoal_dim

        self.actor_name = 'actor_' + str(layer_number)
        if FLAGS.threadings > 1:
            self.actor_name += str(time.time())

        if layer_number == FLAGS.layers - 1:
            self.goal_dim = env.end_goal_dim
        else:
            self.goal_dim = env.subgoal_dim

        self.state_dim = env.state_dim
        self.learning_rate = learning_rate
        self.tau = tau                                                   
        self.batch_size = batch_size
        self.state_ph = tf.placeholder(tf.float32, shape=(None, self.state_dim))
        self.goal_ph = tf.placeholder(tf.float32, shape=(None, self.goal_dim))
        self.features_ph = tf.concat([self.state_ph, self.goal_ph], axis=1)

        # actor network
        self.infer = self.create_nn(self.features_ph, self.actor_name)
        self.weights = [v for v in tf.trainable_variables() if self.actor_name in v.op.name]
        
        # target actor network, we didn't use it in experiment because we found it has no improvemnet
        self.target = self.create_nn(self.features_ph, name = self.actor_name + '_target')
        self.target_weights = [v for v in tf.trainable_variables() if self.actor_name in v.op.name][len(self.weights):]   

        self.update_target_weights = [self.target_weights[i].assign(tf.multiply(self.weights[i], self.tau) +
                                    tf.multiply(self.target_weights[i], 1. - self.tau))
                                    for i in range(len(self.target_weights))]                                             
	
        self.action_derivs = tf.placeholder(tf.float32, shape=(None, self.action_space_size))                       
        self.unnormalized_actor_gradients = tf.gradients(self.infer, self.weights, -self.action_derivs)
        self.policy_gradient = list(map(lambda x: tf.div(x, self.batch_size), self.unnormalized_actor_gradients))             
        self.train = tf.train.AdamOptimizer(learning_rate).apply_gradients(zip(self.policy_gradient, self.weights))

        if FLAGS.imitation:
            self.demo_action = tf.placeholder(tf.float32, shape=(None, self.action_space_size))
            self.imit_loss = tf.multiply(self.imit_ratio, tf.reduce_mean(tf.square(self.infer - self.demo_action)))
            self.imit_train = tf.train.AdamOptimizer(imit_learning_rate).minimize(self.imit_loss)

    def get_action(self, state, goal):
        actions = self.sess.run(self.infer,
                feed_dict={
                    self.state_ph: state,
                    self.goal_ph: goal
                })
        return actions

    def get_target_action(self, state, goal):
        actions = self.sess.run(self.target,
                feed_dict={
                    self.state_ph: state,
                    self.goal_ph: goal
                })
        return actions

    def update(self, state, goal, action_derivs):
        weights, policy_grad, _ = self.sess.run([self.weights, self.policy_gradient, self.train],
                feed_dict={
                    self.state_ph: state,
                    self.goal_ph: goal,
                    self.action_derivs: action_derivs
                })
    
    def imit_update(self, imit_state, imit_goal, imit_action):
        imit_loss, imit_train = self.sess.run(
                [self.imit_loss, self.imit_train],
                feed_dict={
                    self.state_ph:imit_state,
                    self.goal_ph:imit_goal,
                    self.demo_action:imit_action
                })
        return imit_loss, imit_train

    
    def create_nn(self, features, name=None):
        if name is None:
            name = self.actor_name
        with tf.variable_scope(name + '_fc_1'):
            fc1 = layer(features, 64)
        with tf.variable_scope(name + '_fc_2'):
            fc2 = layer(fc1, self.action_space_size, is_output=True)

        output = tf.tanh(fc2) * self.action_space_bounds + self.action_offset
        return output
 
       
