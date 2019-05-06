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
            tau=0.05):

        self.sess = sess
        # self.seed = FLAGS.seed

        # Determine range of actor network outputs.  This will be used to configure outer layer of neural network
        if layer_number == 0:             # 最底层输出动作而不是目标
            self.action_space_bounds = env.action_bounds
            self.action_offset = env.action_offset
        else:
            self.action_space_bounds = env.subgoal_bounds_symmetric
            self.action_offset = env.subgoal_bounds_offset
     
        # Dimensions of action will depend on layer level     
        if layer_number == 0:
            self.action_space_size = env.action_dim
        else:
            self.action_space_size = env.subgoal_dim

        self.actor_name = 'actor_' + str(layer_number) + str(time.time())

        # Dimensions of goal placeholder will differ depending on layer level
        if layer_number == FLAGS.layers - 1:
            self.goal_dim = env.end_goal_dim
        else:
            self.goal_dim = env.subgoal_dim

        self.state_dim = env.state_dim

        self.learning_rate = learning_rate
        # self.exploration_policies = exploration_policies
        self.tau = tau                                                   # what's tau
        self.batch_size = batch_size
        
        self.state_ph = tf.placeholder(tf.float32, shape=(None, self.state_dim))
        self.goal_ph = tf.placeholder(tf.float32, shape=(None, self.goal_dim))
        self.features_ph = tf.concat([self.state_ph, self.goal_ph], axis=1)

        # Create actor network
        self.infer = self.create_nn(self.features_ph, self.actor_name)

        # Target network code "repurposed" from Patrick Emani :^)
        self.weights = [v for v in tf.trainable_variables() if self.actor_name in v.op.name]
        # self.num_weights = len(self.weights)
        
        # Create target actor network
        self.target = self.create_nn(self.features_ph, name = self.actor_name + '_target')
        self.target_weights = [v for v in tf.trainable_variables() if self.actor_name in v.op.name][len(self.weights):]   # 在原来的网络之后加入的网络

        self.update_target_weights = \
	    [self.target_weights[i].assign(tf.multiply(self.weights[i], self.tau) +
                                                  tf.multiply(self.target_weights[i], 1. - self.tau))
                    for i in range(len(self.target_weights))]                                             # 平滑地去更新target网络
	
        self.action_derivs = tf.placeholder(tf.float32, shape=(None, self.action_space_size))                        # 动作的权重，确定性动作是单点策略
        self.unnormalized_actor_gradients = tf.gradients(self.infer, self.weights, -self.action_derivs)
        self.policy_gradient = list(map(lambda x: tf.div(x, self.batch_size), self.unnormalized_actor_gradients))     # map将第二个iterable的参数给第一个函数执行

        # self.policy_gradient = tf.gradients(self.infer, self.weights, -self.action_derivs)
        self.train = tf.train.AdamOptimizer(learning_rate).apply_gradients(zip(self.policy_gradient, self.weights))


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

        return len(weights)

        # self.sess.run(self.update_target_weights)

    # def create_nn(self, state, goal, name='actor'):
    def create_nn(self, features, name=None):
        
        if name is None:
            name = self.actor_name

        with tf.variable_scope(name + '_fc_1'):
            fc1 = layer(features, 64)
        # with tf.variable_scope(name + '_fc_2'):
        #     fc2 = layer(fc1, 64)
        # with tf.variable_scope(name + '_fc_3'):
        #     fc3 = layer(fc2, 64)
        with tf.variable_scope(name + '_fc_4'):
            fc4 = layer(fc1, self.action_space_size, is_output=True)

        output = tf.tanh(fc4) * self.action_space_bounds + self.action_offset
        return output
 
       
