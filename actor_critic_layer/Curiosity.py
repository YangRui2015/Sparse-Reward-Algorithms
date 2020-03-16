import tensorflow as tf
import numpy as np
from .utils import create_nn

class ForwardDynamics:
    def __init__(self, state_dim, action_dim, name="", learning_rate=1e-5):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.input_dim = state_dim + action_dim
        self.lr = learning_rate
        self.name = name
        self._build_net()
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def _build_net(self):    
        self.state = tf.placeholder(tf.float32, [None, self.state_dim], name=self.name + 'state')  # input
        self.next_state = tf.placeholder(tf.float32, [None, self.state_dim], name=self.name + 'next_state')
        self.action = tf.placeholder(tf.float32, [None, self.action_dim], name=self.name + 'action')
        self.input = tf.concat((self.state, self.action),axis=-1)

        with tf.variable_scope(self.name + 'train_net'):
            l1 = create_nn(self.input, self.input_dim, 64, relu=True, trainable=True, name='l1')
            self.train_net_output = create_nn(l1, 64, self.state_dim,relu=False, trainable=True, name='output')

        self.loss = tf.reduce_mean(tf.squared_difference(self.train_net_output, self.next_state)) # loss 
        self.intrinsic_reward = tf.reduce_mean(tf.squared_difference(self.train_net_output, self.next_state), axis=-1)  
        self._train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

    def train(self, state, action, next_state):
        loss, train_op = self.sess.run([self.loss, self._train_op], feed_dict={self.state: state,self.action: action,self.next_state: next_state })
        return loss

    def get_intrinsic_reward(self, state, action, next_state):   
        return self.sess.run(self.intrinsic_reward, feed_dict={
            self.state:state,
            self.action:action,
            self.next_state:next_state
            })

    def predict(self, state, action):
        return self.sess.run(self.train_net_output, feed_dict={
            self.state: state,
            self.action:action
            })









