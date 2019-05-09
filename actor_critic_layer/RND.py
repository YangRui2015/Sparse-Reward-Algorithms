import tensorflow as tf
import numpy as np


def create_nn(input, input_num, output_num, init_val=0.001, relu=True, trainable=True, name=''):
    shape = [input_num, output_num]

    w_init = tf.random_uniform_initializer(minval=-init_val, maxval=init_val)
    b_init = tf.random_uniform_initializer(minval=-init_val, maxval=init_val)

    weights = tf.get_variable(name + "weights", shape, initializer=w_init, trainable=trainable)
    biases = tf.get_variable(name + "biases", [output_num], initializer=b_init, trainable=trainable)

    dot = tf.matmul(input, weights) + biases

    if not relu:
        return dot

    dot = tf.nn.relu(dot)
    return dot


class RND:
    def __init__(self, s_features, out_features=3, name="", learning_rate=0.001):
        self.s_features = s_features
        self.out_features = out_features
        self.lr = learning_rate
        self.name = name
        self._build_net()
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def _build_net(self):    # 创建两个random网络，一个训练一个固定
        self.state = tf.placeholder(tf.float32, [None, self.s_features], name=self.name + 'state')  # input

        with tf.variable_scope(self.name + 'train_net'):
            l1 = create_nn(self.state, self.s_features, 64, relu=True, trainable=True, name='l1')
            self.train_net_output = create_nn(l1, 64, self.out_features,relu=False, trainable=True, name='output')

        with tf.variable_scope(self.name + 'target_net'):
            l1_ = create_nn(self.state, self.s_features, 64, init_val=10, relu=True, trainable=False, name='l1')
            self.target_net_output = create_nn(l1_, 64, self.out_features, init_val=10, relu=False, trainable=False, name='output')

        self.loss = tf.reduce_mean(tf.squared_difference(self.train_net_output, self.target_net_output))
        self.intrinsic_reward = tf.reduce_mean(tf.squared_difference(self.train_net_output, self.target_net_output), axis=1)
        self._train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

    def train(self, state):
        loss, train_op = self.sess.run([self.loss, self._train_op], feed_dict={
            self.state: state,
        })
        return loss

    def get_intrinsic_reward(self, state):
        return self.sess.run(self.intrinsic_reward, feed_dict={self.state:state})

    def get_target(self, state):
        return self.sess.run(self.target_net_output, feed_dict={self.state: state})









