#!usr/bin/python
# -*- coding: utf-8 -*-

# Main file for NN model
import tensorflow._api.v2.compat.v1 as tf
tf.disable_v2_behavior()

import numpy as np
from enum import Enum
from data import Distribution
from functools import wraps

tf.compat.v1.disable_eager_execution()

DATA_TYPE = Distribution.RANDOM

def set_data_type(data_type):
    global DATA_TYPE
    DATA_TYPE = data_type

# using cache
def memoize(func):
    memo = {}

    @wraps(func)
    def wrapper(*args):
        if args in memo:
            return memo[args]
        else:
            rv = func(*args)
            memo[args] = rv
            return rv

    return wrapper

# set parameter
class Parameter:
    def __init__(self, stages, cores, train_steps, batch_sizes, learning_rates):
        self.stage_set = stages
        self.core_set = cores
        self.train_step_set = train_steps
        self.batch_size_set = batch_sizes
        self.learning_rate_set = learning_rates

# parameter pool
class ParameterPool(Enum):
    LINEAR = Parameter(stages=[1, 10, 100, 10, 1], cores=[[1, 1], [1, 1], [1, 1], [1, 1], [1, 1]], train_steps=[20000, 20000, 20000, 20000, 20000],
                       batch_sizes=[50, 50, 50, 50, 50], learning_rates=[0.0001, 0.0001, 0.0001, 0.0001, 0.0001])
    RANDOM = Parameter(stages=[1, 10, 100, 10, 1], cores=[[1, 1], [1, 1], [1, 1], [1, 1], [1, 1]], train_steps=[20000, 20000, 20000, 20000, 20000],
                       batch_sizes=[50, 50, 50, 50, 50], learning_rates=[0.0001, 0.0001, 0.0001, 0.0001, 0.0001])
    LOGNORMAL = Parameter(stages=[1, 100, 100, 10, 1], cores=[[1, 16, 16, 1], [1, 8, 1]], train_steps=[2000, 400],
                          batch_sizes=[100, 50], learning_rates=[0.0001, 0.001])
    EXPONENTIAL = Parameter(stages=[1, 100, 100, 10, 1], cores=[[1, 8, 1], [1, 8, 1], [1, 8, 1], [1, 8, 1], [1, 8, 1]], train_steps=[30000, 20000, 20000, 20000, 20000],
                            batch_sizes=[50, 50, 50, 50, 50], learning_rates=[0.0001, 0.001, 0.001, 0.001, 0.001])
    # EXPONENTIAL = Parameter(stages=[1, 100], cores=[[1, 16, 16, 1], [1, 8, 1]], train_steps=[20000, 300],
    #                       batch_sizes=[20, 50], learning_rates=[0.0001, 0.001], keep_ratios=[1.0, 1.0])
    NORMAL = Parameter(stages=[1, 100, 100, 10, 1], cores=[[1, 8, 1], [1, 8, 1]], train_steps=[20000, 300],
                       batch_sizes=[50, 50], learning_rates=[0.0001, 0.001])

# initialize weight marrix
def weight_variable(shape):
    if DATA_TYPE == Distribution.LINEAR:
        initial = tf.constant(0.1, shape=shape)
    elif DATA_TYPE == Distribution.RANDOM:
        initial = tf.constant(0.1, shape=shape)
    elif DATA_TYPE == Distribution.LOGNORMAL:
        initial = tf.truncated_normal(shape=shape, stddev=0.1)
        # initial = tf.constant(0.1, shape=shape)
    elif DATA_TYPE == Distribution.EXPONENTIAL:
        # initial = tf.truncated_normal(shape=shape, stddev=0.1)
        initial = tf.constant(0.1, shape=shape)
    elif DATA_TYPE == Distribution.NORMAL:
        initial = tf.truncated_normal(shape=shape, mean=0.1, stddev=0.1)
    else:
        initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

# initialize 
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

# extract matrix for predicting position
class AbstractNN:
    def __init__(self, weights, bias, core_nums, mean_err):
        self.weights = weights
        self.bias = bias        
        self.core_nums = core_nums
        self.mean_err = mean_err

    @memoize
    def predict(self, input_key):
        # np.mat函数将输入转换为矩阵的形式
        tmp_res = np.mat(input_key) * np.mat(self.weights[0]) + np.mat(self.bias[0])
        for i in range(1, len(self.core_nums) - 1):
            tmp_res = np.mat(tmp_res) * np.mat(self.weights[i]) + np.mat(self.bias[i])
        return int(np.round(tmp_res[0][0]))

# Netural Network Model
class TrainedNN:
    def __init__(self, threshold, useThreshold, cores, train_step_num, batch_size, learning_rate, train_x, train_y,
                 test_x, test_y):
        #set parameters
        if cores is None:
            cores = []
        self.threshold_nums = threshold
        self.useThreshold = useThreshold
        self.core_nums = cores
        self.train_step_nums = train_step_num
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.train_x = train_x
        self.train_y = train_y
        self.test_x = np.array([test_x]).T
        self.test_y = np.array([test_y]).T
        self.sess = tf.Session()
        self.batch = 1
        self.batch_x = np.array([self.train_x[0:self.batch_size]]).T
        self.batch_y = np.array([self.train_y[0:self.batch_size]]).T
        self.y_ = tf.placeholder(tf.float32, shape=[None, self.core_nums[-1]])
        self.w_fc = []
        self.b_fc = []
        for i in range(len(self.core_nums) - 1):
            self.w_fc.append(weight_variable([self.core_nums[i], self.core_nums[i + 1]]))
            self.b_fc.append(bias_variable([self.core_nums[i + 1]]))
        self.h_fc = [None for i in range(len(self.core_nums))]
        self.h_fc[0] = tf.placeholder(tf.float32, shape=[None, self.core_nums[0]])

    # get next batch of data
    def next_batch(self):
        if self.batch * self.batch_size + self.batch_size < len(self.train_x):
            self.batch_x = np.array([self.train_x[self.batch * self.batch_size:(self.batch + 1) * self.batch_size]]).T
            self.batch_y = np.array([self.train_y[self.batch * self.batch_size:(self.batch + 1) * self.batch_size]]).T
            self.batch += 1
        else:
            # 最后一个 batch 超出数据集的长度，则取到数据集结尾
            self.batch_x = np.array([self.train_x[self.batch * self.batch_size:len(self.train_x)]]).T
            self.batch_y = np.array([self.train_y[self.batch * self.batch_size:len(self.train_y)]]).T
            self.batch = 0

    # train model
    def train(self):
        for i in range(len(self.core_nums) - 1):
            # 全连接 + relu层
            self.h_fc[i + 1] = tf.matmul(self.h_fc[i], self.w_fc[i]) + self.b_fc[i]

        # 计算 cross_entropy 损失
        self.cross_entropy = tf.reduce_mean(tf.losses.mean_squared_error(self.y_, self.h_fc[len(self.core_nums) - 1]))
        # adam 优化器
        self.train_step = tf.train.AdamOptimizer(self.learning_rate).minimize(self.cross_entropy)
        self.sess.run(tf.global_variables_initializer())
        
        last_err = 0
        err_count = 0
        for step in range(0, self.train_step_nums):
            self.sess.run(self.train_step,
                          feed_dict={self.h_fc[0]: self.batch_x, self.y_: self.batch_y})
            # check every 100 steps
            if step % 100 == 0:
                err = self.sess.run(self.cross_entropy, feed_dict={self.h_fc[0]: np.array([self.train_x]).T,
                                                                   self.y_: np.array([self.train_y]).T})
                print("cross_entropy: %f" % err)
                if step == 0:
                    last_err = err 
                # use threhold to stop train 
                if self.useThreshold:
                    if err < self.threshold_nums:
                        return
                # not use threshold, stop when error stop decreasing
                elif err > last_err:
                    err_count += 1
                    if err_count == 10:
                        return
                last_err = err

            self.next_batch()    

    # calculate mean error
    def cal_err(self):
        mean_err = self.sess.run(self.cross_entropy, feed_dict={self.h_fc[0]: np.array([self.train_x]).T,
                                                                self.y_: np.array([self.train_y]).T})
        return mean_err

    # save model
    def save(self, path):
        saver = tf.train.Saver()
        saver.save(self.sess, path)

    # get weight matrix
    def get_weights(self):
        weights = []
        for i in range(len(self.core_nums) - 1):
            weights.append(self.sess.run(self.w_fc[i], feed_dict={self.h_fc[0]: np.array([self.train_x]).T,
                                                                       self.y_: np.array([self.train_y]).T}).tolist())
        return weights

    # get bias matrix
    def get_bias(self):
        bias = []
        for i in range(len(self.core_nums) - 1):
            bias.append(self.sess.run(self.b_fc[i], feed_dict={self.h_fc[0]: np.array([self.train_x]).T,
                                                                    self.y_: np.array([self.train_y]).T}).tolist())
        return bias
