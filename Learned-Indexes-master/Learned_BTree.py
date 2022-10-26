from __future__ import print_function

import random

import pandas as pd
from Trained_NN import TrainedNN, AbstractNN, ParameterPool, set_data_type
from btree import BTree,Item
from data import create_data_no_storage, Distribution
from linear_model import LinearModel
# from sklearn.linear_model import LinearRegression
# from sklearn.linear_model import LogisticRegression
import time, gc, json
import os, sys, getopt
import numpy as np

BLOCK_SIZE = 4096
MAX_SUB_NUM = int(BLOCK_SIZE / 8)
DEGREE = int((MAX_SUB_NUM + 1) / 2)

# Setting
MAX_NUMBER = 10000000

filePath = {
    Distribution.LINEAR: "data/linear.csv",
    Distribution.RANDOM: "data/random.csv",
    Distribution.EXPONENTIAL: "data/exponential.csv",
    Distribution.NORMAL: "data/normal.csv",
    Distribution.LOGNORMAL: "data/lognormal.csv",
    Distribution.WIKI: "data/wiki_ts_200M_uint64.csv",
    Distribution.OSM: "data/osm_cellids_200M_uint64.csv",
    Distribution.BOOKS: "data/books_200M_uint64.csv",
    Distribution.FB: "data/fb_200M_uint64.csv"
}

# result record path
pathString = {
    Distribution.LINEAR: "Linear",
    Distribution.RANDOM: "Random",
    Distribution.EXPONENTIAL: "Exponential",
    Distribution.NORMAL: "Normal",
    Distribution.LOGNORMAL: "Lognormal",
    Distribution.WIKI: "WIKI",
    Distribution.OSM: "OSM",
    Distribution.BOOKS: "BOOKS",
    Distribution.FB: "FB"
}

# threshold for train (judge whether stop train and replace with BTree)
thresholdPool = {
    Distribution.LINEAR: [1, 5, 5, 5, 5],
    Distribution.RANDOM: [1, 4, 4, 4, 4],
    Distribution.EXPONENTIAL: [55, 10000, 10000, 10000, 10000],
    Distribution.NORMAL: [10, 100, 100, 100, 100],
    Distribution.LOGNORMAL: [55, 10000, 10000, 10000, 10000],
    Distribution.WIKI: [1, 4, 4, 4, 4],
    Distribution.OSM: [1, 4, 4, 4, 4],
    Distribution.BOOKS: [1, 4, 4, 4, 4],
    Distribution.FB: [1, 4, 4, 4, 4]
}

# whether use threshold to stop train for models in stages
useThresholdPool = {
    Distribution.LINEAR: [True, False, False, False, False],
    Distribution.RANDOM: [True, False, False, False, False],
    Distribution.EXPONENTIAL: [True, False, False, False, False],
    Distribution.NORMAL: [True, False, False, False, False],
    Distribution.LOGNORMAL: [True, False, False, False, False],
    Distribution.WIKI: [True, False, False, False, False],
    Distribution.OSM: [True, False, False, False, False],
    Distribution.BOOKS: [True, False, False, False, False],
    Distribution.FB: [True, False, False, False, False]
}

class Learned_Index:
    def __init__(self, stage, distribution, model):
        self.distribution = distribution
        # read parameter
        if distribution == Distribution.LINEAR:
            self.parameter = ParameterPool.LINEAR.value
        elif distribution == Distribution.RANDOM:
            self.parameter = ParameterPool.RANDOM.value
        elif distribution == Distribution.LOGNORMAL:
            self.parameter = ParameterPool.LOGNORMAL.value
        elif distribution == Distribution.EXPONENTIAL:
            self.parameter = ParameterPool.EXPONENTIAL.value
        elif distribution == Distribution.NORMAL:
            self.parameter = ParameterPool.NORMAL.value
        elif distribution == Distribution.WIKI:
            self.parameter = ParameterPool.WIKI.value
        elif distribution == Distribution.FB:
            self.parameter = ParameterPool.FB.value
        elif distribution == Distribution.OSM:
            self.parameter = ParameterPool.OSM.value
        elif distribution == Distribution.BOOKS:
            self.parameter = ParameterPool.BOOKS.value
        else:
            raise ValueError('The Distribution Is Wrong.')
        self.stage = stage
        self.model = model

        self.stage_set = self.parameter.stage_set
        # set number of models for the rest stage
        self.stage_set[1] = 10  # (1 model deal with 1000 records)
        self.stage_set[2] = 10  # (1 model deal with 100 records)
        self.stage_set[3] = 10  # (1 model deal with 10 records)
        self.stage_set[4] = 1

        # 存放最后一层模型中容纳的item个数
        self.last_stage_num = 10000
        # 存放最后一层模型中的index
        self.last_item_index = [[] for i in range(self.last_stage_num)]

        # 存储整个learned index中的key和value值
        self.index = [[None for j in range(self.stage_set[i + 1])] for i in range(stage)]
        self.keys = [None for i in range (MAX_NUMBER)]
        self.values = [None for i in range (MAX_NUMBER)]
        self.kv_index = [k for k in range (MAX_NUMBER)]

    # hybrid training structure, 2 stages
    # threshold 用于模型训练的早停以及判断是否用b-tree来替代最后一层的learned index
    # use_threshold 用于表示是否使用threshold来进行模型训练的早停
    def hybrid_training(self, threshold, use_threshold, stage, stage_nums, core_nums, train_step_nums, batch_size_nums,
                        learning_rate_nums, train_data_x, train_data_y, test_data_x, test_data_y):
        # stage_nums 为一个列表，存储着每个 stage 中 model 的个数
        stage_length = stage  # stage_length 表示 stage 的个数
        # 格式【stage个数个【】,其中包含model个数个【】】
        tmp_inputs = [[[] for j in range(stage_nums[i])] for i in range(stage_length)]
        tmp_labels = [[[] for j in range(stage_nums[i])] for i in range(stage_length)]
        # 放入对应的输入和标签
        tmp_inputs[0][0] = train_data_x  # 把数据全塞入第一个stage中的模型
        tmp_labels[0][0] = train_data_y
        test_inputs = test_data_x
        # 计算最后一层stage中存放的item个数
        self.last_stage_num = len(train_data_x)
        for i in range(1,stage_length):
            self.last_stage_num = int(self.last_stage_num / stage_nums[i])
        # 分配每个模型中的item值进行训练
        for i in range(0, stage_length):
            print("the stage ", i, " is training")
            for j in range(0, stage_nums[i]):
                TOTAL_NUMBER = len(tmp_inputs[i][j])  # 获取第i层stage第j个model中数据的数量
                if TOTAL_NUMBER == 0:  # 这个模型中没有数据，直接跳过
                    continue
                inputs = tmp_inputs[i][j]
                labels = []
                test_labels = []
                if i <= stage_length - 2:
                    # 此层中对应的位置 / 此层中总共的数据数量 * 下一层model的个数
                    divisor = stage_nums[i + 1] * 1.0 / TOTAL_NUMBER    # eg：此层共1000个数，下一层10个model，即前100个数应划分至下一层的第一个model
                    # 将label转换成对应的下一层所对应stage的位置
                    for k in tmp_labels[i][j]:
                        labels.append(int(k * divisor))
                    for k in test_data_y:
                        test_labels.append(int(k * divisor))
                else:
                    # 在最后一层stage中，无需将label转换成下一层所对应stage的位置，只需为原始的label即可
                    labels = tmp_labels[i][j]
                    test_labels = test_data_y
                    self.last_item_index[j] = labels
                    # train model
                print("the model ", j, " is training")
                tmp_index = TrainedNN(threshold[i], use_threshold[i], core_nums[i], train_step_nums[i],
                                      batch_size_nums[i],
                                      learning_rate_nums[i],
                                      inputs, labels, test_inputs, test_labels)
                tmp_index.train()
                # get parameters in model (weight matrix and bias matrix)
                self.index[i][j] = AbstractNN(tmp_index.get_weights(), tmp_index.get_bias(), core_nums[i],
                                              tmp_index.cal_err())
                del tmp_index
                gc.collect()  # 垃圾回收
                # 存在下一个stage
                if i < stage_length - 1:
                    # allocate data into training set for models in next stage
                    # 第一个stage中的数据为tmp_inputs，为其中的每一个数据预测其在下一个stage中的位置，并将其放入
                    for ind in range(len(tmp_inputs[i][j])):
                        # pick model in next stage with output of this model
                        # 预测每一个数据对应的下一个stage位置
                        p = self.index[i][j].predict(tmp_inputs[i][j][ind])
                        if p > stage_nums[i + 1] - 1:
                            p = stage_nums[i + 1] - 1
                        if p < 0:
                            p = 0
                        # 放入下一个stage的数据
                        tmp_inputs[i + 1][p].append(tmp_inputs[i][j][ind])
                        tmp_labels[i + 1][p].append(tmp_labels[i][j][ind])

        # 最后一层的每个stage
        for i in range(stage_nums[stage_length - 1]):
            # 这个stage中没有数据
            if self.index[stage_length - 1][i] is None:
                continue
            mean_abs_err = self.index[stage_length - 1][i].mean_err
            # stage的损失大于阈值，则用btree代替
            if mean_abs_err > threshold[stage_length - 1]:
                # replace model with BTree if mean error > threshold
                print("Using BTree")
                # 构建Btree并插入数据
                self.index[stage_length - 1][i] = BTree(DEGREE)
                self.index[stage_length - 1][i].bulk_load(tmp_inputs[stage_length - 1][i], tmp_labels[stage_length - 1][i])

    # # hybrid linear training structure, 2 stages
    # def hybrid_linear_training(self, threshold, stage, stage_nums, train_data_x, train_data_y, test_data_x, test_data_y, model):
    #     # stage_nums 为一个列表，存储着每个 stage 中 model 的个数
    #     stage_length = stage  # stage_length 表示 stage 的个数
    #     # 格式【stage个数个【】,其中包含model个数个【】】
    #     tmp_inputs = [[[] for j in range(stage_nums[i])] for i in range(stage_length)]
    #     tmp_labels = [[[] for j in range(stage_nums[i])] for i in range(stage_length)]
    #     # 放入对应的输入和标签
    #     tmp_inputs[0][0] = train_data_x  # 把数据全塞入第一个stage中的模型
    #     tmp_labels[0][0] = train_data_y
    #     for i in range(0, stage_length):
    #         print("the stage ", i, " is training")
    #         for j in range(0, stage_nums[i]):
    #             TOTAL_NUMBER = len(tmp_inputs[i][j])  # 获取第i层stage第j个model中数据的数量
    #             if TOTAL_NUMBER == 0:  # 这个模型中没有数据，直接跳过
    #                 continue
    #             inputs = tmp_inputs[i][j]
    #             labels = []
    #             test_labels = []
    #             if i <= stage_length - 2:
    #                 # 此层中对应的位置 / 此层中总共的数据数量 * 下一层model的个数
    #                 divisor = stage_nums[i + 1] * 1.0 / TOTAL_NUMBER   # eg：此层共1000个数，下一层10个model，即前100个数应划分至下一层的第一个model
    #                 # 将label转换成对应的下一层所对应stage的位置
    #                 for k in tmp_labels[i][j]:
    #                     labels.append(int(k * divisor))
    #                 for k in test_data_y:
    #                     test_labels.append(int(k * divisor))
    #             else:
    #                 # 在最后一层stage中，无需将label转换成下一层所对应stage的位置，只需为原始的label即可
    #                 labels = tmp_labels[i][j]
    #             # train model
    #             print("the model ", j, " is training")
    #             # 基于最小二乘法的线性回归
    #             if model == "linear":
    #                 tmp_index = LinearModel(LinearRegression(), inputs, labels)
    #             # 逻辑回归
    #             elif model == "logistic":
    #                 tmp_index = LinearModel(LogisticRegression(), inputs, labels)
    #             tmp_index.train()
    #             tmp_index.set_error(inputs, labels)
    #             # get parameters in model (weight matrix and bias matrix)
    #             self.index[i][j] = tmp_index
    #             del tmp_index
    #             gc.collect()  # 垃圾回收
    #             # 存在下一个stage
    #             if i < stage_length - 1:
    #                 # allocate data into training set for models in next stage
    #                 # 第一个stage中的数据为tmp_inputs，为其中的每一个数据预测其在下一个stage中的位置，并将其放入
    #                 for ind in range(len(tmp_inputs[i][j])):
    #                     # pick model in next stage with output of this model
    #                     # 预测每一个数据对应的下一个stage位置
    #                     p = self.index[i][j].predict(tmp_inputs[i][j][ind])
    #                     if p > stage_nums[i + 1] - 1:
    #                         p = stage_nums[i + 1] - 1
    #                     # 放入下一个stage的数据
    #                     tmp_inputs[i + 1][p].append(tmp_inputs[i][j][ind])
    #                     tmp_labels[i + 1][p].append(tmp_labels[i][j][ind])
    #
    #     # 最后一层的每个stage
    #     for i in range(stage_nums[stage_length - 1]):
    #         # 这个stage中没有数据
    #         if self.index[stage_length - 1][i] is None:
    #             continue
    #         mean_abs_err = self.index[stage_length - 1][i].mean_err
    #         # stage的损失大于阈值，则用btree代替
    #         if mean_abs_err > threshold[stage_length - 1]:
    #             # replace model with BTree if mean error > threshold
    #             print("Using BTree")
    #             # 构建Btree并插入数据
    #             self.index[stage_length - 1][i] = BTree(DEGREE)
    #             self.index[stage_length - 1][i].bulk_load(tmp_inputs[stage_length - 1][i], tmp_labels[stage_length - 1][i])

    def bulk_load(self, the_key, the_value):
        self.keys = the_key
        self.kv_index = [i for i in range(len(the_key))]
        self.values = the_value

        # train index
        if self.model == "neural_net":
            self.hybrid_training(thresholdPool[self.distribution], useThresholdPool[self.distribution], self.stage, self.stage_set, self.parameter.core_set,
                             self.parameter.train_step_set, self.parameter.batch_size_set, self.parameter.learning_rate_set,
                             self.keys, self.kv_index, [], [])
        else:
            self.hybrid_linear_training(thresholdPool[self.distribution], self.stage, self.stage_set, self.keys, self.kv_index, [], [], self.model)

    def range_search(self, k1, k2):
        whole_item = []
        # 首先判断范围查询的前后值是否处在不同的模型中
        pre_k1 = self.get_final_model(k1)
        pre_k2 = self.get_final_model(k2)
        # 在同一个模型中进行寻找
        if pre_k1 == pre_k2:
            if isinstance(self.index[self.stage - 1][pre_k1], BTree):  # 最后一层为 B-Tree
                return self.index[self.stage - 1][pre_k1].range_search(k1, k2)
            else:
                pos_k1 = self.index[self.stage - 1][pre_k1].predict(k1)
                pos_k2 = self.index[self.stage - 1][pre_k1].predict(k2)

                # 找到第一个大于等于k1的位置
                if pos_k1 >= len(self.keys):
                    pos_k1 = len(self.keys) - 1
                if pos_k1 < 0:
                    pos_k1 = 0
                if self.keys[pos_k1] >= k1:
                    while self.keys[pos_k1] >= k1:
                        if pos_k1 == 0:
                            break
                        else:
                            pos_k1 -= 1
                    if self.keys[pos_k1] < k1:
                        pos_k1 += 1
                else:
                    while self.keys[pos_k1] < k1:
                        if pos_k1 == len(self.keys) - 1:
                            break
                        else:
                            pos_k1 += 1

                # 找到最后一个小于等于k2的位置
                if pos_k2 >= len(self.keys):
                    pos_k2 = len(self.keys) - 1
                if pos_k2 < 0:
                    pos_k2 = 0
                if self.keys[pos_k2] > k2:
                    while self.keys[pos_k2] > k2:
                        if pos_k2 == 0:
                            break
                        else:
                            pos_k2 -= 1
                else:
                    while self.keys[pos_k2] <= k2:
                        if pos_k2 == len(self.keys) - 1:
                            break
                        else:
                            pos_k2 += 1
                    if self.keys[pos_k2] > k2:
                        pos_k2 -= 1

                # 获取范围之内的item值
                for i in range(pos_k1,pos_k2 + 1):
                    whole_item.append(Item(self.keys[i],self.values[i]))
                return whole_item
        # 在不同模型中寻找
        else:
            # 获取第一个模型中的item值
            if isinstance(self.index[self.stage - 1][pre_k1], BTree):  # 最后一层为 B-Tree
                whole_item = whole_item + list(self.index[self.stage - 1][pre_k1].range_search(k1, k2))
            else:
                pos_k1 = self.index[self.stage - 1][pre_k1].predict(k1)
                # 找到第一个大于等于k1的位置
                if pos_k1 >= len(self.keys):
                    pos_k1 = len(self.keys) - 1
                if pos_k1 < 0:
                    pos_k1 = 0
                if self.keys[pos_k1] >= k1:
                    while self.keys[pos_k1] >= k1:
                        if pos_k1 == 0:
                            break
                        else:
                            pos_k1 -= 1
                    if self.keys[pos_k1] < k1:
                        pos_k1 += 1
                else:
                    while self.keys[pos_k1] < k1:
                        if pos_k1 == len(self.keys) - 1:
                            break
                        else:
                            pos_k1 += 1
                for j in range(pos_k1, self.last_item_index[pre_k1][len(self.last_item_index[pre_k1]) - 1] + 1):
                    whole_item.append(Item(self.keys[j],self.values[j]))

            # 获取中间模型的所有item值
            if pre_k2 != pre_k1 + 1:
                for k in range(pre_k1 + 1, pre_k2):
                    if isinstance(self.index[self.stage - 1][k], BTree):  # 最后一层为 B-Tree
                        whole_item = whole_item + list(self.index[self.stage - 1][k].range_search(k1, k2))
                    else:
                        for j in self.last_item_index[k]:
                            whole_item.append(Item(self.keys[j], self.values[j]))

            # 获取最后一个模型中的item值
            if isinstance(self.index[self.stage - 1][pre_k2], BTree):  # 最后一层为 B-Tree
                whole_item = whole_item + list(self.index[self.stage - 1][pre_k2].range_search(k1, k2))
            else:
                pos_k2 = self.index[self.stage - 1][pre_k2].predict(k2)
                # 找到最后一个小于等于k2的位置
                if pos_k2 >= len(self.keys):
                    pos_k2 = len(self.keys) - 1
                if pos_k2 < 0:
                    pos_k2 = 0
                if self.keys[pos_k2] > k2:
                    while self.keys[pos_k2] > k2:
                        if pos_k2 == 0:
                            break
                        else:
                            pos_k2 -= 1
                else:
                    while self.keys[pos_k2] <= k2:
                        if pos_k2 == len(self.keys) - 1:
                            break
                        else:
                            pos_k2 += 1
                    if self.keys[pos_k2] > k2:
                        pos_k2 -= 1
                for j in range(self.last_item_index[pre_k2][0], pos_k2 + 1):
                    whole_item.append(Item(self.keys[j],self.values[j]))
            return whole_item


    # 获取关键字所在的最后一层模型
    def get_final_model(self, key):
        pre = 0
        for i in range(self.stage - 1):
            # pick model in next stage
            pre = self.index[i][pre].predict(key)
            if pre < 0:
                pre = 0
            if pre > self.stage_set[i + 1] - 1:
                pre = self.stage_set[i + 1] - 1

        if self.index[self.stage - 1][pre] == None:
            # 存在最后一层所预测的model中没有数据的情况
            out_of_bound = False
            final_position = pre
            while self.index[self.stage - 1][final_position] == None:
                if final_position == 0:
                    out_of_bound = True
                    break
                final_position -= 1
            if out_of_bound == True:
                while self.index[self.stage - 1][pre] == None:
                    pre += 1
            else:
                pre = final_position

        return pre

    def insert(self, key, value):
        pre = 0
        for i in range(self.stage - 1):
            # pick model in next stage
            pre = self.index[i][pre].predict(key)
            if pre < 0:
                pre = 0
            if pre > self.stage_set[i + 1] - 1:
                pre = self.stage_set[i + 1] - 1

        if self.index[self.stage - 1][pre] == None:
            # 存在最后一层所预测的model中没有数据的情况
            out_of_bound = False
            final_position = pre
            while self.index[self.stage - 1][final_position] == None:
                if final_position == 0:
                    out_of_bound = True
                    break
                final_position -= 1
            if out_of_bound == True:
                while self.index[self.stage - 1][pre] == None:
                    pre += 1
            else:
                pre = final_position

        if isinstance(self.index[self.stage - 1][pre], BTree):  # 最后一层为 B-Tree
            value, index = self.index[self.stage - 1][pre].predict(key)
            if index >= 0:    # B-Tree中存在这个key
                return -1
            else:
                self.index[self.stage - 1][pre].insert(Item(key,value))
                return 0
        else:  # 最后一层为 Learned Index
            first_position = final_position = self.index[self.stage - 1][pre].predict(key)
            if final_position >= len(self.keys):
                final_position = len(self.keys) - 1
            if final_position < 0:
                final_position = 0
            while True:
                if final_position == len(self.keys) and key > self.keys[len(self.keys) - 1]:
                    break
                if final_position == 0 and key < self.keys[0]:
                    break
                # learned index中存在这个key
                if self.keys[final_position] == key:
                    return -1
                if key > self.keys[final_position]:
                    final_position += 1
                elif key < self.keys[final_position - 1]:
                    final_position -= 1
                else:
                    break
            error = abs(final_position - first_position)
            self.keys.insert(final_position, key)
            self.kv_index.append(len(self.kv_index))
            self.values.insert(final_position, value)
            return error

    def search(self, key):
        pre = 0
        for i in range(self.stage - 1):
            # pick model in next stage
            pre = self.index[i][pre].predict(key)
            if pre < 0:
                pre = 0
            if pre > self.stage_set[i + 1] - 1:
                pre = self.stage_set[i + 1] - 1

        if self.index[self.stage - 1][pre] == None:
            # 存在最后一层所预测的model中没有数据的情况
            out_of_bound = False
            final_position = pre
            while self.index[self.stage - 1][final_position] == None:
                if final_position == 0:
                    out_of_bound = True
                    break
                final_position -= 1
            if out_of_bound == True:
                while self.index[self.stage - 1][pre] == None:
                    pre += 1
            else:
                pre = final_position

        # predict the final stage position
        if isinstance(self.index[self.stage - 1][pre], BTree):  # 最后一层为 B-Tree
            # 预测关键字是否存在，以及在节点中遍历的error
            return self.index[self.stage - 1][pre].predict(key)
        else:  # 最后一层为 Learned Index
            first_position = final_position = self.index[self.stage - 1][pre].predict(key)
            if final_position >= len(self.keys):
                final_position = len(self.keys) - 1
            if final_position < 0:
                final_position = 0
            begin_bound = end_bound = False
            if self.keys[final_position] != key:
                flag = 1
                off = 1
                # 从预测错误的位置，先向右寻找一位，再向左寻找两位，从而不断寻找左右两边的位置，直到找到正确的位置为止
                while self.keys[final_position] != key:
                    if final_position == 0:
                        begin_bound = True
                    if final_position == len(self.keys) - 1:
                        end_bound = True

                    if begin_bound == True:
                        if final_position != len(self.keys) - 1:
                            final_position += 1
                        else:
                            return 0, -1
                    elif end_bound == True:
                        if final_position != 0:
                            final_position -= 1
                        else:
                            return 0, -1
                    else:
                        final_position += flag * off
                        flag = -flag
                        off += 1
            return self.values[final_position], abs(final_position - first_position)

    def delete(self, key):
        pre = 0
        for i in range(self.stage - 1):
            # pick model in next stage
            pre = self.index[i][pre].predict(key)
            if pre < 0:
                pre = 0
            if pre > self.stage_set[i + 1] - 1:
                pre = self.stage_set[i + 1] - 1

        if self.index[self.stage - 1][pre] == None:
            # 存在最后一层所预测的model中没有数据的情况
            out_of_bound = False
            final_position = pre
            while self.index[self.stage - 1][final_position] == None:
                if final_position == 0:
                    out_of_bound = True
                    break
                final_position -= 1
            if out_of_bound == True:
                while self.index[self.stage - 1][pre] == None:
                    pre += 1
            else:
                pre = final_position

        if isinstance(self.index[self.stage - 1][pre], BTree):  # 最后一层为 B-Tree
            value, index = self.index[self.stage - 1][pre].predict(key)
            if index < 0:  # B-Tree中不存在这个key
                return -1
            else:
                self.index[self.stage - 1][pre].remove(Item(key, value))
                return 0
        else:  # 最后一层为 Learned Index
            final_position = self.index[self.stage - 1][pre].predict(key)
            if final_position >= len(self.keys):
                final_position = len(self.keys) - 1
            if final_position < 0:
                final_position = 0
            begin_bound = end_bound = False
            if self.keys[final_position] != key:
                flag = 1
                off = 1
                # 从预测错误的位置，先向右寻找一位，再向左寻找两位，从而不断寻找左右两边的位置，直到找到正确的位置为止
                while self.keys[final_position] != key:
                    if final_position == 0:
                        begin_bound = True
                    if final_position == len(self.keys) - 1:
                        end_bound = True

                    if begin_bound == True:
                        if final_position != len(self.keys) - 1:
                            final_position += 1
                        else:
                            return -1
                    elif end_bound == True:
                        if final_position != 0:
                            final_position -= 1
                        else:
                            return -1
                    else:
                        final_position += flag * off
                        flag = -flag
                        off += 1
            del self.keys[final_position]
            del self.kv_index[len(self.kv_index) - 1]
            del self.values[final_position]
            return 0

    def update(self, key, updated_value):
        pre = 0
        for i in range(self.stage - 1):
            # pick model in next stage
            pre = self.index[i][pre].predict(key)
            if pre > self.stage_set[i + 1] - 1:
                pre = self.stage_set[i + 1] - 1

        if self.index[self.stage - 1][pre] == None:
            # 存在最后一层所预测的model中没有数据的情况
            out_of_bound = False
            final_position = pre
            while self.index[self.stage - 1][final_position] != None:
                if final_position == 0:
                    out_of_bound = True
                    break
                final_position -= 1
            if out_of_bound == True:
                while self.index[self.stage - 1][pre] != None:
                    pre += 1
            else:
                pre = final_position

        if isinstance(self.index[self.stage - 1][pre], BTree):  # 最后一层为 B-Tree
            present_value, flag = self.index[self.stage - 1][pre].update(key, updated_value)
            return flag
        else:  # 最后一层为 Learned Index
            final_position = self.index[self.stage - 1][pre].predict(key)
            if final_position >= len(self.keys):
                final_position = len(self.keys) - 1
            if final_position < 0:
                final_position = 0
            begin_bound = end_bound = False
            if self.keys[final_position] != key:
                flag = 1
                off = 1
                # 从预测错误的位置，先向右寻找一位，再向左寻找两位，从而不断寻找左右两边的位置，直到找到正确的位置为止
                while self.keys[final_position] != key:
                    if final_position == 0:
                        begin_bound = True
                    if final_position == len(self.keys) - 1:
                        end_bound = True

                    if begin_bound == True:
                        if final_position != len(self.keys) - 1:
                            final_position += 1
                        else:
                            return -1
                    elif end_bound == True:
                        if final_position != 0:
                            final_position -= 1
                        else:
                            return -1
                    else:
                        final_position += flag * off
                        flag = -flag
                        off += 1
            self.values[final_position] = updated_value
            return 0


def data_processing(distribution_bulk_load, distribution_insert, flag, num):
    the_value = [1.0 * i / BLOCK_SIZE for i in range(num)]
    if distribution_bulk_load == distribution_insert:
        # 生成两倍服从相应分布的数据
        whole_data = create_data_no_storage(distribution_bulk_load,num * 2)

        if flag == 0:
            # 对字典的key进行排序
            bulk_load_key = sorted(whole_data[:num])
            insert_key = sorted(whole_data[num:num * 2])
        else:
            # 对字典的key进行排序
            bulk_load_key = sorted(whole_data[num:num * 2])
            insert_key = sorted(whole_data[:num])
    else:
        whole_bulk_load_data = create_data_no_storage(distribution_bulk_load,num)
        bulk_load_key = sorted(whole_bulk_load_data)
        whole_insert_data = create_data_no_storage(distribution_insert,num)
        insert_key = sorted(whole_insert_data)

    return bulk_load_key,the_value,insert_key,the_value

# help message
def show_help_message(msg):
    help_message = {'command': 'python Learned_BTree.py -l <Distribution> -i <Distribution> -c <Compare> -n <Number> -s <Stage> -m <Model>',
                    'l': 'Distribution for bulk load',
                    'i': 'Distribution for insert',
                    'compare': 'Whether the data for insert larger than the data for bulk load, default value = False',
                    'distribution': 'Distribution: linear, random, exponential, normal, lognormal',
                    'number': 'Number: 10,000-1,000,000, default value = 300,000',
                    'stage': 'Stage: 2-3, default value = 2',
                    'model': 'Model: linear, logistic, neural_net, default value = linear',
                    'noDistributionLoadError': 'Please choose the distribution for bulk load data first',
                    'noDistributionInsertError': 'Please choose the distribution for insert data first.'}
    help_message_key = ['command', 'l', 'i', 'compare', 'distribution', 'number', 'stage', 'model']
    if msg == 'all':
        for k in help_message_key:
            print(help_message[k])

    else:
        print(help_message['command'])
        print('Error! ' + help_message[msg])

# command line
def main(argv):
    distribution_bulk_load = None
    distribution_insert = None
    num = 300000
    stage = 2
    model = "linear"
    is_distribution_bulk_load = False
    is_distribution_insert = False
    is_larger = True
    try:
        # 处理传入的参数内容
        opts, args = getopt.getopt(argv, "hd:l:i:c:n:s:m:")
    except getopt.GetoptError:
        show_help_message('command')
        sys.exit(2)
    for opt, arg in opts:
        arg = str(arg).lower()
        # 显示 help 的帮助信息
        if opt == '-h':
            show_help_message('all')
            return

        # -l <Distribution> -i <Distribution> -c <Compare> -n <Number> -s <Stage> -m <Model>
        # Parameters:
        # 'l': 'Distribution for bulk load',
        # 'i': 'Distribution for insert',
        # 'compare': 'Whether the data for insert larger than the data for bulk load',
        # 'distribution': 'Distribution: linear, random, exponential, normal, lognormal',
        # 'number': 'Number: 10,000-10,000,000, default value = 300,000',
        # 'stage': 'Stage: 2-3, default value = 2'
        #  'model': 'Model: linear, logistic, neural_net, default value = linear'

        elif opt == '-l':
            if arg == "linear":
                distribution_bulk_load = Distribution.LINEAR
                is_distribution_bulk_load = True
            elif arg == "random":
                distribution_bulk_load = Distribution.RANDOM
                is_distribution_bulk_load = True
            elif arg == "exponential":
                distribution_bulk_load = Distribution.EXPONENTIAL
                is_distribution_bulk_load = True
            elif arg == "normal":
                distribution_bulk_load = Distribution.NORMAL
                is_distribution_bulk_load = True
            elif arg == "lognormal":
                distribution_bulk_load = Distribution.LOGNORMAL
                is_distribution_bulk_load = True
            elif arg == "wiki":
                distribution_bulk_load = Distribution.WIKI
                is_distribution_bulk_load = True
            elif arg == "osm":
                distribution_bulk_load = Distribution.OSM
                is_distribution_bulk_load = True
            elif arg == "books":
                distribution_bulk_load = Distribution.BOOKS
                is_distribution_bulk_load = True
            elif arg == "fb":
                distribution_bulk_load = Distribution.FB
                is_distribution_bulk_load = True
            else:
                show_help_message('l')
                show_help_message('distribution')
                return

        elif opt == '-i':
            if arg == "linear":
                distribution_insert = Distribution.LINEAR
                is_distribution_insert = True
            elif arg == "random":
                distribution_insert = Distribution.RANDOM
                is_distribution_insert = True
            elif arg == "exponential":
                distribution_insert = Distribution.EXPONENTIAL
                is_distribution_insert = True
            elif arg == "normal":
                distribution_insert = Distribution.NORMAL
                is_distribution_insert = True
            elif arg == "lognormal":
                distribution_insert = Distribution.LOGNORMAL
                is_distribution_insert = True
            elif arg == "wiki":
                distribution_insert = Distribution.WIKI
                is_distribution_insert = True
            elif arg == "osm":
                distribution_insert = Distribution.OSM
                is_distribution_insert = True
            elif arg == "books":
                distribution_insert = Distribution.BOOKS
                is_distribution_insert = True
            elif arg == "fb":
                distribution_insert = Distribution.FB
                is_distribution_insert = True
            else:
                show_help_message('i')
                show_help_message('distribution')
                return

        elif opt == '-c':
            if not is_distribution_bulk_load:
                show_help_message('noDistributionLoadError')
                return
            if not is_distribution_insert:
                show_help_message('noDistributionInsertError')
                return
            if arg == "false":
                is_larger = False

        elif opt == '-n':
            if not is_distribution_bulk_load:
                show_help_message('noDistributionLoadError')
                return
            if not is_distribution_insert:
                show_help_message('noDistributionInsertError')
                return
            num = int(arg)
            if not 10 <= num <= 10000000:
                show_help_message('number')
                return

        elif opt == '-s':
            if not is_distribution_bulk_load:
                show_help_message('noDistributionLoadError')
                return
            if not is_distribution_insert:
                show_help_message('noDistributionInsertError')
                return
            stage = int(arg)
            if not 2 <= stage <= 3:
                show_help_message('stage')
                return

        elif opt == '-m':
            if arg == "logistic":
                model = "logistic"
            elif arg == "neural_net":
                model = "neural_net"

        else:
            print("Unknown parameters, please use -h for instructions.")
            return

    if not is_distribution_bulk_load:
        show_help_message('noDistributionLoadError')
        return
    if not is_distribution_insert:
        show_help_message('noDistributionInsertError')
        return

    train_index(stage, distribution_bulk_load, distribution_insert, is_larger, num, model)


def train_index(stage, distribution_bulk_load, distribution_insert, is_larger, num, model):
    if is_larger:
        # 插入的数据比批量加载的数据大，则批量加载数据取前num条，插入数据取后num条
        bulk_load_flag = 0
        insert_flag = 1
    else:
        # 插入的数据比批量加载的数据小，则批量加载数据取后num条，插入数据取前num条
        bulk_load_flag = 1
        insert_flag = 0

    # 批量加载和插入操作数据
    load_keys, load_values, insert_keys, insert_values = data_processing(distribution_bulk_load, distribution_insert, bulk_load_flag, num)
    # print(load_keys)

    """
    -------  LEARNED INDEX  --------
    """
    print("-------  LEARNED INDEX  --------")
    learned_index = Learned_Index(stage, distribution_bulk_load, model)

    # 测试索引构建时间
    print("*************start Learned NN************")
    print("Start Train")
    load_start_time = time.time()
    learned_index.bulk_load(load_keys, load_values)
    load_end_time = time.time()
    learn_time = load_end_time - load_start_time
    print("Build Learned NN time ", learn_time)
    print("*************end Learned NN************\n")

    # 测试查找操作时间
    print("*************start Search Key************")
    print("Calculate Time And Error")
    search_start_time = time.time()
    search_error = 0
    for the_key in load_keys:
        value, search_err = learned_index.search(the_key)
        if search_err < 0:
            print("We Can Not Find The Key!")
        else:
            search_error += search_err
    search_end_time = time.time()
    search_time = (search_end_time - search_start_time) / len(load_keys)
    print("Search time %f " % search_time)
    search_mean_error = search_error * 1.0 / len(load_keys)
    print("mean search error = ", search_mean_error)
    print("*************end Search Key************\n")

    # 测试范围查找操作时间
    print("*************start Range Search Key************")
    print("Calculate Time And Error")
    range_search_start_time = time.time()
    range_search_result = learned_index.range_search(load_keys[0], load_keys[num - 1])
    range_search_end_time = time.time()
    range_search_time = (range_search_end_time - range_search_start_time)
    print("Range Search time %f " % range_search_time)
    print("*************end Range Search Key************\n")

    # 测试存储空间大小
    # write parameter into files
    result_stage1 = {0: {"weights": learned_index.index[0][0].weights, "bias": learned_index.index[0][0].bias}}
    result_stage2 = {}
    for ind in range(len(learned_index.index[1])):
        if learned_index.index[1][ind] is None:
            continue
        if isinstance(learned_index.index[1][ind], BTree):
            tmp_result = []
            tmp = learned_index.index[1][ind].root.__str__()
            tmp_result.append(tmp)
            result_stage2[ind] = tmp_result
        else:
            result_stage2[ind] = {"weights": learned_index.index[1][ind].weights,
                                  "bias": learned_index.index[1][ind].weights,
                                  "positions": learned_index.last_item_index[ind]}
    result = [{"stage": 1, "parameters": result_stage1}, {"stage": 2, "parameters": result_stage2}]

    with open("model/" + pathString[distribution_bulk_load] + "/full_train/NN/" + str(num) + ".json", "w") as jsonFile:
        json.dump(result, jsonFile)

    print(os.path.getsize("model/" + pathString[distribution_bulk_load] + "/full_train/NN/" + str(num) + ".json"))

    # 测试插入操作时间
    print("*************start Insert Key************")
    print("Calculate Time And Error")
    insert_start_time = time.time()
    insert_error = 0
    for the_index in range(len(insert_keys)):
        err = learned_index.insert(insert_keys[the_index], insert_values[the_index])
        if err < 0 :
            print("The Key Is Duplicatied！")
        else:
            insert_error += err
    insert_end_time = time.time()
    insert_time = (insert_end_time - insert_start_time) / len(insert_keys)
    print("Insert time %f " % insert_time)
    insert_mean_error = insert_error * 1.0 / len(insert_keys)
    print("mean insert error = ", insert_mean_error)
    print("*************end Insert Key************\n")

    # # 判断是否成功插入数据
    # for the_key in insert_keys:
    #     value, search_err = learned_index.search(the_key)
    #     if search_err < 0:
    #         print("We Can Not Find The Key!")

    #测试查找操作时间
    print("*************start Search Key************")
    print("Calculate Time And Error")
    search_start_time = time.time()
    search_error = 0
    for the_key in load_keys:
        value, search_err = learned_index.search(the_key)
        if search_err < 0:
            print("We Can Not Find The Key!")
        else:
            search_error += search_err
    search_end_time = time.time()
    search_time = (search_end_time - search_start_time) / len(load_keys)
    print("Search time %f " % search_time)
    search_mean_error = search_error * 1.0 / len(load_keys)
    print("mean search error = ", search_mean_error)
    print("*************end Search Key************\n")

    # 测试删除操作时间
    print("*************start Delete Key************")
    print("Calculate Time And Error")
    delete_start_time = time.time()
    for the_key in insert_keys:
        delete_err = learned_index.delete(the_key)
        if delete_err == -1:
            print("There Is No Key!")
    delete_end_time = time.time()
    delete_time = (delete_end_time - delete_start_time) / len(insert_keys)
    print("Delete time %f " % delete_time)
    print("*************end Delete Key************\n")

    # # 判断是否成功删除数据
    # delete_count = 0
    # for the_key in insert_keys:
    #     value, search_err = learned_index.search(the_key)
    #     if search_err < 0:
    #         delete_count += 1
    # print(delete_count)

    """
    -------  B-TREE  --------
    """
    print("-------  B-TREE  --------")
    bt = BTree(DEGREE)
    # 测试索引构建时间
    print("*************start Build B-Tree************")
    print("Start Build")
    build_start_time = time.time()
    bt.build(load_keys, load_values)
    build_end_time = time.time()
    build_time = build_end_time - build_start_time
    print("Build B-Tree time ", build_time)
    print("*************end Build B-Tree************\n")

    # 测试查找操作时间
    print("*************start Search Key************")
    print("Calculate Time And Error")
    search_start_time = time.time()
    search_error = 0
    for the_key in load_keys:
        value, search_err = bt.predict(the_key)
        if search_err < 0:
            print("We Can Not Find The Key!")
        else:
            search_error += search_err
    search_end_time = time.time()
    search_time = (search_end_time - search_start_time) / len(load_keys)
    print("Search time %f " % search_time)
    search_mean_error = search_error * 1.0 / len(load_keys)
    print("mean search error = ", search_mean_error)
    print("*************end Search Key************\n")

    # 测试范围查找操作时间
    print("*************start Range Search Key************")
    print("Calculate Time And Error")
    range_search_start_time = time.time()
    range_search_result = bt.range_search(load_keys[0], load_keys[num - 1])
    range_search_end_time = time.time()
    range_search_time = (range_search_end_time - range_search_start_time)
    print("Range Search time %f " % range_search_time)
    print("*************end Range Search Key************\n")

    # 测试存储空间大小
    # write BTree into files
    result = []
    tmp = bt.root.__str__()
    result.append(tmp)

    with open("model/" + pathString[distribution_bulk_load] + "/full_train/BTree/" + str(num) + ".json",
              "w") as jsonFile:
        json.dump(result, jsonFile)

    print(os.path.getsize("model/" + pathString[distribution_bulk_load] + "/full_train/BTree/" + str(num) + ".json"))

    # 测试插入操作时间
    print("*************start Insert Key************")
    print("Calculate Time And Error")
    insert_start_time = time.time()
    for the_index in range(len(insert_keys)):
        flag = bt.insert(Item(insert_keys[the_index], insert_values[the_index]))
        if flag == False:
            print("The Key Is Duplicatied！")
    insert_end_time = time.time()
    insert_time = (insert_end_time - insert_start_time) / len(insert_keys)
    print("Insert time %f " % insert_time)
    print("*************end Insert Key************\n")

    # # 判断是否成功插入数据
    # for the_key in insert_keys:
    #     value, search_err = bt.predict(the_key)
    #     if search_err < 0:
    #         print("We Can Not Find The Key!")

    # 测试查找操作时间
    print("*************start Search Key************")
    print("Calculate Time And Error")
    search_start_time = time.time()
    search_error = 0
    for the_key in load_keys:
        value, search_err = bt.predict(the_key)
        if search_err < 0:
            print("We Can Not Find The Key!")
        else:
            search_error += search_err
    search_end_time = time.time()
    search_time = (search_end_time - search_start_time) / len(load_keys)
    print("Search time %f " % search_time)
    search_mean_error = search_error * 1.0 / len(load_keys)
    print("mean search error = ", search_mean_error)
    print("*************end Search Key************\n")

    # 测试删除操作时间
    print("*************start Delete Key************")
    print("Calculate Time And Error")
    delete_start_time = time.time()
    for the_index in range(len(insert_keys)):
        delete_flag = bt.remove(Item(insert_keys[the_index], insert_values[the_index]))
        if delete_flag == False:
            print("There Is No Key!")
    delete_end_time = time.time()
    delete_time = (delete_end_time - delete_start_time) / len(insert_keys)
    print("Delete time %f " % delete_time)
    print("*************end Delete Key************\n")

    # # 判断是否成功删除数据
    # delete_count = 0
    # for the_key in insert_keys:
    #     value, search_err = bt.predict(the_key)
    #     if search_err < 0:
    #         delete_count += 1
    # print(delete_count)

if __name__ == "__main__":
    args = ['-l', 'linear', '-i', 'linear', '-c', 'False', '-n', '1000000', '-s', '2', '-m', 'neural_net']
    main(args)
    # main(sys.argv[1:])
