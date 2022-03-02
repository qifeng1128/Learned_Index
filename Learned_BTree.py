from __future__ import print_function

import random

import pandas as pd
from Trained_NN import TrainedNN, AbstractNN, ParameterPool, set_data_type
from btree import BTree,Item
from data import create_data, Distribution
import time, gc, json
import os, sys, getopt
import numpy as np

BLOCK_SIZE = 4096
MAX_SUB_NUM = int(BLOCK_SIZE / 8)
DEGREE = int((MAX_SUB_NUM + 1) / 2)

MAX_NUMBER = 10000000

filePath = {
    Distribution.LINEAR: "data/linear.csv",
    Distribution.RANDOM: "data/random.csv",
    Distribution.BINOMIAL: "data/binomial.csv",
    Distribution.POISSON: "data/poisson.csv",
    Distribution.EXPONENTIAL: "data/exponential.csv",
    Distribution.NORMAL: "data/normal.csv",
    Distribution.LOGNORMAL: "data/lognormal.csv"
}

# result record path
pathString = {
    Distribution.LINEAR: "Linear",
    Distribution.RANDOM: "Random",
    Distribution.BINOMIAL: "Binomial",
    Distribution.POISSON: "Poisson",
    Distribution.EXPONENTIAL: "Exponential",
    Distribution.NORMAL: "Normal",
    Distribution.LOGNORMAL: "Lognormal"
}

# threshold for train (judge whether stop train and replace with BTree)
thresholdPool = {
    Distribution.LINEAR: [1, 2, 2, 2, 2],
    Distribution.RANDOM: [1, 4, 4, 4, 4],
    Distribution.EXPONENTIAL: [55, 10000, 10000, 10000, 10000],
    Distribution.NORMAL: [10, 100, 100, 100, 100],
    Distribution.LOGNORMAL: [55, 10000, 10000, 10000, 10000]
}

# whether use threshold to stop train for models in stages
useThresholdPool = {
    Distribution.LINEAR: [True, False, False, False, False],
    Distribution.RANDOM: [True, False, False, False, False],
    Distribution.EXPONENTIAL: [True, False, False, False, False],
    Distribution.NORMAL: [True, False, False, False, False],
    Distribution.LOGNORMAL: [True, False, False, False, False]
}

def pos_binary_search(data_list, key):
    start = 0
    end = len(data_list) - 1
    mid = 0
    while start <= end:
        mid = int((start + end) / 2)
        if data_list[mid] == key or data_list[mid] == -1:
            return mid
        elif data_list[mid] < key:
            start = mid + 1
        else:
            end = mid - 1
    if data_list[mid] > key:
        return mid - 1
    else:
        return mid

class Learned_Index:
    def __init__(self, stage, distribution):
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
        else:
            raise ValueError('The Distribution Is Wrong.')
        self.stage = stage

        self.stage_set = self.parameter.stage_set
        # set number of models for the rest stage
        self.stage_set[1] = 10  # (1 model deal with 1000 records)
        self.stage_set[2] = 10  # (1 model deal with 100 records)
        self.stage_set[3] = 10  # (1 model deal with 10 records)
        self.stage_set[4] = 1

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
        # initial
        # 格式【stage个数个【】,其中包含model个数个【】】
        tmp_inputs = [[[] for j in range(stage_nums[i])] for i in range(stage_length)]
        tmp_labels = [[[] for j in range(stage_nums[i])] for i in range(stage_length)]
        # 放入对应的输入和标签
        tmp_inputs[0][0] = train_data_x  # 把数据全塞入第一个stage中的模型
        tmp_labels[0][0] = train_data_y
        test_inputs = test_data_x
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
                    divisor = stage_nums[i + 1] * 1.0 / TOTAL_NUMBER  # eg：此层共1000个数，下一层10个model，即前100个数应划分至下一层的第一个model
                    # 将label转换成对应的下一层所对应stage的位置
                    for k in tmp_labels[i][j]:
                        labels.append(int(k * divisor))
                    for k in test_data_y:
                        test_labels.append(int(k * divisor))
                else:
                    # 在最后一层stage中，无需将label转换成下一层所对应stage的位置，只需为原始的label即可
                    labels = tmp_labels[i][j]
                    test_labels = test_data_y
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
                self.index[stage_length - 1][i].build(tmp_inputs[stage_length - 1][i], tmp_labels[stage_length - 1][i])


    def data_processing(self, total_data, num):
        total_number = {}  # 存放key-value对的字典
        # 从总数据中抽取固定数量的数据集
        data = total_data.sample(num)
        the_rest = total_data[~total_data.index.isin(data.index)]
        for i in range(data.shape[0]):
            total_number[data.iloc[i, 0]] = data.iloc[i, 1]
            # train_set_x.append(data.ix[i, 0])
            # train_set_y.append(data.ix[i, 1])

        # 对字典的key进行排序
        the_key = sorted(total_number.keys())
        the_value = [total_number[i] for i in the_key]

        return the_key, the_value, the_rest

    def bulk_load(self, the_key, the_value):
        self.keys = the_key
        self.kv_index = [i for i in range(len(the_key))]
        self.values = the_value

        # train index
        self.hybrid_training(thresholdPool[self.distribution], useThresholdPool[self.distribution], self.stage, self.stage_set, self.parameter.core_set,
                             self.parameter.train_step_set, self.parameter.batch_size_set, self.parameter.learning_rate_set,
                             self.keys, self.kv_index, [], [])

    def insert(self, key, value):
        pre = 0
        for i in range(self.stage - 1):
            # pick model in next stage
            pre = self.index[i][pre].predict(key)
            if pre > self.stage_set[i + 1] - 1:
                pre = self.stage_set[i + 1] - 1
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
                # learned index中存在这个key
                if self.keys[final_position] == key:
                    return -1
                if final_position == len(self.keys) and key > self.keys[len(self.keys) - 1]:
                    break
                if final_position == 0 and key < self.keys[0]:
                    break
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
            if pre > self.stage_set[i + 1] - 1:
                pre = self.stage_set[i + 1] - 1
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
            if pre > self.stage_set[i + 1] - 1:
                pre = self.stage_set[i + 1] - 1
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
                self.keys.pop(final_position)
                self.kv_index.pop(len(self.kv_index) - 1)
                self.values.pop(final_position)
                return 0

# help message
def show_help_message(msg):
    help_message = {'command': 'python Learned_BTree.py -t <Type> -d <Distribution> [-p|-n] [Percent]|[Number] [-c] [New data] [-h]',
                    'type': 'Type: sample, full',
                    'distribution': 'Distribution: linear, random, exponential, normal, lognormal',
                    'percent': 'Percent: 0.1-1.0, default value = 0.5; sample train data size = 300,000',
                    'number': 'Number: 10,000-1,000,000, default value = 300,000',
                    'stage': 'Stage: 2-3, default value = 2',
                    'fpError': 'Percent cannot be assigned in full train.',
                    'snError': 'Number cannot be assigned in sample train.',
                    'noTypeError': 'Please choose the type first.',
                    'noDistributionError': 'Please choose the distribution first.'}
    help_message_key = ['command', 'type', 'distribution', 'percent', 'number']
    if msg == 'all':
        for k in help_message_key:
            print(help_message[k])

    else:
        print(help_message['command'])
        print('Error! ' + help_message[msg])

# command line
def main(argv):
    distribution = None
    per = 0.5
    num = 300000
    stage = 2
    is_sample = False
    is_type = False
    is_distribution = False
    try:
        # 处理传入的参数内容
        opts, args = getopt.getopt(argv, "hd:t:p:n:s:")
    except getopt.GetoptError:
        show_help_message('command')
        sys.exit(2)
    for opt, arg in opts:
        arg = str(arg).lower()
        # 显示 help 的帮助信息
        if opt == '-h':
            show_help_message('all')
            return
        # -t <Type> -d <Distribution> [-p|-n] [Percent]|[Number] [-h]

        # Parameters:
        # 'type': 'Type: sample, full',
        # 'distribution': 'Distribution: linear, random, exponential',
        # 'percent': 'Percent: 0.1-1.0, default value = 0.5; sample train data size = 300,000',
        # 'number': 'Number: 10,000-10,000,000, default value = 300,000'

        elif opt == '-t':
            if arg == "sample":
                is_sample = True
                is_type = True
            elif arg == "full":
                is_sample = False
                is_type = True
            else:
                show_help_message('type')
                return
        elif opt == '-d':
            if not is_type:
                show_help_message('noTypeError')
                return
            if arg == "linear":
                distribution = Distribution.LINEAR
                is_distribution = True
            elif arg == "random":
                distribution = Distribution.RANDOM
                is_distribution = True
            elif arg == "exponential":
                distribution = Distribution.EXPONENTIAL
                is_distribution = True
            elif arg == "normal":
                distribution = Distribution.NORMAL
                is_distribution = True
            elif arg == "lognormal":
                distribution = Distribution.LOGNORMAL
                is_distribution = True
            else:
                show_help_message('distribution')
                return
        elif opt == '-p':
            if not is_type:
                show_help_message('noTypeError')
                return
            if not is_distribution:
                show_help_message('noDistributionError')
                return
            per = float(arg)
            if not 0.1 <= per <= 1.0:
                show_help_message('percent')
                return

        elif opt == '-n':
            if not is_type:
                show_help_message('noTypeError')
                return
            if not is_distribution:
                show_help_message('noDistributionError')
                return
            if is_sample:
                show_help_message('snError')
                return
            num = int(arg)
            if not 10 <= num <= 1000000:
                show_help_message('number')
                return

        elif opt == '-s':
            if not is_type:
                show_help_message('noTypeError')
                return
            if not is_distribution:
                show_help_message('noDistributionError')
                return
            stage = int(arg)
            if not 2 <= stage <= 3:
                show_help_message('stage')
                return

        else:
            print("Unknown parameters, please use -h for instructions.")
            return

    if not is_type:
        show_help_message('noTypeError')
        return
    if not is_distribution:
        show_help_message('noDistributionError')
        return
    if is_sample:
        sample_train(thresholdPool[distribution], useThresholdPool[distribution], stage, distribution, per, filePath[distribution])
    else:
        train_index(stage, distribution, num)

def train_index(stage, distribution, num):
    learned_index = Learned_Index(stage, distribution)
    # 获取总的数据
    path = filePath[distribution]
    total_data = pd.read_csv(path, header=None)
    # 批量加载数据
    load_keys, load_values, the_rest = learned_index.data_processing(total_data, num)
    # print(load_keys)
    # 插入操作数据
    insert_keys, insert_values, the_rest = learned_index.data_processing(the_rest, num)
    # print(insert_keys)

    # 测试索引构建时间
    print("*************start Learned NN************")
    print("Start Train")
    load_start_time = time.time()
    learned_index.bulk_load(load_keys, load_values)
    load_end_time = time.time()
    learn_time = load_end_time - load_start_time
    print("Build Learned NN time ", learn_time)

    # 测试查找操作时间
    print("Calculate Error")
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
    print("mean error = ", search_mean_error)
    print("*************end Learned NN************\n\n")

    # 测试插入操作时间
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
    print("mean error = ", insert_mean_error)

    # 判断是否成功插入数据
    for the_key in insert_keys:
        value, search_err = learned_index.search(the_key)
        if search_err < 0:
            print("We Can Not Find The Key!")

    # 删除刚插入的元素
    for the_key in insert_keys:
        delete_err = learned_index.delete(the_key)
        if delete_err == -1:
            print("There Is No Key!")

    # 判断是否成功删除数据
    delete_count = 0
    for the_key in insert_keys:
        value, search_err = learned_index.search(the_key)
        if search_err < 0:
            delete_count += 1
    print(delete_count)

if __name__ == "__main__":
    args = ['-t', 'full', '-d', 'Linear', '-n', '10000', '-s', '2']
    # main(sys.argv[1:])
    main(args)