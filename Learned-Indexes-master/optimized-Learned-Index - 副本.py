# -*- coding: utf-8 -*-

# BTree Index with Python

import pandas as pd
import numpy as np
import time, gc, json
import os
from data import create_data_no_storage, Distribution

BLOCK_SIZE = 4096
MAX_SUB_NUM = int(BLOCK_SIZE / 8)
DEGREE = int((MAX_SUB_NUM + 1) / 2)

filePath = {
    "linear": "data/linear.csv",
    "random": "data/random.csv",
    "exponential": "data/exponential.csv",
    "normal": "data/normal.csv",
    "lognormal": "data/lognormal.csv",
    "wiki": "data/wiki_ts_200M_uint64.csv",
    "osm": "data/osm_cellids_200M_uint64.csv",
    "books": "data/books_200M_uint64.csv",
    "fb": "data/fb_200M_uint64.csv",
}

def data_processing(path, flag, num):
    total_data = pd.read_csv(path, header=None)
    total_number = {}  # 存放key-value对的字典
    # 从总数据中抽取固定数量的数据集
    if flag == 0:
        data = total_data[:num]
    else:
        data = total_data[-num:]
    for i in range(data.shape[0]):
        total_number[data.iloc[i, 0]] = data.iloc[i, 1]
        # train_set_x.append(data.ix[i, 0])
        # train_set_y.append(data.ix[i, 1])

    # 对字典的key进行排序
    the_key = sorted(total_number.keys())
    the_value = [total_number[i] for i in the_key]

    return the_key, the_value

# threshold for train (judge whether stop train and replace with BTree)
thresholdPool = {
    "linear": 5,
    "random": 4,
    "exponential": 10000,
    "exponential_lognormal": 10000,
    "normal": 100,
    "lognormal": 10000,
    "wiki": 4,
    "osm": 4,
    "books": 4,
    "fb": 4
}

# Leaf Node in BTree
class ModelNode:
    def __init__(self,number = 0, degree=2, slope = 1, bias = 0):    # 初始化
        # model的节点一定为叶子节点
        self.isLeaf = True
        # 包含item的个数
        self.number = number
        # 线性回归的斜率和截距
        self.slope = slope
        self.bias = bias
        # 模型的误差
        self.mean_error = 0
        # 关键字的值数组(个数为2d-1)
        self.items = [None] * (degree * 2 - 1)
        # 子结点数组(个数为2d)
        self.children = [None] * degree * 2

    def __str__(self):  # 通过字符串的形式直观展示b-tree节点中的内容
        returnStr = 'isLeaf:' + str(self.isLeaf) + ';' + 'number:' + str(self.number) + ';' + 'slope:' + str(self.slope) + ';' + 'bias:' + str(self.bias) + ';'
        returnStr += 'keys:['
        for i in range(self.number):
            returnStr += str(self.items[i].k) + ' '
        returnStr += '];values:['
        for i in range(self.number):
            returnStr += str(self.items[i].v) + ' '
        returnStr += '];childrens:['
        for child in self.children:
            returnStr += str(child) + ';'
        returnStr += ']'
        return returnStr

    @classmethod
    def allocate_node(self, degree):  # 为一个新结点分配一个磁盘页
        return ModelNode(degree=degree)

    def bulk_load(self,items):
        self.number = len(items)
        for i in range(self.number):
            self.items[i] = items[i]
        # 计算模型的斜率和截距
        keys = []
        for one_item in items:
            keys.append(one_item.k)
        labels = [i for i in range(self.number)]
        dinominator = 0    # 初始化分母
        numerator = 0      # 初始化分子
        for i in range(0, self.number):  # 求b1
            numerator += (keys[i] - np.mean(keys)) * (labels[i] - np.mean(labels))
            dinominator += (keys[i] - np.mean(keys)) ** 2  # **表示平方

        # print("numerator:" + str(numerator))
        # print("dinominator:" + str(dinominator))

        self.slope = numerator / float(dinominator)  # 得出b1
        self.bias = np.mean(labels) - self.slope * float(np.mean(keys))  # 得出b0

        error = 0
        for i in range(self.number):
            y_predict = self.slope * keys[i] + self.bias
            error += abs(y_predict - labels[i])
        self.mean_error = error / self.number * 1.0

    # 叶子节点查找关键字
    def predict(self, an_item):
        first_position = final_position = int(an_item.k * self.slope + self.bias)
        if final_position >= self.number:
            final_position = self.number - 1
        if final_position < 0:
            final_position = 0
        begin_bound = end_bound = False
        if self.items[final_position] != an_item:
            flag = 1
            off = 1
            # 从预测错误的位置，先向右寻找一位，再向左寻找两位，从而不断寻找左右两边的位置，直到找到正确的位置为止
            while self.items[final_position] != an_item:
                if final_position == 0:
                    begin_bound = True
                if final_position == self.number - 1:
                    end_bound = True

                if begin_bound == True:
                    if final_position != self.number - 1:
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
        return self.items[final_position].v, abs(final_position - first_position)

    def range_search(self, k1, k2, res):
        pos_k1 = int(k1 * self.slope + self.bias)
        pos_k2 = int(k2 * self.slope + self.bias)

        # 找到第一个大于等于k1的位置
        if pos_k1 >= self.number:
            pos_k1 = self.number - 1
        if pos_k1 < 0:
            pos_k1 = 0
        if self.items[pos_k1].k >= k1:
            while self.items[pos_k1].k >= k1:
                if pos_k1 == 0:
                    break
                else:
                    pos_k1 -= 1
            if self.items[pos_k1].k < k1:
                pos_k1 += 1
        else:
            while self.items[pos_k1].k < k1:
                if pos_k1 == self.number - 1:
                    break
                else:
                    pos_k1 += 1

        # 找到最后一个小于等于k2的位置
        if pos_k2 >= self.number:
            pos_k2 = self.number - 1
        if pos_k2 < 0:
            pos_k2 = 0
        if self.items[pos_k2].k > k2:
            while self.items[pos_k2].k > k2:
                if pos_k2 == 0:
                    break
                else:
                    pos_k2 -= 1
        else:
            while self.items[pos_k2].k <= k2:
                if pos_k2 == self.number - 1:
                    break
                else:
                    pos_k2 += 1
            if self.items[pos_k2].k > k2:
                pos_k2 -= 1

        # 获取范围之内的item值
        for i in range(pos_k1, pos_k2 + 1):
            if k1 <= self.items[i].k <= k2:
                res.add(self.items[i])

# Internal Node in BTree
class BTreeNode:
    def __init__(self, degree=2, number=0, isLeaf=True):  # 初始化
        # 是否为叶子节点
        self.isLeaf = isLeaf
        # 结点包含关键字的数量
        self.number = number
        # 关键字的值数组(个数为2d-1)
        self.items = [None] * (degree * 2 - 1)
        # 子结点数组(个数为2d)
        self.children = [None] * degree * 2

    def __str__(self):  # 通过字符串的形式直观展示b-tree节点中的内容
        returnStr = 'isLeaf:' + str(self.isLeaf) + ';' + 'number:' + str(self.number) + ';'
        returnStr += 'keys:['
        for i in range(self.number):
            returnStr += str(self.items[i].k) + ' '
        returnStr += '];values:['
        for i in range(self.number):
            returnStr += str(self.items[i].v) + ' '
        returnStr += '];childrens:['
        for child in self.children:
            returnStr += str(child) + ';'
        returnStr += ']'
        return returnStr

    @classmethod
    def allocate_node(self, degree):  # 为一个新结点分配一个磁盘页
        return BTreeNode(degree=degree)

    # 叶子节点查找关键字
    def predict(self,an_item):
        i = 0
        while i < self.number and an_item > self.items[i]:
            i += 1
        if i < self.number and an_item == self.items[i]:
            return self.items[i].v, i
        else:
            return 0,-1

    def range_search(self, k1, k2, res):
        for i in range(self.number):
            if k1 <= self.items[i].k <= k2:
                res.add(self.items[i])

# BTree Class
class BTree:
    def __init__(self, degree=2, distribution = "linear"):  # 初始化
        # B树的最小度数
        self.D = degree
        # 节点包含关键字的最大个数
        self.KEY_MAX = 2 * self.D - 1
        # 非根结点包含关键字的最小个数
        self.KEY_MIN = self.D - 1
        # 子结点的最大个数
        self.CHILD_MAX = self.KEY_MAX + 1
        # 子结点的最小个数
        self.CHILD_MIN = self.KEY_MIN + 1
        # 根结点
        self.root: BTreeNode = None
        # 输入数据的分布
        self.distribution = distribution

    def __new_node(self):  # 创建新的btree结点
        return BTreeNode.allocate_node(self.D)

    # def display(self):  # 打印树的内容
    #     self.__display(self.root)
    #
    # def __display(self, pNode: BTreeNode):  # 打印树的内容
    #     if pNode.isLeaf == True:
    #         print("leaf key_count =", pNode.number, "key list :", end="")
    #         for i in range(pNode.number):
    #             print(pNode.items[i].k, ", ", end="")
    #         print("")
    #     else:
    #         for i in range(pNode.number + 1):
    #             self.display(pNode.children[i])
    #         print("inner node key_count", pNode.number, "key list :", end="")
    #         for i in range(pNode.number):
    #             print(pNode.items[i].k, ", ", end="")
    #         print("")

    def build(self, keys, values):  # 批量插入key和value，创建b-tree
        # key和value数据个数不相等
        if len(keys) != len(values):
            return
        for ind in range(len(keys)):
            # 插入每组key和value
            self.insert(Item(keys[ind], values[ind]))

    def bulk_load(self, keys, values):
        # key和value数据个数不相等
        if len(keys) != len(values):
            return
        # 用于存放每一层的结点以及结点中存放的item内容
        whole_node = []
        # 用于存放所有的叶子节点和中间结点
        leaves, seps = [[]], []
        # 将所有叶子结点均创建出来
        for ind in range(len(keys)):
            # 按照每个结点可存放的最大item数量划分出叶子节点leaves和中间结点存放的内容seps
            if len(leaves[-1]) < self.KEY_MAX:
                leaves[-1].append(Item(keys[ind], values[ind]))
            else:
                seps.append(Item(keys[ind], values[ind]))
                leaves.append([])
        # 最后的叶子节点未满足最小个数
        if len(leaves[-1]) < self.KEY_MIN and seps:
            last_two = leaves[-2] + [seps.pop()] + leaves[-1]
            leaves[-2] = last_two[:self.KEY_MIN]
            leaves[-1] = last_two[self.KEY_MIN + 1:]
            seps.append(last_two[self.KEY_MIN])
        # 将叶子节点插入whole_node中
        whole_node.append(leaves)

        # 用中间内容seps生成中间结点
        while len(seps) > self.KEY_MAX:
            items, nodes, seps = seps, [[]], []
            for ind in range(len(items)):
                # 按照每个结点可存放的最大item数量划分出这一层节点和上一层结点存放的内容seps
                if len(nodes[-1]) < self.KEY_MAX:
                    nodes[-1].append(items[ind])
                else:
                    seps.append(items[ind])
                    nodes.append([])
            # 最后的节点未满足最小个数
            if len(nodes[-1]) < self.KEY_MIN and seps:
                last_two = nodes[-2] + [seps.pop()] + nodes[-1]
                nodes[-2] = last_two[:self.KEY_MIN]
                nodes[-1] = last_two[self.KEY_MIN + 1:]
                seps.append(last_two[self.KEY_MIN])

            whole_node.append(nodes)
        if len(seps) > 0:
            whole_node.append([seps])

        # 根据列表构建树
        if len(whole_node) == 1:
            # 尝试构建模型
            model = ModelNode(len(whole_node[0][0]),self.D)
            model.bulk_load(whole_node[0][0])
            if model.mean_error > thresholdPool[self.distribution]:
                # 使用B-Tree来代替
                tempt = BTreeNode(self.D, len(whole_node[0][0]), True)
                for i in range(len(whole_node[0][0])):
                    tempt.items[i] = whole_node[0][0][i]
                self.root = tempt
            else:
                self.root = model
        else:
            # 构建叶子结点
            for i in range(len(whole_node[0])):
                # 尝试构建模型
                model = ModelNode(len(whole_node[0][i]), self.D)
                model.bulk_load(whole_node[0][i])
                if model.mean_error > thresholdPool[self.distribution]:
                    tempt = BTreeNode(self.D, len(whole_node[0][i]), True)
                    for j in range(len(whole_node[0][i])):
                        tempt.items[j] = whole_node[0][i][j]
                    whole_node[0][i] = tempt
                else:
                    whole_node[0][i] = model
            # 构建中间结点以及建立其子节点
            for i in range(1, len(whole_node)):
                # 构建中间结点
                for j in range(len(whole_node[i])):
                    tempt = BTreeNode(self.D, len(whole_node[i][j]), False)
                    for k in range(len(whole_node[i][j])):
                        tempt.items[k] = whole_node[i][j][k]
                    # 构建联系
                    if j != len(whole_node[i]) - 1:
                        for k in range(len(whole_node[i][j]) + 1):
                            tempt.children[k] = whole_node[i - 1][0]
                            whole_node[i - 1].pop(0)
                    else:
                        for k in range(len(whole_node[i - 1])):
                            tempt.children[k] = whole_node[i - 1][0]
                            whole_node[i - 1].pop(0)
                    whole_node[i][j] = tempt
            self.root = whole_node[len(whole_node) - 1][0]

    def range_search(self, k1, k2):
        res = set()
        self._range_search(self.root, k1, k2, res)
        return res

    def _range_search(self, pNode, k1, k2, res):
        if pNode.isLeaf == True:
            pNode.range_search(k1, k2, res)
        else:
            if pNode is None:
                return
            for i in range(pNode.number):
                if pNode.items[i].k > k1:
                    self._range_search(pNode.children[i], k1, k2, res)
                if pNode.items[i].k >= k1 and pNode.items[i].k <= k2:
                    res.add(pNode.items[i])
                if pNode.items[i].k < k2:
                    self._range_search(pNode.children[i + 1], k1, k2, res)

    def predict(self, key):  # 预测关键字是否存在，返回key对应的value值 以及 查找key产生的error
        return self.__predict(self.root, Item(key, 0))

    def __predict(self, pNode: BTreeNode, an_item):  # 预测关键字是否存在
        if pNode == None:
            return 0,-1
        # 判断是否为叶子节点
        if pNode.isLeaf == True:
            return pNode.predict(an_item)
        else:
            i = 0
            # 找到使an_item < pNode.items[i]成立的最小下标
            while i < pNode.number and an_item > pNode.items[i]:
                i += 1
            if i < pNode.number and an_item == pNode.items[i]:
                return pNode.items[i].v, i
            else:  # 否则进入子树继续寻找
                return self.__predict(pNode.children[i], an_item)

    def contain(self, an_item):  # 检查该关键字是否存在于b-tree中
        self.__search(self.root, an_item)

    def __search(self, pNode, an_item):  # 查找关键字
        # 检测结点是否为空，若节点为空，则不存在关键字
        if pNode is None:
            return False
        else:
            if pNode.isLeaf == True:
                value, index = pNode.predict(an_item)
                if index == -1:
                    return False
                else:
                    return True
            else:
                i = 0
                # 找到使an_item < pNode.items[i]成立的最小下标
                while i < pNode.number and an_item > pNode.items[i]:
                    i += 1
                if i < pNode.number and an_item == pNode.items[i]:
                    return i
                else:  # 否则进入子树继续寻找
                    return self.__search(pNode.children[i], an_item)


    def insert(self, an_item):  # 插入新的关键字
        # 先查找树中是否存在这个关键字
        if self.contain(an_item) == True:  # 树中已经存在关键字，则插入失败
            return False
        else:
            # 空树情况特殊处理
            if self.root is None:
                # 创建根节点
                node = self.__new_node()
                self.root = node
            # 判断根结点是否已满，若根结点无法添加元素，需通过分裂来产生插入新元素的位置
            if self.root.number == self.KEY_MAX:
                # 创建新的根结点，用于存储分裂后的部分元素
                pNode = self.__new_node()
                pNode.isLeaf = False
                pNode.children[0] = self.root
                # 根结点进行分裂
                self.__split_child(pNode, 0, self.root)
                # 更新结点指针
                self.root = pNode
            # b-tree未满，可继续插入新元素
            self.__insert_non_full(self.root, an_item)
            return True

    def remove(self, an_item):  # 从b-tree中删除关键字
        # 如果关键字不存在，则无法进行删除
        if self.contain(an_item) == False:
            return False
        # 递归删除关键字
        self.__recursive_remove(self.root, an_item)
        return True

    def update(self, key, updated_value):  # 更新关键字所对应的value值
        return self.__update(self.root, key, updated_value)

    def __update(self, pNode: BTreeNode, key, updated_value):  # 更新关键字的value值
        i = 0
        # 找到使key < pNode.items[i].k成立的最小下标
        while i < pNode.number and key > pNode.items[i].k:
            i += 1
        if i < pNode.number and key == pNode.items[i].k:
            # 更新关键字的value
            pNode.items[i].v = updated_value
            return updated_value, i
        else:
            # 检查该结点是否为叶子节点，若为叶子节点，则不存在关键字
            if pNode.isLeaf == True:  # b-tree不存在关键字，index返回-1
                return 0, -1
            else:  # 否则进入子树继续寻找
                return self.__predict(pNode.children[i], key, updated_value)


    def clear(self):  # 清空b-tree
        # 递归删除树的所有子节点
        self.__recursive_clear(self.root)
        # 清空根节点
        self.root = None

    def __recursive_clear(self, pNode: BTreeNode):  # 删除b-tree的所有子节点
        if pNode is not None:
            if not pNode.isLeaf:
                # 递归删除所有子节点
                for i in range(pNode.number):
                    self.__recursive_clear(pNode.children[i])
            # 删除节点pNode
            self.__delete_node(pNode)

    def __delete_node(self, pNode: BTreeNode):  # 删除节点
        if pNode is not None:
            del pNode


    # 父节点pParent中第nChildIndex个子节点（即pChild）的关键字个数满了，其需进行分裂
    def __split_child(self, pParent: BTreeNode, nChildIndex, pChild: BTreeNode):
        # 将pChild分裂成pRightNode和pChild两个结点
        pRightNode = self.__new_node()  # 分裂后的右结点
        pRightNode.isLeaf = pChild.isLeaf
        pRightNode.number = self.KEY_MIN
        # 拷贝结点中的关键字，注意在拷贝完成后，需将其进行清空
        # pChild节点中包含2d-1个关键字，分裂后的结点各包含前d-1和后d-1个关键字，中间的关键字移至父节点中
        for i in range(self.KEY_MIN):
            # 获取原结点pChild中后d-1个关键字，即[d,2d-2]
            pRightNode.items[i] = pChild.items[i + self.CHILD_MIN]
            # 将后d-1个关键字设置为空
            pChild.items[i + self.CHILD_MIN] = None
        # 如果不是叶子结点，需拷贝子结点，注意在拷贝完成后，需将其进行清空
        # pChild包含2d个子节点，分裂后各包含d个子节点
        if not pChild.isLeaf:
            # 获取原结点pChild中后d个子节点，即[d,2d-1]
            for i in range(self.CHILD_MIN):
                pRightNode.children[i] = pChild.children[i + self.CHILD_MIN]
                # 将后d个子节点设置为空
                pChild.children[i + self.CHILD_MIN] = None
        # 更新左子树的关键字个数
        pChild.number = self.KEY_MIN
        # 将中间的关键字插入父节点中
        # 将父结点中的pChildIndex后的所有关键字和子树指针向后移动，共 pParent.number - nChildIndex 个元素
        for i in range(pParent.number - 1, nChildIndex - 1, -1):
            pParent.children[i + 2] = pParent.children[i + 1]
            pParent.items[i + 1] = pParent.items[i]
        # 更新父结点的关键字个数
        pParent.number += 1
        # 存储右子树指针
        pParent.children[nChildIndex + 1] = pRightNode
        # 把结点的中间值提到父结点，即第d-1个元素
        pParent.items[nChildIndex] = pChild.items[self.KEY_MIN]
        # 将左子树的最后元素清空
        pChild.items[self.KEY_MIN] = None

    def __insert_non_full(self, pNode, an_item):  # 在非满节点中插入关键字
        # 获取结点内关键字个数
        i = pNode.number
        # 若pNode为叶子节点，直接在节点中找到位置插入即可
        if pNode.isLeaf == True:
            # 从后往前查找插入关键字的节点位置
            while i > 0 and an_item < pNode.items[i - 1]:
                # 关键字向后移位
                pNode.items[i] = pNode.items[i - 1]
                i -= 1
            # 插入关键字的值
            pNode.items[i] = an_item
            # 更新结点关键字的个数
            pNode.number += 1
        # pNode是内部结点时，需找到插入关键字的下一层结点，并判断对应结点元素是否满了
        else:
            # 从后往前查找插入关键字的子树
            while i > 0 and an_item < pNode.items[i - 1]:
                i -= 1
            # 目标子树结点指针
            pChild = pNode.children[i]
            # 子树结点已经满了，需对其进行分裂
            if pChild.number == self.KEY_MAX:
                # 分裂子树结点
                self.__split_child(pNode, i, pChild)
                # 确定目标子树
                if an_item > pNode.items[i]:
                    pChild = pNode.children[i + 1]
            # 插入关键字到目标子树结点
            self.__insert_non_full(pChild, an_item)

    # 将节点pParent第index个元素以及对应的两个子节点进行合并
    def __merge_child(self, pParent: BTreeNode, index):
        # 获取合并的两个子节点
        pChild1 = pParent.children[index]
        pChild2 = pParent.children[index + 1]
        # 修改左节点的元素个数为 KEY_MAX
        pChild1.number = self.KEY_MAX
        # 将父结点第index个元素下移
        pChild1.items[self.KEY_MIN] = pParent.items[index]
        # 将右节点的值合并到左节点中
        for i in range(self.KEY_MIN):
            pChild1.items[i + self.KEY_MIN + 1] = pChild2.items[i]
        # 合并两个子节点的子节点
        if not pChild1.isLeaf:
            for i in range(self.CHILD_MIN):
                pChild1.children[i + self.CHILD_MIN] = pChild2.children[i]
        # 父结点删除第index个元素，并将index后的元素和子节点前移
        pParent.number -= 1
        for i in range(index, pParent.number):
            pParent.items[i] = pParent.items[i + 1]
            pParent.children[i + 1] = pParent.children[i + 2]
        # 将父节点中最后的元素和子节点清空
        pParent.items[pParent.number] = None
        pParent.children[pParent.number + 1] = None
        # 判断父节点是否为空
        if pParent.number == 0:
            self.__delete_node(pParent)
            self.root = pChild1
        # 删除pChild2
        self.__delete_node(pChild2)

    def __recursive_remove(self, pNode: BTreeNode, an_item):  # 递归删除关键字
        i = 0
        while i < pNode.number and an_item > pNode.items[i]:
            i += 1
        # 关键字key在结点pNode中
        if i < pNode.number and an_item == pNode.items[i]:
            # pNode是个叶结点
            if pNode.isLeaf == True:
                # 直接从pNode中删除关键字
                pNode.number -= 1
                for j in range(i, pNode.number):
                    pNode.items[j] = pNode.items[j + 1]
                pNode.items[pNode.number] = None
                return
            # pNode是个内结点
            else:
                pChildPrev = pNode.children[i]  # 节点pNode的左子节点
                pChildNext = pNode.children[i + 1]  # 节点pNode的右子节点
                # 左子节点中元素个数大于KEY_MIN，可获取替换的关键字
                if pChildPrev.number > self.KEY_MIN:
                    # 获取比key小的最大关键字（即前驱关键字）
                    prevKey = self.predecessor(pChildPrev)
                    # 删除替换的关键字
                    self.__recursive_remove(pChildPrev, prevKey)
                    # 替换关键字
                    pNode.items[i] = prevKey
                    return
                # 右子节点中元素个数大于KEY_MIN，可获取替换的关键字
                elif pChildNext.number > self.KEY_MIN:
                    # 获取比key大的最小关键字（即后驱关键字）
                    nextKey = self.successor(pChildNext)
                    # 删除替换的关键字
                    self.__recursive_remove(pChildNext, nextKey)
                    # 替换关键字
                    pNode.items[i] = nextKey
                    return
                # 左子节点和右子节点均只包含KEY_MIN个元素，需和父节点中的元素进行合并后，再删除关键字
                else:
                    # 节点合并，所有元素均并入左子节点中
                    self.__merge_child(pNode, i)
                    # 从左子节点中删除关键字
                    self.__recursive_remove(pChildPrev, an_item)
                    return
        # 关键字key不在结点pNode中
        else:
            # 包含关键字的子树根结点
            pChildNode = pNode.children[i]
            # 子树根节点仅包含KEY_MIN个元素
            if pChildNode.number == self.KEY_MIN:
                pLeft = None  # 左兄弟结点
                pRight = None  # 右兄弟结点
                if i > 0:
                    pLeft = pNode.children[i - 1]
                if i < pNode.number:
                    pRight = pNode.children[i + 1]
                # 左兄弟节点中存在移动的元素
                if pLeft is not None and pLeft.number > self.KEY_MIN:
                    '''
                    pNode中的元素下移至pChildNode
                    '''
                    # pChildNode中的元素向后移动一位
                    for j in range(pChildNode.number, 0, -1):
                        pChildNode.items[j] = pChildNode.items[j - 1]
                    # 父结点中i-1的关键字下移至pChildNode中
                    pChildNode.items[0] = pNode.items[i - 1]
                    '''
                    pLeft中的子节点移动至pChildNode的子节点中
                    '''
                    # pChildNode中的子节点向后移动一位
                    for j in range(pChildNode.number + 1, 0, -1):
                        pChildNode.children[j] = pChildNode.children[j - 1]
                    # 左兄弟节点中最后一个子节点移动至pChildNode中的第一个节点
                    pChildNode.children[0] = pLeft.children[pLeft.number]
                    # pChildNode关键字个数增加1
                    pChildNode.number += 1
                    '''
                    pLeft中的元素上移至pNode中
                    '''
                    # pLeft元素上移
                    pNode.items[i - 1] = pLeft.items[pLeft.number - 1]
                    # 删除左兄弟节点的最后一个子节点
                    pLeft.children[pLeft.number] = None
                    # 删除左兄弟节点的最后一个元素
                    pLeft.items[pLeft.number - 1] = None
                    # 左兄弟节点关键字个数减少1
                    pLeft.number -= 1
                    '''
                    删除关键字
                    '''
                    self.__recursive_remove(pChildNode, an_item)
                    return
                # 右左兄弟节点中存在移动的元素
                elif pRight is not None and pRight.number > self.KEY_MIN:
                    '''
                    pNode中的元素下移至pChildNode
                    '''
                    # 父结点中i的关键字下移至pChildNode中
                    pChildNode.items[pChildNode.number] = pNode.items[i]
                    '''
                    pRight中的子节点移动至pChildNode的子节点中
                    '''
                    pChildNode.children[pChildNode.number + 1] = pRight.children[0]
                    pChildNode.number += 1
                    '''
                    pRight中的元素上移至pNode中
                    '''
                    # pRight元素上移
                    pNode.items[i] = pRight.items[0]
                    # pRight元素均向前移一位
                    for i in range(pRight.number - 1):
                        pRight.items[i] = pRight.items[i + 1]
                    # pRight子节点均向前移动一位
                    for i in range(pRight.number):
                        pRight.children[i] = pRight.children[i + 1]
                    # pRight元素个数减少1
                    pRight.number -= 1
                    '''
                    删除关键字
                    '''
                    self.__recursive_remove(pChildNode, an_item)
                    return
                # 左右兄弟结点都只包含KEY_MIN个元素
                else:
                    # 当i指向最后一个关键字的时候，合并的时候要往前移动一步
                    if (i >= pNode.number):
                        i -= 1
                    # 节点合并，所有元素均并入左子节点中
                    self.__merge_child(pNode, i)
                    self.__recursive_remove(pChildNode, an_item)
                    return
        self.__recursive_remove(pChildNode, an_item)

    def predecessor(self, pNode: BTreeNode):  # 获取左子节点中最大的元素
        # 左子节点的最右元素
        while not pNode.isLeaf:
            pNode = pNode.children[pNode.number]
        return pNode.items[pNode.number - 1]

    def successor(self, pNode: BTreeNode):  # 获取右子节点中最小的元素
        # 右子节点的最左元素
        while not pNode.isLeaf:
            pNode = pNode.children[0]
        return pNode.items[0]

# Value in Node
class Item():
    def __init__(self, k, v):
        self.k = k
        self.v = v

    def __eq__(self, other):
        if isinstance(other, Item):
            return (self.k == other.k)
        else:
            return False

    def __hash__(self):
        return hash(self.k)

    def __gt__(self, other):
        if self.k > other.k:
            return True
        else:
            return False

    def __ge__(self, other):
        if self.k >= other.k:
            return True
        else:
            return False

    def __le__(self, other):
        if self.k <= other.k:
            return True
        else:
            return False

    def __lt__(self, other):
        if self.k < other.k:
            return True
        else:
            return False

def data_processing_new(distribution_bulk_load, distribution_insert, flag, num):

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

def change_distribution_to_label(distribution):
    global change_distribution
    if distribution == "linear":
        change_distribution = Distribution.LINEAR
    elif distribution == "random":
        change_distribution = Distribution.RANDOM
    elif distribution == "exponential":
        change_distribution = Distribution.EXPONENTIAL
    elif distribution == "normal":
        change_distribution = Distribution.NORMAL
    elif distribution == "lognormal":
        change_distribution = Distribution.LOGNORMAL
    elif distribution == "wiki":
        change_distribution = Distribution.WIKI
    elif distribution == "osm":
        change_distribution = Distribution.OSM
    elif distribution == "books":
        change_distribution = Distribution.BOOKS
    elif distribution == "fb":
        change_distribution = Distribution.FB
    return change_distribution

def b_tree_test1(distribution_bulk_load, distribution_insert, is_larger, num):
    distribution_bulk_load_tempt = distribution_bulk_load.lower()
    distribution_insert_tempt = distribution_insert.lower()
    distribution_bulk_load_tempt = change_distribution_to_label(distribution_bulk_load_tempt)
    distribution_insert_tempt = change_distribution_to_label(distribution_insert_tempt)

    if is_larger:
        # 插入的数据比批量加载的数据大，则批量加载数据取前num条，插入数据取后num条
        bulk_load_flag = 0
        insert_flag = 1
    else:
        # 插入的数据比批量加载的数据小，则批量加载数据取后num条，插入数据取前num条
        bulk_load_flag = 1
        insert_flag = 0

    # 批量加载和插入操作数据
    load_keys, load_values, insert_keys, insert_values = data_processing_new(distribution_bulk_load_tempt,
                                                                             distribution_insert_tempt, bulk_load_flag, num)
    # print(load_keys)

    bt = BTree(DEGREE)
    # 测试索引构建时间
    print("*************start Build B-Tree************")
    print("Start Build")
    build_start_time = time.time()
    bt.bulk_load(load_keys, load_values)
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

    with open("model/" + distribution_bulk_load + "/full_train/BTree/" + str(num) + ".json",
              "w") as jsonFile:
        json.dump(result, jsonFile)

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

    # 测试插入数据后查询时间是否发生变化
    print("*************start Bulk Load Key Search************")
    print("Calculate Time And Error")
    new_bulk_load_search_start_time = time.time()
    for the_key in load_keys:
        value, search_err = bt.predict(the_key)
        if search_err < 0:
            print("We Can Not Find The Key!")
    new_bulk_load_search_end_time = time.time()
    new_bulk_load_search_time = (new_bulk_load_search_end_time - new_bulk_load_search_start_time) / len(load_keys)
    print("Bulk Load Key New Search time %f " % new_bulk_load_search_time)
    print("*************end Bulk Load Key Search************\n")

    # # 判断是否成功插入数据并测试查询插入数据的时间
    # print("*************start Insert Key Search************")
    # print("Calculate Time And Error")
    # new_insert_search_start_time = time.time()
    # for the_key in insert_keys:
    #     value, search_err = bt.predict(the_key)
    #     if search_err < 0:
    #         print("We Can Not Find The Key!")
    # new_insert_search_end_time = time.time()
    # new_insert_search_time = (new_insert_search_end_time - new_insert_search_start_time) / len(insert_keys)
    # print("Insert Key New Search time %f " % new_insert_search_time)
    # print("*************end Insert Key Search************\n")

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
    #
    # # 测试删除数据后查询时间是否发生变化
    # print("*************start Bulk Load Key Search************")
    # print("Calculate Time And Error")
    # delete_bulk_load_search_start_time = time.time()
    # for the_key in load_keys:
    #     value, search_err = bt.predict(the_key)
    #     if search_err < 0:
    #         print("We Can Not Find The Key!")
    # delete_bulk_load_search_end_time = time.time()
    # delete_bulk_load_search_time = (delete_bulk_load_search_end_time - delete_bulk_load_search_start_time) / len(
    #     load_keys)
    # print("Bulk Load Key Delete Search time %f " % delete_bulk_load_search_time)
    # print("*************end Bulk Load Key Search************\n")

    print(os.path.getsize("model/" + distribution_bulk_load + "/full_train/BTree/" + str(num) + ".json"))

    # write performance into files
    performance_BTree = {"type": "BTree", "build time": build_time,
                         "search time": search_time, "search average error": search_mean_error,
                         "range search time": range_search_time,
                         "insert time": insert_time,
                         "bulk load new search time": new_bulk_load_search_time,
                         # "insert new search time": new_insert_search_time,
                         "delete time": delete_time,
                         "store size": os.path.getsize(
                             "model/" + distribution_bulk_load + "/full_train/BTree/" + str(num) + ".json")}
    with open("performance/" + distribution_bulk_load + "/full_train/BTree/" + str(num) + ".json",
              "w") as jsonFile:
        json.dump(performance_BTree, jsonFile)

    del bt
    gc.collect()


if __name__ == '__main__':
    b_tree_test1("Linear", "Linear", True, 1000)

