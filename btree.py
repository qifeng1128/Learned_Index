#!usr/bin/python
# -*- coding: utf-8 -*-

# BTree Index with Python

import pandas as pd

# Node in BTree
class BTreeNode:
    def __init__(self, degree=2, number_of_keys=0, is_leaf=True, items=None, children=None,
                 index=None):
        # 是否为叶子节点
        self.isLeaf = is_leaf
        # 包含item的个数
        self.numberOfKeys = number_of_keys
        # 在所有结点中的index
        self.index = index
        # 2d-1个item
        if items is not None:
            self.items = items
        else:
            self.items = [None] * (degree * 2 - 1)
        # 2d个children
        if children is not None:
            self.children = children
        else:
            self.children = [None] * degree * 2

    def set_index(self, index):
        self.index = index

    def get_index(self):
        return self.index

    # fileIndex即所有节点中对应的位置
    # nodeIndex即某个节点中所有item对应的位置
    def search(self, b_tree, an_item):
        i = 0
        # 找到根节点中第一个大于等于目标值的index
        while i < self.numberOfKeys and an_item > self.items[i]:
            i += 1
        # 若等于目标值，则找到item，返回found--true
        if i < self.numberOfKeys and an_item == self.items[i]:
            return {'found': True, 'fileIndex': self.index, 'nodeIndex': i}
        # 若是叶子节点，则表示未找到item，返回found--false
        if self.isLeaf:
            return {'found': False, 'fileIndex': self.index, 'nodeIndex': i - 1}
        else:
            # 根节点未查询到，进入对应的子节点进行查询
            return b_tree.get_node(self.children[i]).search(b_tree, an_item)

# BTree Class
class BTree:
    # nodes[root_index]则表示为根节点
    def __init__(self, degree=2, nodes=None, root_index=1, free_index=2):
        if nodes is None:
            nodes = {}
        self.degree = degree
        # 还未构建节点
        if len(nodes) == 0:
            # 构建根节点
            self.rootNode = BTreeNode(degree)
            self.nodes = {}
            # 将其index设置为root_index
            self.rootNode.set_index(root_index)
            # 等价于nodes[1] = rootNode
            self.write_at(1, self.rootNode)
        else:
            self.nodes = nodes
            self.rootNode = self.nodes[root_index]
        self.rootIndex = root_index
        self.freeIndex = free_index

    def build(self, keys, values):
        # key和value数据个数不相等
        if len(keys) != len(values):
            return
        for ind in range(len(keys)):
            # 插入每组key和value
            self.insert(Item(keys[ind], values[ind]))

    def search(self, an_item):
        return self.rootNode.search(self, an_item)

    def predict(self, key):
        search_result = self.search(Item(key, 0))
        # 获取对应index的节点
        a_node = self.nodes[search_result['fileIndex']]
        if a_node.items[search_result['nodeIndex']] is None:
            return -1
        # 获取对应节点item中的value值
        return a_node.items[search_result['nodeIndex']].v, search_result['nodeIndex']

    # c_node节点数据满了，将此节点的数据移至其他节点
    def split_child(self, p_node, i, c_node):
        # 创建新节点用于暂时存储转移的数据
        new_node = self.get_free_node()
        new_node.isLeaf = c_node.isLeaf
        # 转移d-1个数据
        new_node.numberOfKeys = self.degree - 1
        # c_node节点中item包含[0,2d-2]，j选取[0,d-2]，new_node选取c_node中的[d,2d-2]
        # 原本数据包含了2d-1个数据，index为[0,2d-2]
        # 原本数据保留d-1个数据，index为[0,d-2]
        # 新节点数据选取后d-1个数据，index为[d,2d-2]
        for j in range(0, self.degree - 1):
            new_node.items[j] = c_node.items[j + self.degree]
        # 进行分裂的不是子节点
        if c_node.isLeaf is False:
            # 将c_node的子节点转移至new_node的子节点中
            # j取[0,d-1]，选取c_node的[d,2d-1]子节点
            # 因为c_node获取的item为[0,d-2]，则子节点保留[0,d-1]
            # 因为new_node获取的item为[d,2d-2]，则子节点保留[d,2d-1]
            for j in range(0, self.degree):
                new_node.children[j] = c_node.children[j + self.degree]
                c_node.children[j + self.degree] = None
        # c_node最后只保留前d-1个数据
        c_node.numberOfKeys = self.degree - 1
        # 判断p_node节点index为i的位置是否有item
        j = p_node.numberOfKeys + 1
        while j > i + 1:
            # 若存在item，需将子节点向后移
            p_node.children[j + 1] = p_node.children[j]
            j -= 1
        # 将包含后d-1个数据的节点new_node插入空出来的位置
        p_node.children[j] = new_node.get_index()
        j = p_node.numberOfKeys
        while j > i:
            # 存在item，将节点中的内容向后移动
            p_node.items[j + 1] = p_node.items[j]
            j -= 1
        # c_node包含前d-1个数据，new_node包含后d-1个数据，中间一个数据放入父节点中，即c_node中第d个数据，index为d-1
        p_node.items[i] = c_node.items[self.degree - 1]
        # 父节点数据增加1
        p_node.numberOfKeys += 1

    def insert(self, an_item):
        # 先查找树中是否存在这个item
        search_result = self.search(an_item)
        # 树中已经存在这个item，返回None
        if search_result['found']:
            return None
        r = self.rootNode
        # 根节点个数已经满了
        if r.numberOfKeys == 2 * self.degree - 1:
            # 构建新的根节点
            s = self.get_free_node()
            self.set_root_node(s)
            s.isLeaf = False
            s.numberOfKeys = 0
            # 将根节点的第一个子节点指向先前的节点
            s.children[0] = r.get_index()
            # 分裂根节点和字节的
            self.split_child(s, 0, r)
            # 传入根节点，插入数据
            self.insert_not_full(s, an_item)
        else:
            self.insert_not_full(r, an_item)

    def insert_not_full(self, inNode, anItem):
        # 节点中最后一个元素的index
        i = inNode.numberOfKeys - 1
        # 根节点为叶子节点，即只存在一个根节点
        if inNode.isLeaf:
            # 节点中的部分元素比插入元素小
            while i >= 0 and anItem < inNode.items[i]:
                # 将比插入元素大的元素逐渐向后移
                inNode.items[i + 1] = inNode.items[i]
                i -= 1
            # 插入元素
            inNode.items[i + 1] = anItem
            # 元素个数增加1
            inNode.numberOfKeys += 1
        else:
            # 找到最后一个比插入元素小的index
            while i >= 0 and anItem < inNode.items[i]:
                i -= 1
            i += 1
            # 判断其子节点中数据是否满了
            if self.get_node(inNode.children[i]).numberOfKeys == 2 * self.degree - 1:
                # 根节点和子节点分裂
                self.split_child(inNode, i, self.get_node(inNode.children[i]))
                # 判断转移至根节点的item是否大于目标值
                if anItem > inNode.items[i]:
                    i += 1
            self.insert_not_full(self.get_node(inNode.children[i]), anItem)

    def set_root_node(self, r):
        self.rootNode = r
        self.rootIndex = self.rootNode.get_index()

    def get_node(self, index):
        return self.nodes[index]

    def get_free_node(self):
        new_node = BTreeNode(self.degree)
        index = self.get_free_index()
        new_node.set_index(index)
        self.write_at(index, new_node)
        return new_node

    def get_free_index(self):
        self.freeIndex += 1
        return self.freeIndex - 1

    def write_at(self, index, a_node):
        self.nodes[index] = a_node

# Value in Node
class Item():
    def __init__(self, k, v):
        self.k = k
        self.v = v

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

    def __eq__(self, other):
        if self.k == other.k:
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

# For Test
def b_tree_main():
    path = "data/random.csv"
    data = pd.read_csv(path)
    b = BTree(2)
    for i in range(10):
        b.insert(Item(data.iloc[i, 0], data.iloc[i, 1]))

    b.delete(5)
    b.delete(7)
    b.delete(8)
    # pos = b.predict(30310)
    # print(pos)
    tempt = 2

if __name__ == '__main__':
    b_tree_main()
