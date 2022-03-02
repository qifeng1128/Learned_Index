# -*- coding: utf-8 -*-

# BTree Index with Python

import pandas as pd
import time

count = 0

# Node in BTree
class BTreeNode:
    def __init__(self, degree=2, number=0, is_leaf=True):  # 初始化
        # 是否为叶子节点
        self.isLeaf = is_leaf
        # 结点包含关键字的数量
        self.number = number
        # 关键字的值数组(个数为2d-1)
        self.keys = [None] * (degree * 2 - 1)
        # 子结点数组(个数为2d)
        self.children = [None] * degree * 2

    def __str__(self):  # 通过字符串的形式直观展示b-tree节点中的内容
        returnStr = 'keys:['
        for i in range(self.number):
            returnStr += str(self.keys[i]) + ' '
        returnStr += '];childrens:['
        for child in self.children:
            returnStr += str(child) + ';'
        returnStr += ']\r\n'
        return returnStr

    def diskread(self):  # 磁盘读取
        # f = open("D:/Downloads/PycharmProjects/chapter18/tempt_read.txt", 'r')
        # tempt = f.readline()
        # f.close()
        global count
        count = count + 1

    def diskwrite(self):  # 磁盘写入
        # f = open("D:/Downloads/PycharmProjects/chapter18/tempt_write.txt", 'w')
        # f.write("111")
        # f.close()
        global count
        count = count + 1

    @classmethod
    def allocate_node(self, degree):  # 为一个新结点分配一个磁盘页
        '''
        假定由ALLOCATE-NODE所创建的结点无需做DISK-READ，因为磁盘上还没有关于该结点的有用信息
        '''
        return BTreeNode(degree = degree)


# BTree Class
class BTree:
    def __init__(self, degree=2):  # 初始化
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

    def __new_node(self):  # 创建新的btree结点
        return BTreeNode.allocate_node(self.D)

    def insert(self, key):  # 插入新的关键字
        # 先查找树中是否存在这个关键字
        if self.contain(key) == True:  # 树中已经存在关键字，则插入失败
            return False
        else:
            # 空树情况特殊处理
            if self.root is None:
                # 创建根节点
                node = self.__new_node()
                node.diskwrite()
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
            self.__insert_non_full(self.root, key)
            return True

    def remove(self, key):   # 从b-tree中删除关键字
        # 如果关键字不存在，则无法进行删除
        if self.contain(key) == False:
            return False
        # 根节点只存在一个关键字的特殊情况
        if self.root.number == 1:
            # 整颗树只存在一个元素，直接将树清空
            if self.root.isLeaf == True:
                self.clear()
            else:
                pChild1 = self.root.children[0]
                pChild2 = self.root.children[1]
                # 若根节点单个元素的两个子节点均包含 KEY_MIN 个元素，则将节点进行合并
                if pChild1.number == self.KEY_MIN and pChild2.number == self.KEY_MIN:
                    # 将元素均合并至pChild1节点
                    self.__merge_child(self.root, 0)
                    # 删除根节点，并将根节点设置为pChild1
                    self.__delete_node(self.root)
                    self.root = pChild1
        #
        self.__recursive_remove(self.root, key)
        return True

    def contain(self, key):  # 检查该关键字是否存在于b-tree中
        self.__search(self.root, key)

    def clear(self):   # 清空b-tree
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

    def __delete_node(self, pNode: BTreeNode):   # 删除节点
        if pNode is not None:
            del pNode

    def __search(self, pNode: BTreeNode, key):  # 查找关键字
        # 检测结点是否为空，若节点为空，则不存在关键字
        if pNode is None:
            return False
        else:
            i = 0
            # 找到使key < pNode.keys[i]成立的最小下标
            while i < pNode.number and key > pNode.keys[i]:
                i += 1
            if i < pNode.number and key == pNode.keys[i]:
                return True
            else:
                # 检查该结点是否为叶子节点，若为叶子节点，则不存在关键字
                if pNode.isLeaf == True:
                    return False
                else:  # 否则进入子树继续寻找
                    return self.__search(pNode.children[i], key)

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
            pRightNode.keys[i] = pChild.keys[i + self.CHILD_MIN]
            # 将后d-1个关键字设置为空
            pChild.keys[i + self.CHILD_MIN] = None
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
            pParent.keys[i + 1] = pParent.keys[i]
        # 更新父结点的关键字个数
        pParent.number += 1
        # 存储右子树指针
        pParent.children[nChildIndex + 1] = pRightNode
        # 把结点的中间值提到父结点，即第d-1个元素
        pParent.keys[nChildIndex] = pChild.keys[self.KEY_MIN]
        # 将左子树的最后元素清空
        pChild.keys[self.KEY_MIN] = None
        pChild.diskwrite()
        pRightNode.diskwrite()
        pParent.diskwrite()

    def __insert_non_full(self, pNode: BTreeNode, key):  # 在非满节点中插入关键字
        # 获取结点内关键字个数
        i = pNode.number
        # 若pNode为叶子节点，直接在节点中找到位置插入即可
        if pNode.isLeaf == True:
            # 从后往前查找插入关键字的节点位置
            while i > 0 and key < pNode.keys[i - 1]:
                # 关键字向后移位
                pNode.keys[i] = pNode.keys[i - 1]
                i -= 1
            # 插入关键字的值
            pNode.keys[i] = key
            # 更新结点关键字的个数
            pNode.number += 1
            pNode.diskwrite()
        # pNode是内部结点时，需找到插入关键字的下一层结点，并判断对应结点元素是否满了
        else:
            # 从后往前查找插入关键字的子树
            while i > 0 and key < pNode.keys[i - 1]:
                i -= 1
            # 目标子树结点指针
            pChild = pNode.children[i]
            pNode.children[i].diskread()
            # 子树结点已经满了，需对其进行分裂
            if pChild.number == self.KEY_MAX:
                # 分裂子树结点
                self.__split_child(pNode, i, pChild)
                # 确定目标子树
                if key > pNode.keys[i]:
                    pChild = pNode.children[i + 1]
            # 插入关键字到目标子树结点
            self.__insert_non_full(pChild, key)

    # 将节点pParent第index个元素以及对应的两个子节点进行合并
    def __merge_child(self, pParent: BTreeNode, index):
        # 获取合并的两个子节点
        pChild1 = pParent.children[index]
        pChild2 = pParent.children[index + 1]
        # 修改左节点的元素个数为 KEY_MAX
        pChild1.number = self.KEY_MAX
        # 将父结点第index个元素下移
        pChild1.keys[self.KEY_MIN] = pParent.keys[index]
        # 将右节点的值合并到左节点中
        for i in range(self.KEY_MIN):
            pChild1.keys[i + self.KEY_MIN + 1] = pChild2.keys[i]
        # 合并两个子节点的子节点
        if not pChild1.isLeaf:
            for i in range(self.CHILD_MIN):
                pChild1.children[i + self.CHILD_MIN] = pChild2.children[i]
        # 父结点删除第index个元素，并将index后的元素和子节点前移
        pParent.number -= 1
        for i in range(index, pParent.number):
            pParent.keys[i] = pParent.keys[i + 1]
            pParent.children[i + 1] = pParent.children[i + 2]
        # 将父节点中最后的元素和子节点清空
        pParent.keys[pParent.number] = None
        pParent.children[pParent.number + 1] = None
        # 删除pChild2
        self.__delete_node(pChild2)

    def __recursive_remove(self, pNode: BTreeNode, key):   # 递归删除关键字
        i = 0
        while i < pNode.number and key > pNode.keys[i]:
            i += 1
        # 关键字key在结点pNode中
        if i < pNode.number and key == pNode.keys[i]:
            # pNode是个叶结点
            if pNode.isLeaf == True:
                # 直接从pNode中删除关键字
                pNode.number -= 1
                for j in range(i, pNode.number):
                    pNode.keys[j] = pNode.keys[j + 1]
                pNode.keys[pNode.number] = None
                return
            # pNode是个内结点
            else:
                pChildPrev = pNode.children[i]          # 节点pNode的左子节点
                pChildNext = pNode.children[i + 1]      # 节点pNode的右子节点
                # 左子节点中元素个数大于KEY_MIN，可获取替换的关键字
                if pChildPrev.number > self.KEY_MIN:
                    # 获取比key小的最大关键字（即前驱关键字）
                    prevKey = self.predecessor(pChildPrev)
                    # 删除替换的关键字
                    self.__recursive_remove(pChildPrev, prevKey)
                    # 替换关键字
                    pNode.keys[i] = prevKey
                    return
                # 右子节点中元素个数大于KEY_MIN，可获取替换的关键字
                elif pChildNext.number > self.KEY_MIN:
                    # 获取比key大的最小关键字（即后驱关键字）
                    nextKey = self.successor(pChildNext)
                    # 删除替换的关键字
                    self.__recursive_remove(pChildNext, nextKey)
                    # 替换关键字
                    pNode.keys[i] = nextKey
                    return
                # 左子节点和右子节点均只包含KEY_MIN个元素，需和父节点中的元素进行合并后，再删除关键字
                else:
                    # 节点合并，所有元素均并入左子节点中
                    self.__merge_child(pNode, i)
                    # 从左子节点中删除关键字
                    self.__recursive_remove(pChildPrev, key)
        # 关键字key不在结点pNode中
        else:
            # 包含关键字的子树根结点
            pChildNode = pNode.children[i]
            # 子树根节点仅包含KEY_MIN个元素
            if pChildNode.number == self.KEY_MIN:
                pLeft = None    # 左兄弟结点
                pRight = None   # 右兄弟结点
                if i > 0:
                    pLeft = pNode.children[i - 1]
                if i < pNode.number:
                    pRight = pNode.children[i + 1]
                # 左兄弟节点中存在移动的元素
                if pLeft is not None and pLeft.number > self.KEY_MIN:
                    # pChildNode中的元素向后移动一位
                    for j in range(pChildNode.number - 1,0,-1):
                        pChildNode.keys[j] = pChildNode.keys[j - 1]
                    # 父结点中i-1的关键字下移至pChildNode中
                    pChildNode.keys[0] = pNode.keys[i - 1]
                    pChildNode.number += 1
                    # 获取左兄弟节点中最大的元素
                    prevKey = self.predecessor(pLeft)
                    # 删除替换的关键字
                    self.__recursive_remove(pLeft, prevKey)
                    # 替换关键字
                    pNode.keys[i - 1] = prevKey

                # 右左兄弟节点中存在移动的元素
                elif pRight is not None and pRight.number > self.KEY_MIN:
                    # 父结点中i的关键字下移至pChildNode中
                    pChildNode.keys[pChildNode.number] = pNode.keys[i]
                    pChildNode.number += 1
                    # 获取右兄弟节点中最小的元素
                    nextKey = self.successor(pRight)
                    # 删除替换的关键字
                    self.__recursive_remove(pRight, nextKey)
                    # 替换关键字
                    pNode.keys[i] = nextKey
                # 左右兄弟结点都只包含KEY_MIN个元素
                else:
                    # 将左右节点和父节点中的元素进行合并
                    self.__merge_child(pNode, i - 1)
                    pChildNode = pLeft
            self.__recursive_remove(pChildNode, key)

    def predecessor(self, pNode: BTreeNode):    # 获取左子节点中最大的元素
        # 左子节点的最右元素
        while not pNode.isLeaf:
            pNode = pNode.children[pNode.number]
        return pNode.keys[pNode.number - 1]

    def successor(self, pNode: BTreeNode):    # 获取右子节点中最小的元素
        # 右子节点的最左元素
        while not pNode.isLeaf:
            pNode = pNode.children[0]
        return pNode.keys[0]


# For Test
def b_tree_main():
    path = "D:\Downloads\PycharmProjects\Learned-Indexes-master\Learned-Indexes-master\data\exponential.csv"
    data = pd.read_csv(path, header=None)
    train_set_x = []
    train_set_y = []
    test_set_x = []

    for i in range(data.shape[0]):
        train_set_x.append(data.iloc[i, 0])

    test_set_x = train_set_x[:]

    print("*************start BTree************")
    tree = BTree(20)
    print("Start Build")
    start_time = time.time()

    for ind in test_set_x:
        tree.insert(ind)

    end_time = time.time()
    build_time = end_time - start_time
    print("Build BTree time ", build_time)
    err = 0
    print("Calculate error")
    start_time = time.time()

    for ind in test_set_x:
        tree.contain(ind)

    end_time = time.time()
    search_time_without_disk = (end_time - start_time) / len(test_set_x)
    search_time_with_disk = (end_time - start_time + 0.00014 * count) / len(test_set_x)
    print("Search time without disk", search_time_without_disk)
    print("Search time with disk", search_time_with_disk)
    print("*************end BTree************")


def b_tree_test1():
    tree = BTree(2)
    for ind in range(8):
        tree.insert(ind)
    tree.remove(2)
    tree.remove(9)


if __name__ == '__main__':
    b_tree_test1()
