import numpy as np

class Node:
    def __init__(self, data, depth=0, lnode=None, rnode=None):
        """对输入的特征空间进行不断切割，将特征空间分为n个部分"""
        self.data = data  # 此节点
        self.depth = depth # 树的深度
        self.lnode = lnode # 左节点
        self.rnode = rnode # 右节点


class KDTree:
    def __init__(self):
        self.n = 0
        self.tree = None
        self.nearest = None

    def walk_tree(self, x, depth=0):
         """KD-Tree创建过程"""
         if len(x) > 0:
            m, self.n = np.shape(x)
            # 按照哪个维度进行分割，比如0：x轴，1：y轴
            axis = depth % self.n
            # 中位数
            mid = m // 2
            # 按照第几个维度（列）进行排序
            x_cp = sorted(x, key=lambda s: s[axis])
            # mid结点为中位数的结点，树深度为depth
            node = Node(x_cp[mid], depth)
            if depth == 0:
                self.tree = node
            # 前mid行为左子结点，此时行数m改变，深度depth+1，axis会换个维度
            node.lnode = self.walk_tree(x_cp[:mid], depth+1)
            node.rnode = self.walk_tree(x_cp[mid+1:], depth+1)
            return node

         return None

    def search(self, x, depth=0):
        assert self.tree is not None, "KDTree has not yet been generated."
        # 从根部开始搜索
        axis = depth % self.n
        if len(x) > 0:
            # 判断向左边还是向右边搜索
            if x[:, axis] >= self.tree.data:
                # 向右
                pass
            else:
                # 向左
                pass

        return



class KNearestNeighbor:
    def __init__(self):
        self.kdtree = None

    def kdtree(self, x):
        pass

    def fit(self, X, y, k, random_state=0):
        pass

    def predict(self, X):
        return 
