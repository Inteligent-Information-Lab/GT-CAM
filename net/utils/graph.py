import numpy as np

class Graph():
    """ The Graph to model the skeletons extracted by the openpose

    Args:
        strategy (string): must be one of the follow candidates
        - uniform: Uniform Labeling
        - distance: Distance Partitioning
        - spatial: Spatial Configuration
        For more information, please refer to the section 'Partition Strategies'
            in our paper (https://arxiv.org/abs/1801.07455).

        layout (string): must be one of the follow candidates
        - openpose: Is consists of 18 joints. For more information, please
            refer to https://github.com/CMU-Perceptual-Computing-Lab/openpose#output
        - ntu-rgb+d: Is consists of 25 joints. For more information, please
            refer to https://github.com/shahroudy/NTURGB-D

        max_hop (int): the maximal distance between two connected nodes
        dilation (int): controls the spacing between the kernel points

    """

    def __init__(self,
                 layout='openpose',
                 strategy='uniform',
                 max_hop=1,
                 dilation=1):
        self.max_hop = max_hop
        self.dilation = dilation

        self.get_edge(layout)
        # 获取邻接矩阵
        self.hop_dis = get_hop_distance(
            self.num_node, self.edge, max_hop=max_hop)
        # 正则化
        self.get_adjacency(strategy)

    def __str__(self):
        return self.A

    # 节点分配
    def get_edge(self, layout):
        if layout == 'openpose':
            self.num_node = 18 # 人体骨架节点数
            self_link = [(i, i) for i in range(self.num_node)] #自己与自己相连的元组
            neighbor_link = [(4, 3), (3, 2), (7, 6), (6, 5), (13, 12), (12,
                                                                        11),
                             (10, 9), (9, 8), (11, 1), (8, 1), (5, 1), (2, 1),
                             (0, 1), (15, 0), (14, 0), (17, 15), (16, 14)] # 所以相连的节点
            self.edge = self_link + neighbor_link # 所有相连的元组
            self.center = 1 # 设置中心脊柱
        elif layout == 'ntu-rgb+d':
            self.num_node = 25
            self_link = [(i, i) for i in range(self.num_node)]
            neighbor_1base = [(1, 2), (2, 21), (3, 21), (4, 3), (5, 21),
                              (6, 5), (7, 6), (8, 7), (9, 21), (10, 9),
                              (11, 10), (12, 11), (13, 1), (14, 13), (15, 14),
                              (16, 15), (17, 1), (18, 17), (19, 18), (20, 19),
                              (22, 23), (23, 8), (24, 25), (25, 12)]
            neighbor_link = [(i - 1, j - 1) for (i, j) in neighbor_1base]  #序号从1开始，故需要 -1，所有相连的元组
            self.edge = self_link + neighbor_link
            self.center = 21 - 1
        elif layout == 'ntu_edge':
            self.num_node = 24
            self_link = [(i, i) for i in range(self.num_node)]
            neighbor_1base = [(1, 2), (3, 2), (4, 3), (5, 2), (6, 5), (7, 6),
                              (8, 7), (9, 2), (10, 9), (11, 10), (12, 11),
                              (13, 1), (14, 13), (15, 14), (16, 15), (17, 1),
                              (18, 17), (19, 18), (20, 19), (21, 22), (22, 8),
                              (23, 24), (24, 12)]
            neighbor_link = [(i - 1, j - 1) for (i, j) in neighbor_1base]
            self.edge = self_link + neighbor_link
            self.center = 2
        # elif layout=='customer settings'
        #     pass
        else:
            raise ValueError("Do Not Exist This Layout.")

    def get_adjacency(self, strategy):
        valid_hop = range(0, self.max_hop + 1, self.dilation)
        adjacency = np.zeros((self.num_node, self.num_node))
        for hop in valid_hop:
            adjacency[self.hop_dis == hop] = 1 #所有相连的节点（包括在能到达的节点）都为1
        normalize_adjacency = normalize_digraph(adjacency)

        if strategy == 'uniform':
            A = np.zeros((1, self.num_node, self.num_node))
            A[0] = normalize_adjacency
            self.A = A
        elif strategy == 'distance':
            A = np.zeros((len(valid_hop), self.num_node, self.num_node))
            for i, hop in enumerate(valid_hop): #对于不同距离的邻接矩阵进行正则化
                A[i][self.hop_dis == hop] = normalize_adjacency[self.hop_dis ==
                                                                hop]
            self.A = A
            #如果按照论文的第三种划分方式
        elif strategy == 'spatial':
            A = []
            for hop in valid_hop:
                a_root = np.zeros((self.num_node, self.num_node))
                a_close = np.zeros((self.num_node, self.num_node))
                a_further = np.zeros((self.num_node, self.num_node))
                for i in range(self.num_node):
                    for j in range(self.num_node):
                        if self.hop_dis[j, i] == hop:# 如果结点j和结点i是邻结点
                	# 比较结点i和结点j分别到中心点的距离，中心点默认为为openpose输出的1结点
                            if self.hop_dis[j, self.center] == self.hop_dis[
                                    i, self.center]:
                                a_root[j, i] = normalize_adjacency[j, i]
                            elif self.hop_dis[j, self.
                                              center] > self.hop_dis[i, self.
                                                                     center]:
                                a_close[j, i] = normalize_adjacency[j, i]
                            else:
                                a_further[j, i] = normalize_adjacency[j, i]
                if hop == 0:
                    A.append(a_root) # A的第一维第1个矩阵:self distance matrix 对角阵
                else:
                    A.append(a_root + a_close) # A的第一维第2个矩阵:列对结点到中心点的距离比行对应点到中心点的距离近或者相等（都为inf)
                    A.append(a_further) # A的第一维第3个矩阵:列对应结点到中心点的距离比行对应点到中心点的距离远
            A = np.stack(A)
            self.A = A
        else:
            raise ValueError("Do Not Exist This Strategy")


## 构建邻接矩阵 获得了带自环的邻接矩阵，非连接处是inf
def get_hop_distance(num_node, edge, max_hop=1):
    A = np.zeros((num_node, num_node))
    for i, j in edge:
        # 相连的为1
        A[j, i] = 1
        A[i, j] = 1

    # compute hop steps
    hop_dis = np.zeros((num_node, num_node)) + np.inf #25*25的矩阵，值为无限大（代表距离）
    transfer_mat = [np.linalg.matrix_power(A, d) for d in range(max_hop + 1)]
    # 即距离为0以内的矩阵（单位矩阵，自己和自己相连），距离为1以内的矩阵。。。以此类推
    # np.linalg.matrix_power()表示矩阵的次方
    arrive_mat = (np.stack(transfer_mat) > 0)#将转移矩阵值变为True和False
    for d in range(max_hop, -1, -1): #从最大距离开始到0，，如2,1,0
        hop_dis[arrive_mat[d]] = d #通过矩阵从距离远到近进行覆盖，实现一个25*25矩阵里的值为两个节点的最近距离，无法到达为inf
    return hop_dis

# 图卷积的预处理
def normalize_digraph(A):
    Dl = np.sum(A, 0)  #计算邻接矩阵的度 每个节点相连的个数
    num_node = A.shape[0]
    Dn = np.zeros((num_node, num_node))
    for i in range(num_node):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i]**(-1) #由每个点的度组成的对角矩阵 值为当前节点连接节点的倒数
    AD = np.dot(A, Dn) # 修改邻接矩阵
    return AD


def normalize_undigraph(A):
    Dl = np.sum(A, 0)
    num_node = A.shape[0]
    Dn = np.zeros((num_node, num_node))
    for i in range(num_node):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i]**(-0.5)
    DAD = np.dot(np.dot(Dn, A), Dn)
    return DAD