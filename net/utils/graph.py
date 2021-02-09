import numpy as np
import pickle

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
                 market='NASDAQ',
                 strategy='uniform',
                 relation_path=None):
        # self.max_hop = max_hop
        # self.dilation = dilation

        self.get_edge(market, relation_path)
        # self.hop_dis = get_hop_distance(
        #     self.num_node, self.edge, max_hop=max_hop)
        # print(self.hop_dis)
        self.get_adjacency(strategy)

    def __str__(self):
        return self.A

    def get_edge(self, market, relation_path):
        if market == 'NASDAQ':
            self.num_node = 854
            self_link = np.eye(self.num_node)
            raw_relation = np.load(relation_path)
            # abandon the last dimension of the raw relation as it is the self connected
            self.relation = raw_relation[:,:,:-1]
            _relation = np.sum(self.relation, axis=2)
            _relation[_relation>=1] = 1
            self.edge = self_link + _relation

        elif market == 'NYSE':
            self.num_node = 1405
            # self_link = [(i, i) for i in range(self.num_node)]
            # neighbor_1base = [(1, 2), (2, 21), (3, 21), (4, 3), (5, 21),
            #                   (6, 5), (7, 6), (8, 7), (9, 21), (10, 9),
            #                   (11, 10), (12, 11), (13, 1), (14, 13), (15, 14),
            #                   (16, 15), (17, 1), (18, 17), (19, 18), (20, 19),
            #                   (22, 23), (23, 8), (24, 25), (25, 12)]
            # neighbor_link = [(i - 1, j - 1) for (i, j) in neighbor_1base]
            self_link = np.eye(self.num_node)
            raw_relation = np.load(relation_path)
            self.relation = raw_relation[:,:,:-1]
            _relation = np.sum(self.relation, axis=2)
            _relation[_relation>=1] = 1
            self.edge = self_link + _relation

        # elif market == 'ntu_edge':
        #     self.num_node = 24
        #     self_link = [(i, i) for i in range(self.num_node)]
        #     neighbor_1base = [(1, 2), (3, 2), (4, 3), (5, 2), (6, 5), (7, 6),
        #                       (8, 7), (9, 2), (10, 9), (11, 10), (12, 11),
        #                       (13, 1), (14, 13), (15, 14), (16, 15), (17, 1),
        #                       (18, 17), (19, 18), (20, 19), (21, 22), (22, 8),
        #                       (23, 24), (24, 12)]
        #     neighbor_link = [(i - 1, j - 1) for (i, j) in neighbor_1base]
        #     self.edge = self_link + neighbor_link
        #     self.center = 2
        else:
            raise ValueError("Do Not Exist This Layout.")

    def get_adjacency(self, strategy):
        # valid_hop = range(0, self.max_hop + 1, self.dilation)
        # adjacency = np.zeros((self.num_node, self.num_node))
        adjacency = self.edge
        # for hop in valid_hop:
        #     adjacency[self.hop_dis == hop] = 1
        # normalize_adjacency = normalize_digraph(adjacency)
        # normalize_adjacency = adjacency

        if strategy == 'uniform':
            normalize_adjacency = normalize_digraph(adjacency)
            A = np.zeros((1, self.num_node, self.num_node))
            A[0] = normalize_adjacency
            self.A = A
        elif strategy == 'distance':
            A = np.zeros((len(valid_hop), self.num_node, self.num_node))
            A[0] = adjacency
            # for i, hop in enumerate(valid_hop):
            #     A[i][self.hop_dis == hop] = normalize_adjacency[self.hop_dis ==
            #                                                     hop]
            self.A = A
        # elif strategy == 'spatial':
        #     A = []
        #     for hop in valid_hop:
        #         a_root = np.zeros((self.num_node, self.num_node))
        #         a_close = np.zeros((self.num_node, self.num_node))
        #         a_further = np.zeros((self.num_node, self.num_node))
        #         for i in range(self.num_node):
        #             for j in range(self.num_node):
        #                 if self.hop_dis[j, i] == hop:
        #                     if self.hop_dis[j, self.center] == self.hop_dis[
        #                             i, self.center]:
        #                         a_root[j, i] = normalize_adjacency[j, i]
        #                     elif self.hop_dis[j, self.
        #                                       center] > self.hop_dis[i, self.
        #                                                              center]:
        #                         a_close[j, i] = normalize_adjacency[j, i]
        #                     else:
        #                         a_further[j, i] = normalize_adjacency[j, i]
        #         if hop == 0:
        #             A.append(a_root)
        #         else:
        #             A.append(a_root + a_close)
        #             A.append(a_further)
        #     A = np.stack(A)
        #     print(A)
        #     self.A = A
        else:
            raise ValueError("Do Not Exist This Strategy")


# def get_hop_distance(num_node, edge, max_hop=1):
#     A = np.zeros((num_node, num_node))
#     for i, j in edge:
#         A[j, i] = 1
#         A[i, j] = 1

#     # compute hop steps
#     hop_dis = np.zeros((num_node, num_node)) + np.inf
#     transfer_mat = [np.linalg.matrix_power(A, d) for d in range(max_hop + 1)]
#     arrive_mat = (np.stack(transfer_mat) > 0)
#     for d in range(max_hop, -1, -1):
#         hop_dis[arrive_mat[d]] = d
#     return hop_dis


def normalize_digraph(A):
    Dl = np.sum(A, 0)
    num_node = A.shape[0]
    Dn = np.zeros((num_node, num_node))
    for i in range(num_node):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i]**(-1)
    AD = np.dot(A, Dn)
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

if __name__ == '__main__':
    graph = Graph(market='NASDAQ',
                 strategy='uniform',
                 relation_path='../../data/NASDAQ/NASDAQ_relation.npy')