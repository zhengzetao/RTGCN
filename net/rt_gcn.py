import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from net.utils.tgcn import ConvTemporalGraphical
from net.utils.graph import Graph

class Model(nn.Module):
    r"""Relational temporal graph convolutional networks.

    Args:
        in_channels (int): Number of channels in the input data, 5 in this work
        num_class (int): Number of classes for the classification task
        graph_args (dict): The arguments for building the graph
        edge_importance_weighting (bool): If ``True``, adds a learnable
            importance weighting to the edges of the graph
        **kwargs (optional): Other parameters for graph convolution units

    Shape:
        - Input: :math:`(N, in_channels, T_{in}, V_{in})`
        - Output: :math:`(N, num_class)` where
            :math:`N` is a batch size,
            :math:`T_{in}` is a length of input sequence,
            :math:`V_{in}` is the number of graph nodes,
            :math:`M_{in}` is the number of instance in a frame.
    """

    def __init__(self, in_channels, graph_args,
                 edge_importance_weighting, **kwargs):
        super().__init__()

        # load graph
        self.graph = Graph(**graph_args)
        A = torch.tensor(self.graph.A, dtype=torch.float32, requires_grad=False)
        relation = torch.tensor(self.graph.relation, dtype=torch.float32, requires_grad=False)
        self.register_buffer('A', A)
        self.register_buffer('relation', relation)
        self.edge_importance_weighting = edge_importance_weighting

        # build networks
        spatial_kernel_size = A.size(0) #3
        temporal_kernel_size = 9
        kernel_size = (temporal_kernel_size, spatial_kernel_size)
        self.data_bn = nn.BatchNorm1d(in_channels * A.size(1))
        kwargs0 = {k: v for k, v in kwargs.items() if k != 'dropout'}

        self.linear = nn.Linear(self.relation.size(2),1)
        self.rt_gcn_networks = nn.ModuleList((# st_gcn(in_channel,out_channel,kernel_size,stride,dropout,residual)
            rt_gcn(in_channels, 8, kernel_size, 2, residual=True, **kwargs0),
            # rt_gcn(64, 64, kernel_size, 1, **kwargs),
            # rt_gcn(8, 8, kernel_size, 4, **kwargs),
            # rt_gcn(16, 16, kernel_size, 2, **kwargs),
            # rt_gcn(256, 256, kernel_size, 1, **kwargs),
            # rt_gcn(32, 32, kernel_size, 1, **kwargs),
        ))

        # initialize parameters for edge importance weighting
        if self.edge_importance_weighting == 'Uniform':
            # self.edge_importance = nn.ParameterList([
            #     nn.Parameter(torch.ones(self.A.size()))
            #     for i in self.rt_gcn_networks
            # ])
            self.edge_importance = [1] * len(self.rt_gcn_networks)
        elif self.edge_importance_weighting == 'Weight':
            self.edge_importance = nn.ModuleList([
                nn.Linear(self.relation.size(2),1)
                for i in self.rt_gcn_networks
                ])
        elif self.edge_importance_weighting == 'Time-aware':
            # self.edge_importance = [1] * len(self.rt_gcn_networks)
            self.edge_importance1 = nn.ModuleList([
                nn.MultiheadAttention(embed_dim=in_channels, num_heads=1),
                nn.MultiheadAttention(embed_dim=8, num_heads=1)
                ])

            self.edge_importance2 = nn.ModuleList([
                nn.Linear(self.relation.size(2),1)
                for i in self.rt_gcn_networks
                ])
        
        # fcn for prediction
        self.fcn = nn.Conv2d(8, 1, kernel_size=1) #output_size=(N,1,1,V)

    def forward(self, x):
        # data normalization
        N, C, T, V = x.size()
        x = x.permute(0, 3, 1, 2).contiguous()
        x = x.view(N, V * C, T)
        x = self.data_bn(x)
        x = x.view(N, V, C, T)
        x = x.permute(0, 2, 3, 1).contiguous()
        x = x.view(N, C, T, V)

        # forwad for uniform strategy
        if self.edge_importance_weighting == 'Uniform':
            for gcn, importance in zip(self.rt_gcn_networks, self.edge_importance):
                x, _ = gcn(x, self.A * importance)
        # forward for weight strategy
        if self.edge_importance_weighting == 'Weight':
            S, M, D = self.relation.size()
            for gcn, importance in zip(self.rt_gcn_networks, self.edge_importance):
                x, _ = gcn(x, self.A * (importance(self.relation.view(S*M, D)).view(S,M)+torch.eye(S).cuda()))
        # forward for time-aware strategy
        if self.edge_importance_weighting == 'Time-aware':
            mask = torch.sum(self.relation, axis=2)
            mask[mask>=1] = 1
            for gcn, importance_atten, importance_rel in zip(self.rt_gcn_networks, self.edge_importance1, self.edge_importance2):
                N, C, T, V = x.size()
                S, M, D = self.relation.size()
                x = x.permute(3,0,2,1).contiguous()
                x = x.view(V, N * T, C) # transfer for the self-attention
                _, attn_weight = importance_atten(query=x, key=x, value=x, attn_mask = mask) # attn_weight shape (N*T,V,V)
                attn_weight = attn_weight.view(N, 1, T, V, V) # 1 conv kernel
                rel_weight = importance_rel(self.relation.view(S*M, D)).view(S,M)+torch.eye(S).cuda()
                weight = attn_weight * rel_weight
                x = x.view(V, N, T, C)
                x = x.permute(1, 3, 2, 0).contiguous()
                # x, _ = gcn(x, attn_weight)
                x, _ = gcn(x, weight)



        # global pooling
        x = F.avg_pool2d(x,(x.size(2),1))
        x = x.view(N,-1,1,V)
        # prediction
        x = self.fcn(x)
        # x = x.view(x.size(0), -1)
        x = x.view(x.size(0), x.size(3))

        return x

    def extract_feature(self, x):

        # data normalization
        N, C, T, V, M = x.size()
        x = x.permute(0, 4, 3, 1, 2).contiguous()
        x = x.view(N * M, V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T)
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        x = x.view(N * M, C, T, V)

        # forwad
        for gcn, importance in zip(self.rt_gcn_networks, self.edge_importance):
            x, _ = gcn(x, self.A * importance)

        _, c, t, v = x.size()
        feature = x.view(N, M, c, t, v).permute(0, 2, 3, 4, 1)

        # prediction
        x = self.fcn(x)
        output = x.view(N, M, -1, t, v).permute(0, 2, 3, 4, 1)

        return output, feature

class rt_gcn(nn.Module):
    r"""Applies a relational temporal graph convolution over an input graph sequence.

    Args:
        in_channels (int): Number of channels in the input sequence data
        out_channels (int): Number of channels produced by the convolution
        kernel_size (tuple): Size of the temporal convolving kernel and graph convolving kernel
        stride (int, optional): Stride of the temporal convolution. Default: 1
        dropout (int, optional): Dropout rate of the final output. Default: 0
        residual (bool, optional): If ``True``, applies a residual mechanism. Default: ``True``

    Shape:
        - Input[0]: Input graph sequence in :math:`(N, in_channels, T_{in}, V)` format
        - Input[1]: Input graph adjacency matrix in :math:`(K, V, V)` format
        - Output[0]: Outpu graph sequence in :math:`(N, out_channels, T_{out}, V)` format
        - Output[1]: Graph adjacency matrix for output data in :math:`(K, V, V)` format

        where
            :math:`N` is a batch size,
            :math:`K` is the spatial kernel size, as :math:`K == kernel_size[1]`,
            :math:`T_{in}/T_{out}` is a length of input/output sequence,
            :math:`V` is the number of graph nodes.

    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 dropout=0,
                 residual=True):
        super().__init__()

        assert len(kernel_size) == 2
        assert kernel_size[0] % 2 == 1
        padding = ((kernel_size[0] - 1) // 2, 0)

        self.gcn = ConvTemporalGraphical(in_channels, out_channels,
                                         kernel_size[1])

        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels,
                (kernel_size[0], 1),
                (stride, 1),
                padding,
            ),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout, inplace=True),
        )

        if not residual:
            self.residual = lambda x: 0

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x

        else:
            self.residual = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=(stride, 1)),
                nn.BatchNorm2d(out_channels),
            )

        self.relu = nn.LeakyReLU(inplace=True,negative_slope=0.2)
        # self.tanh = nn.Tanh()

    def forward(self, x, A):

        res = self.residual(x)
        x, A = self.gcn(x, A)
        x = self.tcn(x) + res

        return self.relu(x), A