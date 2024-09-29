# The based unit of graph convolutional networks.

import torch
import torch.nn as nn

class ConvTemporalGraphical(nn.Module):

    r"""The basic module for applying a graph convolution.

    Args:
        in_channels (int): Number of channels in the input sequence data
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int): Size of the graph convolving kernel
        t_kernel_size (int): Size of the temporal convolving kernel
        t_stride (int, optional): Stride of the temporal convolution. Default: 1
        t_padding (int, optional): Temporal zero-padding added to both sides of
            the input. Default: 0
        t_dilation (int, optional): Spacing between temporal kernel elements.
            Default: 1
        bias (bool, optional): If ``True``, adds a learnable bias to the output.
            Default: ``True``

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
                 t_kernel_size=1,
                 t_stride=1,
                 t_padding=0,
                 t_dilation=1,
                 bias=True):
        super().__init__()

        self.kernel_size = kernel_size
        # 注意：这个keneral_size指的是空间上的kernal size，等于3，也等于划分策略划分的子集数K
        self.conv = nn.Conv2d(
            in_channels,
            out_channels * kernel_size,
            kernel_size=(t_kernel_size, 1),
            padding=(t_padding, 0),
            stride=(t_stride, 1),
            dilation=(t_dilation, 1),
            bias=bias)

    def forward(self, x, A):
        assert A.size(0) == self.kernel_size # 是指论文中多子集邻接矩阵的子集数。

        x = self.conv(x)
        # 这里输入x是(N,C,T,V),经过conv(x)之后变为（N，C*kneral_size,T,V）
        # x维度(batch_size,192,300,25),意思为 batch_size，64*3（通道*最大距离+1）,300帧，25个节点
        n, kc, t, v = x.size()
        # batch_size,通道*（最大距离+1），帧数，节点个数
        x = x.view(n, self.kernel_size, kc//self.kernel_size, t, v)
        # batch_size,最大距离+1，通道数，帧数，节点数
        # 这里把keneral_size的维度单独拿出来，变成(N,K,C,T,V)
        x = torch.einsum('nkctv,kvw->nctw', (x, A)) # 爱因斯坦约定求和法
        # 简化的矩阵点乘，对k，v求和
        # 这里相当于每个节点有64*300个特征，然后共有25个节点，64*300*25的矩阵乘上25*25的邻接矩阵，得到64*300*25的矩阵，共3个不同距离的邻接矩阵，得到3*64*300*25的结果，演着邻接矩阵个数的方向想加，得到最终结果64*300*25的矩阵。
        # 得到batch,通道，帧数，节点数
        # x维度（batch_size,64,300,25)

        return x.contiguous(), A
