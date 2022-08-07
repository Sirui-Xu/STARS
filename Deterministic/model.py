#!/usr/bin/env python
# coding: utf-8


import torch
import torch.nn as nn
import math
import numpy as np




class ConvTemporalGraphical(nn.Module):
    #Source : https://github.com/yysijie/st-gcn/blob/master/net/st_gcn.py
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
        - Input: Input graph sequence in :math:`(N, in_channels, T_{in}, V)` format
        - Output: Outpu graph sequence in :math:`(N, out_channels, T_{out}, V)` format
        where
            :math:`N` is a batch size,
            :math:`K` is the spatial kernel size, as :math:`K == kernel_size[1]`,
            :math:`T_{in}/T_{out}` is a length of input/output sequence,
            :math:`V` is the number of graph nodes. 
    """
    def __init__(self,
                 time_dim,
                 joints_dim
    ):
        super(ConvTemporalGraphical,self).__init__()
        
        self.A=nn.Parameter(torch.FloatTensor(time_dim, joints_dim,joints_dim)) #learnable, graph-agnostic 3-d adjacency matrix(or edge importance matrix)
        stdv = 1. / math.sqrt(self.A.size(1))
        self.A.data.uniform_(-stdv,stdv)

        self.T=nn.Parameter(torch.FloatTensor(joints_dim, time_dim, time_dim)) 
        stdv = 1. / math.sqrt(self.T.size(1))
        self.T.data.uniform_(-stdv,stdv)
        '''
        self.prelu = nn.PReLU()
        
        self.Z=nn.Parameter(torch.FloatTensor(joints_dim, joints_dim, time_dim, time_dim)) 
        stdv = 1. / math.sqrt(self.Z.size(2))
        self.Z.data.uniform_(-stdv,stdv)
        '''
        self.joints_dim = joints_dim
        self.time_dim = time_dim

    def forward(self, x):
        x = torch.einsum('nctv,vtq->ncqv', (x, self.T))
        ## x=self.prelu(x)
        x = torch.einsum('nctv,tvw->nctw', (x, self.A))
        ## x = torch.einsum('nctv,wvtq->ncqw', (x, self.Z))
        return x.contiguous() 


class ConvTemporalGraphicalEnhanced(nn.Module):
    #Source : https://github.com/yysijie/st-gcn/blob/master/net/st_gcn.py
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
        - Input: Input graph sequence in :math:`(N, in_channels, T_{in}, V)` format
        - Output: Outpu graph sequence in :math:`(N, out_channels, T_{out}, V)` format
        where
            :math:`N` is a batch size,
            :math:`K` is the spatial kernel size, as :math:`K == kernel_size[1]`,
            :math:`T_{in}/T_{out}` is a length of input/output sequence,
            :math:`V` is the number of graph nodes. 
    """
    def __init__(self,
                 time_dim,
                 joints_dim,
                 dim_used = [ 2,  3,  4,  5,  7,  8,  9, 10, 12, 13, 14, 15, 17,
                             18, 19, 21, 22, 25, 26, 27, 29, 30],
                 parents=[-1, 0, 1, 2, 3, 4, 0, 6, 7, 8, 9, 0, 11, 12, 13, 14, 12,
                                          16, 17, 18, 19, 20, 19, 22, 12, 24, 25, 26, 27, 28, 27, 30],
                 joints_left=[6, 7, 8, 9, 10, 16, 17, 18, 19, 20, 21, 22, 23],
                 joints_right=[1, 2, 3, 4, 5, 24, 25, 26, 27, 28, 29, 30, 31],
                 version='long'
    ):
        super(ConvTemporalGraphicalEnhanced,self).__init__()
        
        self.A=nn.Parameter(torch.FloatTensor(time_dim, joints_dim, joints_dim)) #learnable, graph-agnostic 3-d adjacency matrix(or edge importance matrix)
        stdv = 1. / math.sqrt(self.A.size(1))
        self.A.data.uniform_(-stdv,stdv)

        self.T=nn.Parameter(torch.FloatTensor(joints_dim, time_dim, time_dim)) 
        stdv = 1. / math.sqrt(self.T.size(1))
        self.T.data.uniform_(-stdv,stdv)
        '''
        self.prelu = nn.PReLU()
        
        self.Z=nn.Parameter(torch.FloatTensor(joints_dim, joints_dim, time_dim, time_dim)) 
        stdv = 1. / math.sqrt(self.Z.size(2))
        self.Z.data.uniform_(-stdv,stdv)
        '''
        self.A_s = torch.zeros((1,joints_dim,joints_dim), requires_grad=False)
        for i, dim in enumerate(dim_used):
            self.A_s[0][i][i] = 1
            if parents[dim] in dim_used:
                parent_index = dim_used.index(parents[dim])
                self.A_s[0][i][parent_index] = 1
                self.A_s[0][parent_index][i] = 1
            if dim in joints_left:
                index = joints_left.index(dim)
                right_dim = joints_right[index]
                right_index = dim_used.index(right_dim)
                if right_dim in dim_used:
                    self.A_s[0][i][right_index] = 1
                    self.A_s[0][right_index][i] = 1
        self.T_s = torch.zeros((1,time_dim,time_dim), requires_grad=False)
        if version == 'long':
            for i in range(time_dim):
                if i > 0:
                    self.T_s[0][i-1][i] = 1
                    self.T_s[0][i][i-1] = 1

                if i < time_dim - 1:
                    self.T_s[0][i+1][i] = 1
                    self.T_s[0][i][i+1] = 1
                
                self.T_s[0][i][i] = 1
        elif version == 'short':
            self.T_s = self.T_s + 1
        else:
            raise Exception("model type should be long or short")

        self.joints_dim = joints_dim
        self.time_dim = time_dim

    def forward(self, x):
        T = self.T * self.T_s.to(x.device)
        A = self.A * self.A_s.to(x.device)
        x = torch.einsum('nctv,vtq->ncqv', (x, T))
        ## x=self.prelu(x)
        x = torch.einsum('nctv,tvw->nctw', (x, A))
        ## x = torch.einsum('nctv,wvtq->ncqw', (x, self.Z))
        return x.contiguous() 

class ST_GCNN_layer(nn.Module):
    """
    Shape:
        - Input[0]: Input graph sequence in :math:`(N, in_channels, T_{in}, V)` format
        - Input[1]: Input graph adjacency matrix in :math:`(K, V, V)` format
        - Output[0]: Outpu graph sequence in :math:`(N, out_channels, T_{out}, V)` format
        where
            :math:`N` is a batch size,
            :math:`K` is the spatial kernel size, as :math:`K == kernel_size[1]`,
            :math:`T_{in}/T_{out}` is a length of input/output sequence,
            :math:`V` is the number of graph nodes. 
            :in_channels= dimension of coordinates
            : out_channels=dimension of coordinates
            +
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 time_dim,
                 joints_dim,
                 dropout,
                 bias=True,
                 version='full',
                 dim_used=None):
        
        super(ST_GCNN_layer,self).__init__()
        self.kernel_size = kernel_size
        assert self.kernel_size[0] % 2 == 1
        assert self.kernel_size[1] % 2 == 1
        padding = ((self.kernel_size[0] - 1) // 2,(self.kernel_size[1] - 1) // 2)
        
        if version == 'full':
            self.gcn=ConvTemporalGraphical(time_dim,joints_dim) # the convolution layer
        else:
            self.gcn=ConvTemporalGraphicalEnhanced(time_dim,joints_dim,dim_used=dim_used,version=version)

        self.tcn = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                (self.kernel_size[0], self.kernel_size[1]),
                (stride, stride),
                padding,
            ),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout, inplace=True),
        )       
        
        if stride != 1 or in_channels != out_channels: 

            self.residual=nn.Sequential(nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=(1, 1)),
                nn.BatchNorm2d(out_channels),
            )
            
            
        else:
            self.residual=nn.Identity()
        
        
        self.prelu = nn.PReLU()

        

    def forward(self, x):
     #   assert A.shape[0] == self.kernel_size[1], print(A.shape[0],self.kernel_size)
        res=self.residual(x)
        x=self.gcn(x) 
        x=self.tcn(x)
        x=x+res
        x=self.prelu(x)
        return x


class Model(nn.Module):
    """ 
    Shape:
        - Input[0]: Input sequence in :math:`(N, in_channels,T_in, V)` format
        - Output[0]: Output sequence in :math:`(N,T_out,in_channels, V)` format
        where
            :math:`N` is a batch size,
            :math:`T_{in}/T_{out}` is a length of input/output sequence,
            :math:`V` is the number of graph nodes. 
            :in_channels=number of channels for the coordiantes(default=3)
            +
    """

    def __init__(self,
                 input_channels,
                 input_time_frame,
                 output_time_frame,
                 st_gcnn_dropout,
                 dim_used,
                 n_pre=20,
                 bias=True,
                 version='long'):
        
        super(Model,self).__init__()
        self.input_time_frame=input_time_frame
        self.output_time_frame=output_time_frame
        dim_used = sorted([dim_used[i] // 3 for i in range(0, dim_used.shape[0], 3)])
        joints_to_consider=len(dim_used)
        self.joints_to_consider=joints_to_consider
        self.st_gcnns=nn.ModuleList()
        self.st_gcnns.append(ST_GCNN_layer(input_channels,128,[3,1],1,n_pre,
                                           joints_to_consider,st_gcnn_dropout,version='full',dim_used=dim_used))

        self.st_gcnns.append(ST_GCNN_layer(128,64,[3,1],1,n_pre,
                                               joints_to_consider,st_gcnn_dropout,version=version,dim_used=dim_used))
            
        self.st_gcnns.append(ST_GCNN_layer(64,128,[3,1],1,n_pre,
                                               joints_to_consider,st_gcnn_dropout,version=version,dim_used=dim_used))
                                       
        self.st_gcnns.append(ST_GCNN_layer(128,64,[3,1],1,n_pre,
                                               joints_to_consider,st_gcnn_dropout,version=version,dim_used=dim_used))   
        
        self.st_gcnns[-1].gcn.A = self.st_gcnns[-3].gcn.A
        # self.st_gcnns[-1].gcn.T = self.st_gcnns[-3].gcn.T

        self.st_gcnns.append(ST_GCNN_layer(64,128,[3,1],1,n_pre,
                                               joints_to_consider,st_gcnn_dropout,version=version,dim_used=dim_used))
        self.st_gcnns[-1].gcn.A = self.st_gcnns[-3].gcn.A
        # self.st_gcnns[-1].gcn.T = self.st_gcnns[-3].gcn.T

        self.st_gcnns.append(ST_GCNN_layer(128,64,[3,1],1,n_pre,
                                               joints_to_consider,st_gcnn_dropout,version=version,dim_used=dim_used))   
        
        self.st_gcnns[-1].gcn.A = self.st_gcnns[-3].gcn.A
        # self.st_gcnns[-1].gcn.T = self.st_gcnns[-3].gcn.T

        self.st_gcnns.append(ST_GCNN_layer(64,128,[3,1],1,n_pre,
                                               joints_to_consider,st_gcnn_dropout,version=version,dim_used=dim_used))
        
        self.st_gcnns[-1].gcn.A = self.st_gcnns[-3].gcn.A

        self.st_gcnns.append(ST_GCNN_layer(128,input_channels,[3,1],1,n_pre,
                                               joints_to_consider,st_gcnn_dropout,version='full',dim_used=dim_used))  
        self.st_gcnns[-1].gcn.A = self.st_gcnns[0].gcn.A    

        self.dct_m, self.idct_m = self.get_dct_matrix(self.input_time_frame + self.output_time_frame)
        self.n_pre = n_pre

    def get_dct_matrix(self, N, is_torch=True):
        dct_m = np.eye(N)
        for k in np.arange(N):
            for i in np.arange(N):
                w = np.sqrt(2 / N)
                if k == 0:
                    w = np.sqrt(1 / N)
                dct_m[k, i] = w * np.cos(np.pi * (i + 1 / 2) * k / N)
        idct_m = np.linalg.inv(dct_m)
        if is_torch:
            dct_m = torch.from_numpy(dct_m)
            idct_m = torch.from_numpy(idct_m)
        return dct_m, idct_m     

    def forward(self, x):
        idx_pad = list(range(self.input_time_frame)) + [self.input_time_frame - 1] * self.output_time_frame
        y = torch.zeros((x.shape[0], x.shape[1], self.output_time_frame, x.shape[3])).to(x.device)
        inp = torch.cat([x, y], dim=2).permute(0, 2, 1, 3)
        N, T, C, V = inp.shape
        dct_m = self.dct_m.to(x.device).float()
        idct_m = self.idct_m.to(x.device).float()
        inp = inp.reshape([N, T, C * V])
        inp = torch.matmul(dct_m[:self.n_pre], inp[:, idx_pad, :]).reshape([N, -1, C, V]).permute(0, 2, 1, 3)
        res = inp
        x = inp

        for gcn in (self.st_gcnns):
            x = gcn(x)

        x += res
        x = x.permute(0, 2, 1, 3).reshape([N, -1, C * V])
        x_re = torch.matmul(idct_m[:, :self.n_pre], x).reshape([N, T, C, V])
        x = x_re
        
        return x[:, self.input_time_frame:], x_re