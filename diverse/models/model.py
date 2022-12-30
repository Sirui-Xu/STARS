#!/usr/bin/env python
# coding: utf-8

import torch
import torch.nn as nn
import math
import numpy as np

class ConvTemporalGraphical(nn.Module):
    #Source : https://github.com/yysijie/st-gcn/blob/master/net/st_gcn.py
    r"""The basic module for applying a graph convolution.
    Shape:
        - Input: Input graph sequence in :math:`(N, in_channels, T_{in}, V)` format
        - Output: Output graph sequence in :math:`(N, out_channels, T_{out}, V)` format
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


class ConvTemporalGraphicalV1(nn.Module):
    #Source : https://github.com/yysijie/st-gcn/blob/master/net/st_gcn.py
    r"""The basic module for applying a graph convolution.
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
                 pose_info
    ):
        super(ConvTemporalGraphicalV1,self).__init__()
        parents=pose_info['parents']
        joints_left=list(pose_info['joints_left'])
        joints_right=list(pose_info['joints_right'])
        keep_joints=pose_info['keep_joints']
        dim_use = list(keep_joints)
        # print(dim_use)
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
        self.A_s = torch.zeros((1,joints_dim,joints_dim), requires_grad=False, dtype=torch.float)
        for i, dim in enumerate(dim_use):
            self.A_s[0][i][i] = 1
            if parents[dim] in dim_use:
                parent_index = dim_use.index(parents[dim])
                self.A_s[0][i][parent_index] = 1
                self.A_s[0][parent_index][i] = 1
            if dim in joints_left:
                index = joints_left.index(dim)
                right_dim = joints_right[index]
                right_index = dim_use.index(right_dim)
                if right_dim in dim_use:
                    self.A_s[0][i][right_index] = 1
                    self.A_s[0][right_index][i] = 1

        self.joints_dim = joints_dim
        self.time_dim = time_dim

    def forward(self, x):
        A = self.A * self.A_s.to(x.device)
        x = torch.einsum('nctv,vtq->ncqv', (x, self.T))
        x = torch.einsum('nctv,tvw->nctw', (x, A))
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
                 version=0,
                 pose_info=None):
        
        super(ST_GCNN_layer,self).__init__()
        self.kernel_size = kernel_size
        assert self.kernel_size[0] % 2 == 1
        assert self.kernel_size[1] % 2 == 1
        padding = ((self.kernel_size[0] - 1) // 2,(self.kernel_size[1] - 1) // 2)
        
        if version == 0:
            self.gcn=ConvTemporalGraphical(time_dim,joints_dim) # the convolution layer
        elif version == 1:
            self.gcn = ConvTemporalGraphicalV1(time_dim,joints_dim,pose_info=pose_info)

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
                 n_pre,
                 nk,
                 input_channels,
                 input_time_frame,
                 output_time_frame,
                 st_gcnn_dropout,
                 joints_to_consider,
                 pose_info):
        
        super(Model,self).__init__()
        self.input_time_frame=input_time_frame
        self.output_time_frame=output_time_frame
        self.joints_to_consider=joints_to_consider
        self.st_gcnns=nn.ModuleList()

        nk1, nk2 = nk
        self.st_gcnns.append(ST_GCNN_layer(input_channels,128,[3,1],1,n_pre,
                                           joints_to_consider,st_gcnn_dropout,pose_info=pose_info))

        self.st_gcnns.append(ST_GCNN_layer(128,64,[3,1],1,n_pre,
                                               joints_to_consider,st_gcnn_dropout, version=1, pose_info=pose_info))
            
        self.st_gcnns.append(ST_GCNN_layer(64,128,[3,1],1,n_pre,
                                               joints_to_consider,st_gcnn_dropout, version=1, pose_info=pose_info))
                  
        self.st_gcnns.append(ST_GCNN_layer(128,64,[3,1],1,n_pre,
                                               joints_to_consider,st_gcnn_dropout, pose_info=pose_info))   
        
        self.st_gcnns.append(ST_GCNN_layer(128,128,[3,1],1,n_pre,
                                               joints_to_consider,st_gcnn_dropout, version=1, pose_info=pose_info))   
        self.st_gcnns[-1].gcn.A = self.st_gcnns[-3].gcn.A

        self.st_gcnns.append(ST_GCNN_layer(128,64,[3,1],1,n_pre,
                                               joints_to_consider,st_gcnn_dropout, pose_info=pose_info))   
        
        self.st_gcnns[-1].gcn.A = self.st_gcnns[-3].gcn.A

        self.st_gcnns.append(ST_GCNN_layer(64,128,[3,1],1,n_pre,
                                               joints_to_consider,st_gcnn_dropout, version=1, pose_info=pose_info))
        self.st_gcnns[-1].gcn.A = self.st_gcnns[-3].gcn.A

        self.st_gcnns.append(ST_GCNN_layer(128,input_channels,[3,1],1,n_pre,
                                               joints_to_consider,st_gcnn_dropout, pose_info=pose_info))     

        self.e_mu_1 = ST_GCNN_layer(64,32,[3,1],1,n_pre,
                                               joints_to_consider,st_gcnn_dropout, pose_info=pose_info)
        self.e_mu_2 = nn.Linear(32, 64)
        self.e_logvar_1 = ST_GCNN_layer(64,32,[3,1],1,n_pre,
                                        joints_to_consider,st_gcnn_dropout, pose_info=pose_info)
        self.e_logvar_2 = nn.Linear(32, 64)

        self.T1=nn.Parameter(torch.FloatTensor(nk1, 1, 128, n_pre, 1)) 
        stdv = 1. / math.sqrt(self.T1.size(1))
        self.T1.data.uniform_(-stdv,stdv)

        self.A1=nn.Parameter(torch.FloatTensor(nk2, 1, 128, 1, joints_to_consider)) 
        stdv = 1. / math.sqrt(self.A1.size(1))
        self.A1.data.uniform_(-stdv,stdv)

        self.T2=nn.Parameter(torch.FloatTensor(nk2, 1, 128, n_pre, 1)) 
        stdv = 1. / math.sqrt(self.T2.size(1))
        self.T2.data.uniform_(-stdv,stdv)

        self.A2=nn.Parameter(torch.FloatTensor(nk1, 1, 128, 1, joints_to_consider)) 
        stdv = 1. / math.sqrt(self.A2.size(1))
        self.A2.data.uniform_(-stdv,stdv)

        self.nk1 = nk1
        self.nk2 = nk2
        self.dct_m, self.idct_m = self.get_dct_matrix(self.input_time_frame + self.output_time_frame)
        self.n_pre = n_pre
        self.i1, self.j1 = torch.meshgrid(torch.arange(self.nk1), torch.arange(self.nk2))
        self.idx1 = torch.arange(self.nk1 * self.nk2).view(self.nk1, self.nk2)
        self.i2, self.j2 = torch.meshgrid(torch.arange(self.nk2), torch.arange(self.nk1))
        self.idx2 = torch.tensor([[jj * self.nk2 + ii for jj in range(self.nk1)] for ii in range(self.nk2)], dtype=torch.long).view(self.nk2, self.nk1)

    def get_dct_matrix(self, N, is_torch=True):
        dct_m = np.eye(N, dtype=np.float32)
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

    def reparameterize(self, x):
        N, C, T, V = x.shape
        x_mu = self.e_mu_1(x).mean(2).mean(2)
        x_logvar = self.e_logvar_1(x).mean(2).mean(2)
        mu = self.e_mu_2(x_mu).unsqueeze(2).unsqueeze(3).repeat((1, 1, T, V))
        logvar = self.e_logvar_2(x_logvar).unsqueeze(2).unsqueeze(3).repeat((1, 1, T, V))
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps * std, mu, logvar

    def forward(self, x, y0):
        x = x.view(x.shape[0], x.shape[1], -1, 3)
        x = x.permute(1, 3, 0, 2)
        idx_pad = list(range(self.input_time_frame)) + [self.input_time_frame - 1] * self.output_time_frame
        y = torch.zeros((x.shape[0], x.shape[1], self.output_time_frame, x.shape[3])).to(x.device)
        inp = torch.cat([x, y], dim=2).permute(0, 2, 1, 3)
        N, T, C, V = inp.shape
        dct_m = self.dct_m.to(x.device)
        idct_m = self.idct_m.to(x.device)
        inp = inp.reshape([N, T, C * V])
        inp = torch.matmul(dct_m[:self.n_pre], inp[:, idx_pad, :]).reshape([N, -1, C, V]).permute(0, 2, 1, 3)
        res = inp
        x = inp

        N, C, T, V = inp.shape

        for gcn in (self.st_gcnns[:-5]):
            x = gcn(x)

        # result = [0 for _ in range(self.nk1 * self.nk2)]
        # for i in range(self.nk1):
        #     # CT -> NCTV
        #     t = self.T1[i]
        #     for j in range(self.nk2):
        #         # CV -> NCTV
        #         a = self.A1[j]
        #         result[i * self.nk2 + j] = x + t + a
        # result = torch.cat(result, dim=0).reshape([self.nk1 * self.nk2 * N, -1, T, V]) # MN C T V
        
        # speed up by using tensor operation
        result = x + self.T1[self.i1] + self.A1[self.j1]
        result = result.reshape([self.nk1 * self.nk2 * N, -1, T, V])

        
        for gcn in (self.st_gcnns[-5:-4]):
            result = gcn(result)

        x_rand, mu, logvar = self.reparameterize(result)
        result = torch.cat([result, x_rand], dim=1)

        for gcn in (self.st_gcnns[-4:-3]):
            result = gcn(result)

        result = result.reshape([self.nk1 * self.nk2, N, -1, T, V])
        # result2 = [0 for _ in range(self.nk1 * self.nk2)]
        # for i in range(self.nk2):
        #     # CT -> NCTV
        #     t = self.T2[i]
        #     for j in range(self.nk1):
        #         # CV -> NCTV
        #         a = self.A2[j]
        #         result2[i * self.nk1 + j] = result[j * self.nk2 + i] + t + a
        # result = torch.cat(result2, dim=0).reshape([self.nk1 * self.nk2 * N, -1, T, V]) # MN C T V

        result = result[self.idx2] + self.T2[self.i2] + self.A2[self.j2]
        result = result.reshape([self.nk1 * self.nk2 * N, -1, T, V])
        
        for gcn in (self.st_gcnns[-3:]):
            result = gcn(result)

        N, C, T, V = res.shape
        result += res.repeat(self.nk1 * self.nk2, 1, 1, 1)
        x = result.permute(0, 2, 1, 3).reshape([self.nk1 * self.nk2 * N, -1, C * V])
        x_re = torch.matmul(idct_m[:, :self.n_pre], x).reshape([self.nk1 * self.nk2, N, -1, C, V])
        
        x = x_re.permute(2, 1, 0, 4, 3).contiguous().view(-1, y0.shape[1], self.nk1 * self.nk2, y0.shape[2], 1).squeeze(4)
        return x, mu, logvar