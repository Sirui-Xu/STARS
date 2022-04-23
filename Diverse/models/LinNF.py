import torch
import torch.nn as nn

from torch.nn import functional as F

from models import Orthogonal
from torch.nn.parameter import Parameter
import math


class LinQR(nn.Module):
    """
    Implementation of the additive coupling layer from section 3.2 of the NICE
    paper.
    """

    def __init__(self, data_dim):
        super().__init__()

        self.Q = Orthogonal.Orthogonal(d=data_dim)
        self.R = Parameter(torch.Tensor(data_dim, data_dim))
        self.bias = Parameter(torch.Tensor(data_dim))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.R, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.R)
            bound = 1 / math.sqrt(fan_in)
            torch.nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x, logdet):
        """
        x,x_cond: [bs,data_dim]

        """
        diagR = torch.diag(self.R)
        R = torch.triu(self.R, diagonal=1) + torch.diag(torch.exp(diagR))
        x = x.matmul(R.t())
        x = self.Q(x)
        x = x + self.bias
        logdet += diagR[None, :].repeat([x.shape[0], 1]).sum(dim=1)
        return x, logdet

    def inverse(self, x, logdet):
        """
        x,x_cond: [bs,data_dim]

        """

        x = x - self.bias
        x = self.Q.inverse(x)
        diagR = torch.diag(self.R)
        R = torch.triu(self.R, diagonal=1) + torch.diag(torch.exp(diagR))
        invR = torch.inverse(R)
        x = x.matmul(invR.t())

        logdet += diagR[None, :].repeat([x.shape[0], 1]).sum(dim=1)
        return x, logdet


class prelu(nn.Module):

    def __init__(self, num_parameters: int = 1, init: float = 0.25):
        self.num_parameters = num_parameters
        super(prelu, self).__init__()
        self.weight = Parameter(torch.Tensor(num_parameters).fill_(init))

    def forward(self, input, logdet):
        s = torch.zeros_like(input)
        s[input < 0] = torch.log(self.weight)
        logdet += torch.sum(s, dim=1)
        return F.prelu(input, self.weight), logdet

    def inverse(self, input, logdet):
        s = torch.zeros_like(input)
        s[input < 0] = torch.log(self.weight)
        logdet += torch.sum(s, dim=1)
        return F.prelu(input, 1 / self.weight), logdet


class LinNF(nn.Module):
    def __init__(self, data_dim, num_layer=5, with_prelu=True):
        super().__init__()
        self.num_layer = num_layer
        self.w1 = nn.ModuleList()
        for i in range(num_layer - 1):
            self.w1.append(LinQR(data_dim=data_dim))
            if with_prelu:
                self.w1.append(prelu())
        self.w1.append(LinQR(data_dim=data_dim))

    def forward(self, x):

        z = x
        log_det_jacobian = 0

        for i, w in enumerate(self.w1):
            z, log_det_jacobian = w(z, log_det_jacobian)

        return z, log_det_jacobian

    def inverse(self, z):
        x = z
        log_det_jacobian = 0

        for i in range(len(self.w1) - 1, -1, -1):
            x, log_det_jacobian = self.w1[i].inverse(x, log_det_jacobian)

        return x, log_det_jacobian


if __name__ == '__main__':
    bs = 32
    node_n = 48
    data_dim = 25
    hidden_dim = 128
    con_dim = 100
    num_flow_layer = 10
    num_ds_layer = 6

    sf = LinNF(data_dim=data_dim)
    # a = sf.prior.sample([10000, 48, 25])
    # print(torch.mean(sf.prior.log_prob(a).sum([1, 2])))
    sf.double()
    sf.cuda()
    for i in range(10):
        x = torch.randn([bs, data_dim]).double().cuda()

        y1, logdet = sf(x)
        x1, logdet = sf.inverse(y1)
        err = (x1 / x - 1).abs().max()
        print(err)
        print(1)
