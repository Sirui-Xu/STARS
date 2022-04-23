import torch
from torch import nn


class Orthogonal(nn.Module):

    def __init__(self, d):
        super(Orthogonal, self).__init__()
        self.d = d
        self.U = torch.nn.Parameter(torch.zeros((d, d)).normal_(0, 0.5))

        self.reset_parameters()

    def reset_parameters(self):
        self.U.data = self.U.data / torch.norm(self.U.data, dim=1, keepdim=True)

    def sequential_mult(self, V, X):
        for row in range(V.shape[0] - 1, -1, -1):
            X = X - 2 * (X @ V[row:row + 1, :].t() @ V[row:row + 1, :]) / \
                (V[row:row + 1, :] @ V[row:row + 1, :].t())[0, 0]
        return X

    def forward(self, X, invert=False):
        """

        @param X:
        @return:
        """
        if not invert:
            X = self.sequential_mult(self.U, X)
        else:
            X = self.inverse(X)
        return X

    def inverse(self, X):
        X = self.sequential_mult(torch.flip(self.U, dims=[0]), X)
        return X

    def lgdet(self, X):
        return 0
