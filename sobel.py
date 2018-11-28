import torch
import torch.nn as nn
import numpy as np


class Sobel(nn.Module):
    def __init__(self, device):
        super(Sobel, self).__init__()
        self.conv1 = nn.Conv2d(1, 2, 3, stride=1, padding=1, bias=False)
        dx = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
        weight1 = torch.from_numpy(dx).float().unsqueeze(0).unsqueeze(0)
        self.conv1.weight = nn.Parameter(weight1.to(device))

        self.conv2 = nn.Conv2d(1, 2, 3, stride=1, padding=1, bias=False)
        dy = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
        weight2 = torch.from_numpy(dy).float().unsqueeze(0).unsqueeze(0)
        self.conv2.weight = nn.Parameter(weight2.to(device))

        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        out1 = self.conv1(x)
        out2 = self.conv2(x)
        return out1, out2
