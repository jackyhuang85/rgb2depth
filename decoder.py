import torch
import torch.nn as nn
import torch.nn.functional as F


class DeConv(nn.Module):
    def __init__(self, in_channels, kernel_size):
        super(DeConv, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels//2, kernel_size,
                                     stride=2, padding=(kernel_size-1)//2,
                                     output_padding=kernel_size % 2, bias=False)
        self.bn = nn.BatchNorm2d(in_channels//2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.up(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class Unpool(nn.Module):
    def __init__(self, in_channels, stride=2):
        super(Unpool, self).__init__()
        self.in_channels = in_channels
        self.stride = stride
        self.weights = torch.autograd.Variable(
            torch.zeros(in_channels, 1, stride, stride))
        self.weights[:, :, 0, 0] = 1

    def forward(self, x):
        if x.is_cuda:
            self.weights = self.weights.cuda()
        return F.conv_transpose2d(x, self.weights, stride=self.stride, groups=self.in_channels)


class UpConv(nn.Module):
    def __init__(self, in_channels):
        super(UpConv, self).__init__()
        self.unpool = Unpool(in_channels)
        self.conv = nn.Conv2d(in_channels, in_channels//2, 5,
                              stride=1, padding=2, bias=False)
        self.bn = nn.BatchNorm2d(in_channels//2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.unpool(x)
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class UpProject(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpProject, self).__init__()

        self.conv11 = nn.Conv2d(in_channels, out_channels, 3, bias=False)
        self.conv12 = nn.Conv2d(in_channels, out_channels, (2, 3), bias=False)
        self.conv13 = nn.Conv2d(in_channels, out_channels, (3, 2), bias=False)
        self.conv14 = nn.Conv2d(in_channels, out_channels, 2, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv21 = nn.Conv2d(in_channels, out_channels, 3, bias=False)
        self.conv22 = nn.Conv2d(in_channels, out_channels, (2, 3), bias=False)
        self.conv23 = nn.Conv2d(in_channels, out_channels, (3, 2), bias=False)
        self.conv24 = nn.Conv2d(in_channels, out_channels, 2, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(out_channels, out_channels,
                               3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)

    def _interleaving(self, x1, x2, x3, x4):
        n, c, h, w = x1.size()

        x12 = torch.stack(
            (x1, x2),
            dim=-3
        ).permute(0, 1, 3, 4, 2).contiguous().view(n, c, h, w * 2)
        x34 = torch.stack(
            (x3, x4),
            dim=-3
        ).permute(0, 1, 3, 4, 2).contiguous().view(n, c, h, w*2)
        x = torch.stack(
            (x12, x34),
            dim=-3
        ).permute(0, 1, 3, 2, 4).contiguous().view(n, c, 2*h, 2*w)
        return x

    def forward(self, x):
        x11 = self.conv11(F.pad(x, (1, 1, 1, 1)))
        x12 = self.conv12(F.pad(x, (1, 1, 1, 0)))
        x13 = self.conv13(F.pad(x, (1, 0, 1, 1)))
        x14 = self.conv14(F.pad(x, (1, 0, 1, 0)))
        x21 = self.conv21(F.pad(x, (1, 1, 1, 1)))
        x22 = self.conv22(F.pad(x, (1, 1, 1, 0)))
        x23 = self.conv23(F.pad(x, (1, 0, 1, 1)))
        x24 = self.conv24(F.pad(x, (1, 0, 1, 0)))

        x1 = self._interleaving(x11, x12, x13, x14)
        x1 = self.bn1(x1)
        x1 = self.relu(x1)
        x1 = self.conv3(x1)
        x1 = self.bn3(x1)

        x2 = self._interleaving(x21, x22, x23, x24)
        x2 = self.bn2(x2)

        return self.relu(x1+x2)
