import torch
import torch.nn as nn
import torch.nn.functional as F
from sobel import Sobel


class L1LogLoss(nn.Module):
    def __init__(self, alpha=0.5):
        super(L1LogLoss, self).__init__()
        self.alpha = alpha

    def forward(self, pred, target):
        return torch.log(torch.abs(pred-target)+self.alpha).mean()


class GradLogLoss(nn.Module):
    def __init__(self, device, alpha=0.5):
        super(GradLogLoss, self).__init__()
        self.device = device
        self.alpha = alpha

    def forward(self, pred, target):
        sobel = Sobel(self.device)
        target_dx, target_dy = sobel(target)
        pred_dx, pred_dy = sobel(pred)
        l_dx = torch.log(torch.abs(pred_dx-target_dx)+self.alpha).mean()
        l_dy = torch.log(torch.abs(pred_dy-target_dy)+self.alpha).mean()

        return l_dx + l_dy


class BerHuLoss(nn.Module):
    def __init__(self):
        super(BerHuLoss, self).__init__()

    def forward(self, pred, target):
        diff = torch.abs(target-pred)
        c = 0.2 * torch.max(diff).data.cpu().numpy()

        # if -diff > -c, then use -diff
        # same as diff <= c, then use diff
        # third argument must be constant
        loss1 = -F.threshold(-diff, -c, 0)

        diff2 = diff**2
        c2 = c**2
        # if diff < c, loss2 = 0
        # else loss2 = diff2 + c2
        loss2 = F.threshold(diff2-c2, 0, -2*c2) + 2*c2
        loss2 /= 2*c
        return torch.mean(loss1 + loss2)


class RMSELoss(nn.Module):
    def __init__(self):
        super(BerHuLoss, self).__init__()

    def forward(self, pred, target):
        return torch.sqrt(F.mse_loss(pred, target))
