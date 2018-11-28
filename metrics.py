import torch


def mre(pred, gt):
    '''
    gt  : ground truth depth
    pred: predicted depth
    shape: (N, H, W)
    '''
    N = pred.shape[0]
    loss = 0
    for i in range(N):
        loss += torch.sum(torch.div(torch.abs(gt[i] - pred[i]),
                                    torch.sum(gt[i])))
    return loss / N


def delta(pred, gt, i):
    '''
    gt  : ground truth depth
    pred: predicted depth
    shape: (N, H, W)
    i   : order of delta
    '''
    threshold = 1.25**i
    N = pred.shape[0]
    loss = 0
    for j in range(N):
        acc = torch.max(pred[j]/gt[j], gt[j]/pred[j]) < threshold
        loss += torch.sum(torch.mean(acc.float()))
    return loss / N
