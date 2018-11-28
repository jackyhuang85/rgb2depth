import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import importlib
from skimage.transform import resize


def show_image_depth(image, depth, block=True):
    fig = plt.figure()
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.imshow(image)
    # ax1.imshow(image, cmap='viridis', vmin=0, vmax=80)
    plt.title('Image')
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.imshow(depth, cmap='viridis', vmin=0, vmax=10)
    plt.title('Ground truth depth')
    return fig


def resize_image_depth(image, pred, gt):
    h, w = image.shape[2:]
    image = image[0].data.cpu().numpy()
    image = np.transpose(image, (2, 1, 0))
    depth_gt = gt[0, 0].data.cpu().numpy()
    depth_gt = resize(depth_gt, (h, w), preserve_range=True)
    depth_pred = pred[0, 0].data.cpu().numpy()
    depth_pred = resize(depth_pred, (h, w), preserve_range=True)

    return image, depth_gt, depth_pred


def make_depth_fig(img, depth_gt, depth_pred):
    fig = plt.figure()
    ax1 = fig.add_subplot(1, 3, 1)
    ax1.imshow(img)
    plt.title('Image')
    ax2 = fig.add_subplot(1, 3, 2)
    ax2.imshow(depth_gt, cmap='viridis',
               vmin=0, vmax=10)
    plt.title('Ground Truth')
    ax3 = fig.add_subplot(1, 3, 3)
    ax3.imshow(depth_pred, cmap='viridis',
               vmin=0, vmax=10)
    plt.title('Predict Depth')
    return fig


def adjust_lr(optimizer, epoch, lr, reduce=0.2, step=5):
    lr = lr * (reduce**(epoch//step))
    for param in optimizer.param_groups:
        param['lr'] = lr


def create_loss(loss):
    loss_name = loss
    loss_lib = importlib.import_module('loss')

    loss = None

    for name, cls in loss_lib.__dict__.items():
        if name.lower() == (loss_name+'loss').lower() and issubclass(cls, nn.Module):
            loss = cls

    if loss is None:
        print("There does not exist %s loss function in ./loss.py" % (loss_name))
        exit(0)

    instance = loss()

    return instance

def make_error_map(img, depth_gt, depth_pred):
    fig = plt.figure()
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.imshow(img)
    plt.title('Image')
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.imshow((depth_pred-depth_gt).abs(), vmin=0, vmax=2)
    plt.title('Error map')
    return fig

