import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data as data
from tensorboardX import SummaryWriter
from NYUDepth import NYUDepth
from model import ResNetDepth
# from utils import resize_image_depth, make_depth_fig, adjust_lr
from utils import *
# from loss import L1LogLoss, GradLogLoss, BerHuLoss
from options.train_options import TrainOptions

def train(train_loader, model, optimizer, criterion, device, writer, epoch):
    model.train()
    total_loss = 0
    for i, (img, depth) in enumerate(train_loader):
        img, depth = img.to(device).float(), depth.to(device).float()
        depth = depth.unsqueeze(1)
        output = model(img)

        output[depth == 0] = 0

        optimizer.zero_grad()
        loss = criterion(depth, output)
        total_loss += loss
        loss.backward()
        optimizer.step()

        if i % 10 == 0:
            print('Iteration: %d, loss = %f' % (i, loss))
            niter = epoch * len(train_loader) + i
            writer.add_scalar('Train/Loss', loss, niter)

        if i == 0:
            image, depth_gt, depth_pred = resize_image_depth(
                img, depth, output)
            fig = make_depth_fig(image, depth_gt.T, depth_pred.T)
            writer.add_figure('Train/depth', fig, global_step=epoch)
    total_loss /= len(train_loader)
    print('Training loss = %f at epoch %d' % (total_loss, epoch))


def validate(test_loader, model, criterion, device, writer, epoch):
    model.eval()
    loss = 0
    with torch.no_grad():
        for i, (img, depth) in enumerate(test_loader):
            img, depth = img.to(device).float(), depth.to(device).float()
            depth = depth.unsqueeze(1)
            output = model(img)

            loss += criterion(output, depth)
        loss = loss/len(test_loader)

        print('Validation loss = %f' % (loss,))
        writer.add_scalar('Validation/Loss', loss, epoch)

        image, depth_gt, depth_pred = resize_image_depth(img, depth, output)
        fig = make_depth_fig(image, depth_gt.T, depth_pred.T)
        writer.add_figure('Validation/depth', fig, global_step=epoch)


if __name__ == "__main__":
    opt = TrainOptions().parse()
    batch_size = opt.batch_size
    epoch = opt.epoch
    lr = opt.lr
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Using:', device)

    train_dataset = NYUDepth(root=opt.dataroot,
                             mode='train')
    train_loader = data.DataLoader(train_dataset,
                                   batch_size=batch_size,
                                   shuffle=True)
    print('Size of training dataset:', len(train_dataset))

    test_dataset = NYUDepth(root=opt.dataroot,
                            mode='test')
    test_loader = data.DataLoader(test_dataset,
                                  batch_size=batch_size,
                                  shuffle=False)
    print('Size of validation dataset:', len(test_dataset))

    model = ResNetDepth(decoder=opt.upsample).to(device)

    writer = SummaryWriter(log_dir=opt.name)
    optimizer = optim.SGD(model.parameters(), lr=lr,
                          momentum=opt.momentum, weight_decay=opt.weight_decay)
    criterion = create_loss(opt.loss)
    for i in range(epoch):
        # if (i+1) % 5 == 0:
        #     lr = lr * 0.2
        adjust_lr(optimizer, i, lr, reduce=0.2, step=5)

        print('Epoch:', i)
        train(train_loader, model, optimizer, criterion, device, writer, i)

        if (i+1) % 5 == 0 and opt.checkpoint != 0:
            torch.save(model, opt.save_path)

        print('Validation at epoch:', i)
        validate(test_loader, model, criterion, device, writer, i)

    writer.close()

    torch.save(model, opt.save_path)
