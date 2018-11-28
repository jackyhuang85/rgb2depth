import numpy as np
import torch
import torch.utils.data as data
from torchvision import transforms
import matplotlib.pyplot as plt
from NYUDepth import NYUDepth
from utils import make_depth_fig, resize_image_depth, make_error_map, dump


if __name__ == "__main__":
    model_path = 'upproj.pth'
    model = torch.load(model_path)
    model.to('cpu')
    model.eval()

    train_dataset = NYUDepth(root='/data/mengli/undergrade//nyu-dataset',
                             mode='train')
    test_dataset = NYUDepth(root='/data/mengli/undergrade//nyu-dataset',
                            mode='test')
    train_loader = data.DataLoader(train_dataset,
                                   batch_size=1,
                                   shuffle=True)
    test_loader = data.DataLoader(test_dataset,
                                  batch_size=1,
                                  shuffle=True)
    print('Size of train dataset:', len(train_dataset))
    print('Size of test dataset:', len(test_dataset))

    with torch.no_grad():
        for i, (img, depth) in enumerate(train_loader):
            img, depth = img.float(), depth.float()
            depth = depth.unsqueeze(1)
            output = model(img)

            image, depth_gt, depth_pred = resize_image_depth(
                img, depth, output)
            fig = make_depth_fig(image, depth_gt.T, depth_pred.T)

            fig.suptitle('Training dataset')
            fig2, error_map = make_error_map(image, depth_gt.T, depth_pred.T)
            plt.show(block=False)

            # dump
            dump(image=image,
                 depth=depth_pred.T,
                 depth_gt=depth_gt.T,
                 error_map=error_map,
                 prefix='infer_train',
                 n=i)

            break
        for i, (img, depth) in enumerate(test_loader):
            img, depth = img.float(), depth.float()
            depth = depth.unsqueeze(1)
            output = model(img)

            image, depth_gt, depth_pred = resize_image_depth(
                img, depth, output)
            fig = make_depth_fig(image, depth_gt.T, depth_pred.T)
            fig.suptitle('Testing dataset')
            fig2 = make_error_map(image, depth_gt.T, depth_pred.T)
            plt.show()


            dump(image=image,
                 depth=depth_pred.T,
                 depth_gt=depth_gt.T,
                 error_map=error_map,
                 prefix='infer_test',
                 n=i)
            break
