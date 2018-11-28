import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50
from decoder import DeConv, UpConv, UpProject


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels*self.expansion, 1,
                               bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels*self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.stride = stride
        self.downsample = downsample

    def forward(self, x):
        residual = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.bn3(x)
        if self.downsample is not None:
            residual = self.downsample(residual)

        x += residual
        x = self.relu(x)
        return x


class ResNetDepth(nn.Module):
    def __init__(self, decoder='upproj'):
        super(ResNetDepth, self).__init__()
        pretrained = resnet50(pretrained=True)

        self.in_channels = 64

        # ResNet
        # self.conv1 = nn.Conv2d(3, 64, kernel_size=7,
        #                        stride=2, padding=3, bias=False)
        # self.bn1 = nn.BatchNorm2d(64)
        # self.relu = nn.ReLU(inplace=True)
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # self.layer1 = self._make_layer(Bottleneck, 64, 3)
        # self.layer2 = self._make_layer(Bottleneck, 128, 4, stride=2)
        # self.layer3 = self._make_layer(Bottleneck, 256, 6, stride=2)
        # self.layer4 = self._make_layer(Bottleneck, 512, 3, stride=2)
        self.conv1 = pretrained._modules['conv1']
        self.bn1 = pretrained._modules['bn1']
        self.relu = pretrained._modules['relu']
        self.maxpool = pretrained._modules['maxpool']
        self.layer1 = pretrained._modules['layer1']
        self.layer2 = pretrained._modules['layer2']
        self.layer3 = pretrained._modules['layer3']
        self.layer4 = pretrained._modules['layer4']

        # del pretrained

        # up sampling
        self.conv2 = nn.Conv2d(2048, 1024, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(1024)

        if decoder == 'upproj':
            self.up1 = UpProject(1024, 512)
            self.up2 = UpProject(512, 256)
            self.up3 = UpProject(256, 128)
            self.up4 = UpProject(128, 64)
        elif decoder == 'upconv':
            self.up1 = UpConv(1024)
            self.up2 = UpConv(512)
            self.up3 = UpConv(256)
            self.up4 = UpConv(128)
        elif decoder == 'deconv2':
            self.up1 = DeConv(1024, 2)
            self.up2 = DeConv(512, 2)
            self.up3 = DeConv(256, 2)
            self.up4 = DeConv(128, 2)
        elif decoder == 'deconv3':
            self.up1 = DeConv(1024, 3)
            self.up2 = DeConv(512, 3)
            self.up3 = DeConv(256, 3)
            self.up4 = DeConv(128, 3)
        else:
            raise ValueError(
                'decoder must be \'upproj\', \'upconv\',\'deconv2\',\'deconv3\'')

        # self.drop = nn.Dropout2d(0.5)

        self.conv3 = nn.Conv2d(64, 1, 3, padding=1, bias=False)

        # self.conv2.apply(weights_init)
        # self.bn2.apply(weights_init)
        # self.up1.apply(weights_init)
        # self.up2.apply(weights_init)
        # self.up3.apply(weights_init)
        # self.up4.apply(weights_init)
        # self.conv3.apply(weights_init)
    # def _make_layer(self, block, out_channels, blocks, stride=1):
    #     downsample = None
    #     if stride != 1 or self.in_channels != out_channels * block.expansion:
    #         downsample = nn.Sequential(
    #             nn.Conv2d(self.in_channels, out_channels * block.expansion, 1,
    #                       stride=stride, bias=False),
    #             nn.BatchNorm2d(out_channels * block.expansion),
    #         )

    #     layers = []
    #     layers.append(block(self.in_channels, out_channels,
    #                         stride, downsample))
    #     self.in_channels = out_channels * block.expansion
    #     for i in range(1, blocks):
    #         layers.append(block(self.in_channels, out_channels))

    #     return nn.Sequential(*layers)

    def forward(self, x):
        # resnet
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # upsampling
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)

        # x = self.drop(x)
        # import pdb
        # pdb.set_trace()
        x = self.conv3(x)
        x = self.relu(x)

        x = F.interpolate(x, (304, 228), mode='bilinear',
                          align_corners=True)
        return x


# def weights_init(m):
#     # Initialize filters with Gaussian random weights
#     if isinstance(m, nn.Conv2d):
#         n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
#         m.weight.data.normal_(0, (2./n)**0.5)
#         if m.bias is not None:
#             m.bias.data.zero_()
#     elif isinstance(m, nn.ConvTranspose2d):
#         n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
#         m.weight.data.normal_(0, (2./n)**0.5)
#         if m.bias is not None:
#             m.bias.data.zero_()
#     elif isinstance(m, nn.BatchNorm2d):
#         m.weight.data.fill_(1)
#         m.bias.data.zero_()


class DepthNN(nn.Module):
    def __init__(self):
        super(DepthNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 7, padding=3)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, 3)
        self.bn4 = nn.BatchNorm2d(64)
        self.conv5 = nn.Conv2d(64, 64, 1)
        self.bn5 = nn.BatchNorm2d(64)
        self.conv6 = nn.Conv2d(64, 64, 3)
        self.bn6 = nn.BatchNorm2d(64)
        self.conv7 = nn.Conv2d(64, 128, 3)
        self.bn7 = nn.BatchNorm2d(128)
        self.conv8 = nn.Conv2d(128, 128, 3)
        self.bn8 = nn.BatchNorm2d(128)
        self.up1 = nn.ConvTranspose2d(128, 64, 3)
        self.up_conv1 = nn.Conv2d(64, 64, 1)
        self.up_bn1 = nn.BatchNorm2d(64)
        self.up2 = nn.ConvTranspose2d(64, 64, 3)
        self.up_conv2 = nn.Conv2d(64, 64, 1)
        self.up_bn2 = nn.BatchNorm2d(64)
        self.up3 = nn.ConvTranspose2d(64, 64, 3)
        self.up_conv3 = nn.Conv2d(64, 64, 1)
        self.up_bn3 = nn.BatchNorm2d(64)
        self.up = nn.ConvTranspose2d(64, 8, 3)
        self.conv = nn.Conv2d(8, 1, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(self.bn1(x))
        x = self.conv2(x)
        x = F.relu(self.bn2(x))
        x = self.conv3(x)
        x = F.relu(self.bn3(x))
        x = self.conv4(x)
        x = F.relu(self.bn4(x))
        x = self.conv5(x)
        x = F.relu(self.bn5(x))
        x = self.conv6(x)
        x = F.relu(self.bn6(x))
        x = self.conv7(x)
        x = F.relu(self.bn7(x))
        x = self.conv8(x)
        x = F.relu(self.bn8(x))
        x = self.up1(x)
        x = self.up_conv1(x)
        x = F.relu(self.up_bn1(x))
        x = self.up2(x)
        x = self.up_conv2(x)
        x = F.relu(self.up_bn2(x))
        x = self.up3(x)
        x = self.up_conv3(x)
        x = F.relu(self.up_bn3(x))
        x = self.up(x)
        x = self.conv(x)
        return x.squeeze(1)
