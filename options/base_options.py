import argparse
import os
import torch

class BaseOptions():
    def __init__(self):
        self.initialized = False

    def initialize(self, parser):
        parser.add_argument('--dataroot', type=str, required=True, help='path to data root[train+test] dir')
        parser.add_argument('--dataset', type=str, required=True, help='The dataloader will be used. [make3d | nyu]')
        parser.add_argument('--batch_size', type=int, default=16, help='input batch size')
        parser.add_argument('--name', type=str, default='train', help='for tensorboardX writer name')

        self.initialized = True
        return parser

    def gather_options(self):
        if not self.initialized:
            parser = argparse.ArgumentParser(
                        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)

        opt, _ = parser.parse_known_args()
        self.parser = parser

        return parser.parse_args()


    def print_options(self, opt):
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)


    def parse(self):
        opt = self.gather_options()
        opt.isTrain = self.isTrain

        self.print_options(opt)

        self.opt = opt
        return self.opt
