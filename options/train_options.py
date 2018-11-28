from .base_options import BaseOptions

class TrainOptions(BaseOptions):
    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
        parser.add_argument('--optim', type=str, default='adam', help='adam|sgd')
        parser.add_argument('--loss', type=str, default='rmse', help='L1log | berhu | rmse | gradlog')
        parser.add_argument('--epoch', type=int, default=20, help='epoch num')
        parser.add_argument('--momentum', type=float, default=0.9, help='momentum for optimizer')
        parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay for optimizer')
        parser.add_argument('--upsample', type=str, default='upproj', help='Up sampling method used in \
                building net. [upproj | upconv | deconv2 | deconv3]')
        parser.add_argument('--save_path', type=str, default='nyu_rgb2depth.pth', help='output trained model filename')
        parser.add_argument('--checkpoint', type=int, default=5, help='Auto save the model after [X] epochs. \
                if zero, no checkpoint will set')
        self.isTrain = True
        return parser
