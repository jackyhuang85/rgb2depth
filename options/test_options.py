from .base_options import BaseOptions

class TrainOptions(BaseOptions):
    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        parser.add_argument('--load_model', type=str, required=True, help='The trained model to be eval.')

        self.isTrain = False
        return parser
