from .base_options import BaseOptions


class RLOptions(BaseOptions):
    def initialize(self, parser):
        # Options for the reconstruction model
        parser = BaseOptions.initialize(self, parser)
        parser.add_argument('--ntest', type=int, default=float("inf"), help='# of test examples.')
        parser.add_argument('--results_dir', type=str, default='./results/', help='saves results here.')
        parser.add_argument('--aspect_ratio', type=float, default=1.0, help='aspect ratio of result images')
        parser.add_argument('--phase', type=str, default='test', help='train, val, test, etc')
        parser.add_argument('--which_epoch', type=str, default='latest',
                            help='which epoch to load? set to latest to use latest cached model')
        parser.add_argument('--how_many', type=int, default=50, help='how many test images to run')
        parser.set_defaults(model='test')
        # To avoid cropping, the loadSize should be the same as fineSize
        parser.set_defaults(loadSize=parser.get_default('fineSize'))
        parser.add_argument('--shuffle_testset_loader', action='store_true', help='if shuffle test set loader')

        # Reinforcement learning options
        parser.add_argument('--initial_num_lines', type=int, default=10)
        parser.add_argument('--budget', type=int, default=5)
        parser.add_argument('--rl_model_type', type=str, default='two_streams')
        parser.add_argument('--epsilon_start', type=float, default=0.99)
        parser.add_argument('--epsilon_end', type=float, default=0.01)
        parser.add_argument('--epsilon_decay', type=float, default=10000)
        parser.add_argument('--num_episodes', type=int, default=10000)
        parser.add_argument('--rl_batch_size', type=int, default=16)
        parser.add_argument('--agent_test_episode_freq', type=int, default=20)
        parser.add_argument('--target_net_update_freq', type=int, default=500)
        parser.add_argument('--gamma', type=float, default=0.999)
        parser.add_argument('--debug', dest='debug', action='store_true')
        parser.add_argument('--no_replacement_policy', dest='no_replacement_policy', action='store_true')

        self.isTrain = False
        return parser
