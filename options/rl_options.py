from .base_options import BaseOptions


class RLOptions(BaseOptions):

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        # Environment options
        parser.add_argument(
            '--reconstructor_dir',
            type=str,
            default=None,
            help='Directory where reconstructor is stored.')
        parser.add_argument(
            '--obs_type',
            choices=['fourier_space', 'image_space', 'mask_embedding'],
            default='fourier_space')
        parser.add_argument('--obs_to_numpy', action='store_true')
        parser.add_argument(
            '--policy',
            choices=['dqn', 'random', 'lowfirst', 'evaluator_net', 'evaluator++'],
            default='random')
        parser.add_argument('--initial_num_lines', type=int, default=10)
        parser.add_argument('--budget', type=int, default=5)
        parser.add_argument('--num_test_images', type=int, default=200)
        parser.add_argument('--num_train_images', type=int, default=10000000)
        parser.add_argument(
            '--use_reconstructions', dest='use_reconstructions', action='store_false')
        parser.add_argument(
            '--use_score_as_reward',
            dest='use_score_as_reward',
            action='store_true',
            help='If true, the reward is the score (e.g., MSE, SSIM). Otherwise, the reward is the '
            'decrease in score with respect to previous step.')
        parser.add_argument('--debug', dest='debug', action='store_true')
        parser.add_argument('--seed', type=int, default=0)
        parser.add_argument(
            '--sequential_images',
            dest='sequential_images',
            action='store_true',
            help='If true, then the reconstruction environment\'s reset() function '
            'will return images '
            'in order.')
        parser.add_argument('--freq_save_test_stats', type=int, default=500)
        parser.add_argument(
            '--rl_env_train_no_seed',
            dest='rl_env_train_no_seed',
            action='store_true',
            help='If true, order of training images does not depend on the seed (useful when'
            'using preemption to avoid having to keep track of current image index).')

        # Options for the simple baselines
        parser.add_argument('--greedymc_num_samples', type=int, default=10)
        parser.add_argument('--greedymc_horizon', type=int, default=1)
        parser.add_argument(
            '--evaluator_dir', type=str, default=None, help='Directory where evaluator is stored.')

        # Evaluator++ options
        parser.add_argument(
            '--evaluator_pp_path',
            type=str,
            default=None,
            help='Full path to the evaluator++ model to use.')

        # Reinforcement learning options
        parser.add_argument('--rl_model_type', choices=['cnn_plus_masks'], default='cnn_plus_masks')
        parser.add_argument('--dqn_resume', dest='dqn_resume', action='store_true')
        parser.add_argument(
            '--dqn_only_test',
            dest='dqn_only_test',
            action='store_true',
            help='If true, no training will be done. A policy will be loaded from disk and tested.')
        parser.add_argument(
            '--dqn_weights_dir',
            type=str,
            default=None,
            help='Where to load the DQN weights from if dqn_only_test is used.')
        parser.add_argument('--replay_buffer_size', type=int, default=100000)
        parser.add_argument('--epsilon_start', type=float, default=0.99)
        parser.add_argument('--epsilon_end', type=float, default=0.01)
        parser.add_argument('--epsilon_decay', type=float, default=10000)
        parser.add_argument('--dqn_learning_rate', type=float, default=6.25e-5)
        parser.add_argument('--num_train_steps', type=int, default=1000000)
        parser.add_argument('--rl_batch_size', type=int, default=16)
        parser.add_argument('--test_set', choices=['train', 'val', 'test'], default='test')
        parser.add_argument(
            '--rl_burn_in',
            type=int,
            default=200,
            help='Before this many steps nothing will be sampled from the replay buffer.')
        parser.add_argument('--dqn_test_episode_freq', type=int, default=20)
        parser.add_argument('--target_net_update_freq', type=int, default=500)
        parser.add_argument('--gamma', type=float, default=0.999)
        parser.add_argument(
            '--no_replacement_policy', dest='no_replacement_policy', action='store_true')

        self.isTrain = False
        return parser
