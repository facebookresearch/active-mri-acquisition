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
            choices=['two_streams', 'concatenate_mask', 'spectral_maps'],
            default='two_streams')
        parser.add_argument(
            '--policy',
            choices=[
                'dqn', 'random', 'lowfirst', 'evaluator_net', 'evaluator_net_offp', 'greedymc',
                'greedymc_gt', 'greedyfull1_gt', 'greedyfull1', 'greedyzero', 'evaluator++'
            ],
            default='random')
        parser.add_argument('--initial_num_lines', type=int, default=10)
        parser.add_argument('--budget', type=int, default=5)
        parser.add_argument('--num_test_images', type=int, default=200)
        parser.add_argument('--num_train_images', type=int, default=10000000)
        parser.add_argument(
            '--use_reconstructions', dest='use_reconstructions', action='store_true')
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
        parser.add_argument(
            '--rl_model_type',
            choices=['two_streams', 'spectral_maps', 'large_two_streams'],
            default='two_streams')
        parser.add_argument('--dqn_resume', dest='dqn_resume', action='store_true')
        parser.add_argument(
            '--dqn_only_test',
            dest='dqn_only_test',
            action='store_true',
            help='If true, no training will be done. A policy will be loaded from disk and tested.')
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

        # ########################
        # PPO options
        # ########################
        parser.add_argument('--log_dir', type=str, default=None)
        parser.add_argument(
            '--clip-param',
            type=float,
            default=0.2,
            help='ppo clip parameter (default: 0.2)')
        parser.add_argument(
            '--ppo-epoch',
            type=int,
            default=4,
            help='number of ppo epochs (default: 4)')
        parser.add_argument(
            '--num-mini-batch',
            type=int,
            default=32,
            help='number of batches for ppo (default: 32)')
        parser.add_argument(
            '--value-loss-coef',
            type=float,
            default=0.5,
            help='value loss coefficient (default: 0.5)')
        parser.add_argument(
            '--entropy-coef',
            type=float,
            default=0.01,
            help='entropy term coefficient (default: 0.01)')
        parser.add_argument(
            '--eps',
            type=float,
            default=1e-5,
            help='RMSprop optimizer epsilon (default: 1e-5)')
        parser.add_argument(
            '--use_clipped_value_loss',
            type=bool,
            default=True)
        parser.add_argument(
            '--num_processes',
            type=int,
            default=1)
        parser.add_argument(
            '--num_steps',
            type=int,
            default=2048)
        parser.add_argument(
            '--lr_ppo_actor_critic', type=float, default=0.0002, help='initial learning rate for adam')
        parser.add_argument(
            '--use_linear_lr_decay',
            type=bool,
            default=True)
        parser.add_argument(
            '--num_env_steps',
            type=int,
            default=1000000)
        parser.add_argument(
            '--use_gae',
            action='store_true',
            default=False,
            help='use generalized advantage estimation')
        parser.add_argument(
            '--gae_lambda',
            type=float,
            default=0.95,
            help='gae lambda parameter (default: 0.95)')
        parser.add_argument(
            '--use_proper_time_limits',
            action='store_true',
            default=False,
            help='compute returns taking into account time limits')
        parser.add_argument(
            '--max_grad_norm',
            type=float,
            default=0.5,
            help='max norm of gradients (default: 0.5)')
        parser.add_argument(
            '--save_interval',
            type=int,
            default=100,
            help='save interval, one save per n updates (default: 100)')
        parser.add_argument(
            '--log_interval',
            type=int,
            default=10,
            help='log interval, one log per n updates (default: 10)')

        self.isTrain = False
        return parser
