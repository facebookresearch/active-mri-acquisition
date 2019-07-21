from .base_options import BaseOptions


class RLOptions(BaseOptions):

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        parser.add_argument('--results_dir', type=str, default=None, help='saves results here.')
        parser.add_argument(
            '--rl_logs_subdir',
            type=str,
            default='debug',
            help='sub-directory of results_dir to store results of RL runs.')

        # General options for all active acquisition algorithms
        parser.add_argument(
            '--policy',
            choices=[
                'dqn', 'random', 'random_r', 'lowfirst', 'lowfirst_r', 'evaluator_net',
                'evaluator_net_r', 'evaluator_net_offp', 'evaluator_net_offp_r', 'greedymc',
                'greedymc_gt', 'greedymc_r', 'greedymc_gt_r', 'greedyfull1_gt_r', 'greedyfull1_r',
                'greedyfull1nors_gt', 'greedyfull1nors_gt_r', 'evaluator++_r'
            ],
            default='random')
        parser.add_argument('--initial_num_lines', type=int, default=10)
        parser.add_argument('--budget', type=int, default=5)
        parser.add_argument('--num_test_images', type=int, default=1000)
        parser.add_argument('--num_train_images', type=int, default=10000)
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

        # Options for the simple baselines
        parser.add_argument('--greedymc_num_samples', type=int, default=10)
        parser.add_argument('--greedymc_horizon', type=int, default=1)

        # Evaluator++ options
        parser.add_argument('--evaluator_pp_path', type=str, default=None)

        # Reinforcement learning options
        parser.add_argument(
            '--rl_obs_type',
            choices=['two_streams', 'concatenate_mask', 'spectral_maps'],
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
        parser.add_argument('--num_train_episodes', type=int, default=10000)
        parser.add_argument('--rl_batch_size', type=int, default=16)
        parser.add_argument(
            '--rl_burn_in',
            type=int,
            default=200,
            help='Before this many steps nothing will be sampled from the replay buffer.')
        parser.add_argument('--agent_test_episode_freq', type=int, default=20)
        parser.add_argument('--target_net_update_freq', type=int, default=500)
        parser.add_argument('--gamma', type=float, default=0.999)
        parser.add_argument(
            '--no_replacement_policy', dest='no_replacement_policy', action='store_true')

        self.isTrain = False
        return parser
