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
            choices=['fourier_space', 'image_space', 'only_mask'],
            default='image_space')
        parser.add_argument('--obs_to_numpy', action='store_true')
        parser.add_argument(
            '--policy',
            choices=[
                'dqn', 'random', 'lowfirst', 'one_step_greedy', 'evaluator_net', 'evaluator++'
            ],
            default='random')
        parser.add_argument('--initial_num_lines_per_side', type=int, default=10)
        parser.add_argument('--budget', type=int, default=5)
        parser.add_argument('--num_test_images', type=int, default=200)
        parser.add_argument('--num_train_images', type=int, default=10000000)
        parser.add_argument(
            '--test_set_shift',
            type=int,
            default=None,
            help='If given, the indices in the test set are rotated. '
            'This is useful to parallelize test runs (each job starts at a different index).')
        parser.add_argument(
            '--no_use_reconstructions', dest='use_reconstructions', action='store_false')
        parser.add_argument(
            '--use_score_as_reward',
            dest='use_score_as_reward',
            action='store_true',
            help='If true, the reward is the score (e.g., MSE, SSIM). Otherwise, the reward is the '
            'decrease in score with respect to previous step.')
        parser.add_argument(
            '--reward_metric', choices=['mse', 'nmse', 'ssim', 'psnr'], default='mse')
        parser.add_argument('--reward_scaling', type=float, default=100)
        parser.add_argument('--debug', dest='debug', action='store_true')
        parser.add_argument('--job_name', type=str, default='active_acq')
        parser.add_argument('--seed', type=int, default=0)
        parser.add_argument('--freq_save_test_stats', type=int, default=500)
        parser.add_argument(
            '--rl_env_train_no_seed',
            dest='rl_env_train_no_seed',
            action='store_true',
            help='If true, order of training images does not depend on the seed (useful when'
            'using preemption to avoid having to keep track of current image index).')
        parser.add_argument(
            '--train_with_fixed_initial_mask',
            dest='train_with_fixed_initial_mask',
            action='store_true')
        parser.add_argument(
            '--mask_type',
            type=str,
            choices=[
                'basic',
                'symmetric_basic',
                'low_to_high',
                'grid',
                'symmetric_grid',
                'basic_rnl',
                'symmetric_basic_rnl',
                'low_to_high_rnl',
            ],
            help='The type of mask to use as initial state for episodes.')
        parser.add_argument(
            '--rnl_params',
            type=str,
            default=None,
            help='Format is min_lowf_lines,max_lowf_lines,highf_beta_alpha,highf_beta_beta')
        parser.add_argument(
            '--normalize_rewards_on_val', dest='normalize_rewards_on_val', action='store_true')

        # Options for the simple baselines
        parser.add_argument('--greedy_max_num_actions', type=int, default=None)
        parser.add_argument(
            '--evaluator_dir', type=str, default=None, help='Directory where evaluator is stored.')

        # Evaluator++ options
        parser.add_argument(
            '--evaluator_pp_path',
            type=str,
            default=None,
            help='Full path to the evaluator++ model to use.')
        parser.add_argument(
            '--add_mask_eval',
            action='store_true',
            help='Summation of mask to observation in evaluator')

        # Reinforcement learning options
        parser.add_argument(
            '--dqn_model_type', choices=['basic', 'evaluator', 'simple_mlp'], default='evaluator')
        parser.add_argument('--no_dqn_resume', dest='dqn_resume', action='store_false')
        parser.add_argument('--dqn_normalize', dest='dqn_normalize', action='store_true')
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
        parser.add_argument('--replay_buffer_size', type=int, default=1000000)
        parser.add_argument('--epsilon_start', type=float, default=0.99)
        parser.add_argument('--epsilon_end', type=float, default=0.001)
        parser.add_argument('--epsilon_decay', type=float, default=500000)
        parser.add_argument('--dqn_learning_rate', type=float, default=6.25e-5)
        parser.add_argument('--num_train_steps', type=int, default=5000000)
        parser.add_argument('--rl_batch_size', type=int, default=16)
        parser.add_argument('--test_set', choices=['train', 'val', 'test'], default='test')
        parser.add_argument(
            '--test_num_cols_cutoff',
            type=int,
            default=None,
            help='Specifies a cutoff point for test (and validation). '
            'Once an image reaches this number of scanned columns, '
            'the test episode will stop. Default (None) indicates full budget.')
        parser.add_argument(
            '--dqn_burn_in',
            type=int,
            default=200,
            help='Before this many steps nothing will be sampled from the replay buffer.')
        parser.add_argument(
            '--dqn_test_episode_freq',
            type=int,
            default=None,
            help='Specifies the frequency (in terms of training episodes) for '
            'running DQN evaluation. `None` indicates that training will '
            'run uninterrupted by evaluation episodes.')
        parser.add_argument('--dqn_eval_train_set_episode_freq', type=int, default=None)
        parser.add_argument('--target_net_update_freq', type=int, default=5000)
        parser.add_argument('--gamma', type=float, default=0.5)
        parser.add_argument(
            '--allow_replace_actions', dest='no_replacement_policy', action='store_false')
        parser.add_argument('--freq_dqn_checkpoint_save', type=int, default=200)
        parser.add_argument('--use_dueling_dqn', dest='use_dueling_dqn', action='store_true')

        # Alternate optimization options
        parser.add_argument(
            '--dqn_alternate_opt_per_epoch',
            dest='dqn_alternate_opt_per_epoch',
            action='store_true')
        parser.add_argument('--num_epochs_train_reconstructor', type=int, default=10)
        parser.add_argument('--frequency_train_reconstructor', type=int, default=5000)
        parser.add_argument('--reconstructor_lr', type=float, default=0.00002)
        parser.add_argument('--alternate_opt_batch_size', type=int, default=40)
        parser.add_argument(
            '--num_steps_with_fixed_dqn_params',
            type=int,
            default=0,
            help='If the number of steps is less than this number, then DQN parameters are not '
            'updated. The exploration epsilon is also held constant during this time')

        parser.add_argument('--partition', type=str, default='learnfair')

        self.isTrain = False
        return parser

    def parse(self):
        options = super(RLOptions, self).parse()
        options.batchSize = 1
        options.masks_dir = None  # Ignored, only here for compatibility with loader
        return options
