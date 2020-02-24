import logging
import os
import random
import sys

import numpy as np
import tensorboardX
import torch

import options.rl_options
import rl_env
import util.rl.dqn
import util.rl.evaluator_plus_plus
import util.rl.replay_buffer
import util.rl.simple_baselines
import util.rl.utils


def get_experiment_str(options_):
    if options_.policy == 'dqn':
        if options_.dqn_only_test:
            policy_str = 'eval'
        else:
            policy_str = f'{options_.obs_type}.tupd{options_.target_net_update_freq}.' \
                f'bs{options_.rl_batch_size}.' \
                f'edecay{options_.epsilon_decay}.gamma{options_.gamma}.' \
                f'lr{options_.dqn_learning_rate}.repbuf{options_.replay_buffer_size}' \
                f'norepl{int(options_.no_replacement_policy)}.nimgtr{options_.num_train_images}.' \
                f'metric{options_.reward_metric}.usescoasrew{int(options_.use_score_as_reward)}'
    else:
        policy_str = options_.policy
        if 'greedymc' in options_.policy:
            policy_str = '{}.nsam{}.hor{}_'.format(policy_str, options_.greedymc_num_samples,
                                                   options_.greedymc_horizon)
    return f'{policy_str}_bu{options_.budget}_seed{options_.seed}_neptest{options_.num_test_images}'


# Not a great policy specification protocol, but works for now.
def get_policy(env, writer, logger, options_):
    valid_actions = env.valid_actions

    # This options affects how the score is computed by the environment
    # (whether it passes through reconstruction network or not after a new col. is scanned)
    logger.info(f'Use reconstructions is {options_.use_reconstructions}')
    if 'random' in options_.policy:
        policy = util.rl.simple_baselines.RandomPolicy(valid_actions)
    elif 'lowfirst' in options_.policy:
        policy = util.rl.simple_baselines.NextIndexPolicy(valid_actions, not env.conjugate_symmetry)
    elif options_.policy == 'one_step_greedy':
        policy = util.rl.simple_baselines.OneStepGreedy(
            env, options_.reward_metric, max_actions_to_eval=options_.greedy_max_num_actions)
    elif 'evaluator_net' in options_.policy:
        assert options_.obs_type == 'image_space'
        # At the moment, Evaluator gets valid actions in a mask - preprocess data function.
        # So, no need to pass `valid_actions`
        policy = util.rl.simple_baselines.EvaluatorNetwork(env)
    elif 'evaluator++' in options_.policy:
        policy = util.rl.evaluator_plus_plus.EvaluatorPlusPlusPolicy(
            options_.options.evaluator_pp_path, options_.initial_num_lines_per_side, rl_env.device)
    elif 'dqn' in options_.policy:
        assert options_.obs_to_numpy
        dqn_trainer = util.rl.dqn.DQNTrainer(options_, env, writer, logger)
        dqn_trainer()
        policy = dqn_trainer.policy
    else:
        raise ValueError

    return policy


def main(options_, logger):
    writer = tensorboardX.SummaryWriter(options_.checkpoints_dir, flush_secs=60)
    env = rl_env.ReconstructionEnv(options_)
    options_.mask_embedding_dim = env.metadata['mask_embed_dim']
    options_.image_width = env.image_width
    env.set_training()
    policy = get_policy(env, writer, logger, options_)  # Trains if necessary
    logger.info(f'Created environment with {env.action_space.n} actions')
    env.set_testing()
    util.rl.utils.test_policy(env, policy, writer, logger, 0, options_)


if __name__ == '__main__':
    # Reading options
    opts = options.rl_options.RLOptions().parse()

    random.seed(opts.seed)
    np.random.seed(opts.seed)
    torch.manual_seed(opts.seed)

    experiment_str = get_experiment_str(opts)
    opts.checkpoints_dir = os.path.join(opts.checkpoints_dir, experiment_str)
    if not os.path.isdir(opts.checkpoints_dir):
        os.makedirs(opts.checkpoints_dir)

    # Initializing logger
    logger_ = logging.getLogger()
    logger_.setLevel(logging.DEBUG)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.DEBUG)
    fh = logging.FileHandler(os.path.join(opts.checkpoints_dir, 'train.log'))
    fh.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(threadName)s - %(levelname)s: %(message)s')
    ch.setFormatter(formatter)
    fh.setFormatter(formatter)
    logger_.addHandler(ch)
    logger_.addHandler(fh)

    logger_.info(f'Results will be saved at {opts.checkpoints_dir}.')

    logger_.info('Creating RL acquisition run with the following options:')
    for key, value in vars(opts).items():
        if key == 'device':
            value = value.type
        elif key == 'gpu_ids':
            value = 'cuda : ' + str(value) if torch.cuda.is_available() else 'cpu'
        logger_.info(f"    {key:>25}: {'None' if value is None else value:<30}")

    main(opts, logger_)
