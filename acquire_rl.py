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


def update_statistics(value, episode_step, statistics):
    """ Updates a running mean and standard deviation for `episode_step`, given `value`.
        https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford's_online_algorithm
    """
    if episode_step not in statistics:
        statistics[episode_step] = {'mean': 0, 'm2': 0, 'count': 0}
    statistics[episode_step]['count'] += 1
    delta = value - statistics[episode_step]['mean']
    statistics[episode_step]['mean'] += delta / statistics[episode_step]['count']
    delta2 = value - statistics[episode_step]['mean']
    statistics[episode_step]['m2'] += delta * delta2


def compute_test_score_from_stats(statistics):
    """ Computes a single-value score from a set of time step statistics. """
    score = 0
    for episode_step, step_stats in statistics.items():
        score += step_stats['mean']
    return score


def test_policy(env, policy, writer, logger, num_episodes, step, options_):
    """ Evaluates a given policy for the environment on the test set. """
    env.set_testing()
    average_total_reward = 0
    episode = 0
    statistics_mse = {}
    statistics_ssim = {}
    statistics_psnr = {}
    import time
    start = time.time()
    all_actions = []
    while True:
        obs, _ = env.reset()
        policy.init_episode()
        if episode == num_episodes or obs is None:
            break
        episode += 1
        done = False
        total_reward = 0
        actions = []
        episode_step = 0
        # TODO make these 3 be computed on a single call
        update_statistics(
            env.compute_score(options_.use_reconstructions, kind='mse')[0], episode_step,
            statistics_mse)
        update_statistics(
            env.compute_score(options_.use_reconstructions, kind='ssim')[0], episode_step,
            statistics_ssim)
        update_statistics(
            env.compute_score(options_.use_reconstructions, kind='psnr')[0], episode_step,
            statistics_psnr)
        while not done:
            action = policy.get_action(obs, 0., actions)
            actions.append(action)
            next_obs, reward, done, _ = env.step(action)
            total_reward += reward
            obs = next_obs
            episode_step += 1
            update_statistics(
                env.compute_score(options_.use_reconstructions, kind='mse')[0], episode_step,
                statistics_mse)
            update_statistics(
                env.compute_score(options_.use_reconstructions, kind='ssim')[0], episode_step,
                statistics_ssim)
            update_statistics(
                env.compute_score(options_.use_reconstructions, kind='psnr')[0], episode_step,
                statistics_psnr)
        average_total_reward += total_reward
        all_actions.append(actions)
        logger.debug('Actions and reward: {}, {}'.format(actions, total_reward))
        if episode % options_.freq_save_test_stats == 0 or episode == num_episodes:
            logger.info(f'Episode {episode}. Saving statistics to {options_.checkpoints_dir}.')
            np.save(
                os.path.join(options_.checkpoints_dir, 'test_stats_mse_{}'.format(episode)),
                statistics_mse)
            np.save(
                os.path.join(options_.checkpoints_dir, 'test_stats_ssim_{}'.format(episode)),
                statistics_ssim)
            np.save(
                os.path.join(options_.checkpoints_dir, 'test_stats_psnr_{}'.format(episode)),
                statistics_ssim)
            np.save(os.path.join(options_.checkpoints_dir, 'all_actions'), np.array(all_actions))
    end = time.time()
    logger.debug('Test run lasted {} seconds.'.format(end - start))
    writer.add_scalar('eval/average_reward', average_total_reward / episode, step)
    env.set_training()

    return compute_test_score_from_stats(statistics_mse)


def get_experiment_str(options_):
    if options_.policy == 'dqn':
        policy_str = f'{options_.obs_type}.tupd{options_.target_net_update_freq}.' \
            f'bs{options_.rl_batch_size}.' \
            f'edecay{options_.epsilon_decay}.gamma{options_.gamma}.' \
            f'norepl{int(options_.no_replacement_policy)}.nimgtr{options_.num_train_images}.'
    else:
        policy_str = options_.policy
        if 'greedymc' in options_.policy:
            policy_str = '{}.nsam{}.hor{}_'.format(policy_str, options_.greedymc_num_samples,
                                                   options_.greedymc_horizon)
    return f'{policy_str}_bu{options_.budget}_seed{options_.seed}_neptest{options_.num_test_images}'


# Not a great policy specification protocol, but works for now.
def get_policy(env, writer, logger, options_):
    # This options affects how the score is computed by the environment
    # (whether it passes through reconstruction network or not after a new col. is scanned)
    logger.info(f'Use reconstructions is {options_.use_reconstructions}')
    if 'random' in options_.policy:
        policy = util.rl.simple_baselines.RandomPolicy(range(env.action_space.n))
    elif 'lowfirst' in options_.policy:
        assert rl_env.CONJUGATE_SYMMETRIC
        policy = util.rl.simple_baselines.NextIndexPolicy(range(env.action_space.n))
    elif 'greedymc' in options_.policy:
        policy = util.rl.simple_baselines.GreedyMC(
            env,
            samples=options_.greedymc_num_samples,
            horizon=options_.greedymc_horizon,
            use_reconstructions=options_.use_reconstructions,
            use_ground_truth='_gt' in options_.policy)
    elif 'greedyfull1' in options_.policy:
        assert options_.use_reconstructions
        policy = util.rl.simple_baselines.FullGreedy(
            env, num_steps=1, use_ground_truth='_gt' in options_.policy, use_reconstructions=True)
    elif 'greedyzero' in options_.policy:
        assert options_.use_reconstructions
        policy = util.rl.simple_baselines.ZeroStepGreedy(env)
    elif 'evaluator_net' in options_.policy:
        policy = util.rl.simple_baselines.EvaluatorNetwork(env)
    elif 'evaluator_net_offp' in options_.policy:
        assert options_.evaluator_name is not None and options_.evaluator_name != options_.name
        policy = util.rl.simple_baselines.EvaluatorNetwork(env)
    elif 'evaluator++' in options_.policy:
        assert options_.obs_type == 'concatenate_mask'
        policy = util.rl.evaluator_plus_plus.EvaluatorPlusPlusPolicy(
            options_.options.evaluator_pp_path, options_.initial_num_lines, rl_env.device)
    elif 'dqn' in options_.policy:
        dqn_trainer = util.rl.dqn.DQNTrainer(options_, env, writer, logger)
        dqn_trainer()
        policy = dqn_trainer.policy
    else:
        raise ValueError

    return policy


def main(options_, logger):
    writer = tensorboardX.SummaryWriter(options_.checkpoints_dir)
    env = rl_env.ReconstructionEnv(
        rl_env.generate_initial_mask(options_.initial_num_lines), options_)
    env.set_training()
    logger.info(f'Created environment with {env.action_space.n} actions')
    policy = get_policy(env, writer, logger, options_)  # Trains if necessary
    env.set_testing()
    test_policy(env, policy, writer, logger, None, 0, options_)


if __name__ == '__main__':
    # Reading options
    opts = options.rl_options.RLOptions().parse()
    opts.batchSize = 1
    opts.mask_type = 'grid'  # This is ignored, only here for compatibility with loader

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

    main(opts, logger_)
