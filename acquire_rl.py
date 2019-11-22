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
    assert len(value) == len(statistics)
    for k, v in statistics.items():
        if episode_step not in statistics[k]:
            statistics[k][episode_step] = {'mean': 0, 'm2': 0, 'count': 0}
        statistics[k][episode_step]['count'] += 1
        delta = value[k] - statistics[k][episode_step]['mean']
        statistics[k][episode_step]['mean'] += delta / statistics[k][episode_step]['count']
        delta2 = value[k] - statistics[k][episode_step]['mean']
        statistics[k][episode_step]['m2'] += delta * delta2


def compute_test_score_from_stats(statistics):
    """ Computes a single-value score from a set of time step statistics. """
    score = 0
    for episode_step, step_stats in statistics.items():
        score += step_stats['mean']
    return score


def save_statistics_and_actions(statistics, all_actions, episode, logger, options_):
    logger.info(f'Episode {episode}. Saving statistics to {options_.checkpoints_dir}.')
    np.save(
        os.path.join(options_.checkpoints_dir, 'test_stats_mse_{}'.format(episode)),
        statistics['mse'])
    np.save(
        os.path.join(options_.checkpoints_dir, 'test_stats_ssim_{}'.format(episode)),
        statistics['ssim'])
    np.save(
        os.path.join(options_.checkpoints_dir, 'test_stats_psnr_{}'.format(episode)),
        statistics['psnr'])
    np.save(os.path.join(options_.checkpoints_dir, 'all_actions'), np.array(all_actions))


def test_policy(env,
                policy,
                writer,
                logger,
                step,
                options_,
                test_on_train=False,
                test_with_full_budget=False,
                leave_no_trace=False):
    """ Evaluates a given policy for the environment on the test set. """
    env.set_testing(use_training_set=test_on_train)
    old_budget = env.options.budget
    if test_with_full_budget:
        env.options.budget = env.action_space.n
    episode = 0
    statistics = {'mse': {}, 'ssim': {}, 'psnr': {}, 'rewards': {}}
    import time
    start = time.time()
    all_actions = []
    while True:
        obs, _ = env.reset(start_with_initial_mask=True)
        policy.init_episode()
        if obs is None:
            if not leave_no_trace:
                save_statistics_and_actions(statistics, all_actions, episode, logger, options_)
            break
        episode += 1
        done = False
        total_reward = 0
        actions = []
        episode_step = 0
        reconstruction_results = env.compute_score(
            options_.use_reconstructions, use_current_score=True)[0]
        reconstruction_results['rewards'] = 0
        update_statistics(reconstruction_results, episode_step, statistics)
        while not done:
            action = policy.get_action(obs, 0., actions)
            actions.append(action)
            next_obs, reward, done, _ = env.step(action)
            total_reward += reward
            obs = next_obs
            episode_step += 1
            reconstruction_results = env.compute_score(use_current_score=True)[0]
            reconstruction_results['rewards'] = reward
            update_statistics(reconstruction_results, episode_step, statistics)
        all_actions.append(actions)
        if not leave_no_trace:
            logger.debug('Actions and reward: {}, {}'.format(actions, total_reward))
        if not test_on_train and not leave_no_trace \
                and (episode % options_.freq_save_test_stats == 0):
            save_statistics_and_actions(statistics, all_actions, episode, logger, options_)
    end = time.time()
    if not leave_no_trace:
        logger.debug('Test run lasted {} seconds.'.format(end - start))
    test_score = compute_test_score_from_stats(statistics[options_.reward_metric])
    split = 'train' if test_on_train else 'test'
    if not leave_no_trace:
        writer.add_scalar(f'eval/{split}_score__{options_.reward_metric}_auc', test_score, step)
    env.set_training()
    if test_with_full_budget:
        env.options.budget = old_budget

    if options_.reward_metric == 'mse':  # DQN maximizes but we want to minimize MSE
        test_score = -test_score

    return test_score, statistics


def get_experiment_str(options_):
    if options_.policy == 'dqn':
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
    # This options affects how the score is computed by the environment
    # (whether it passes through reconstruction network or not after a new col. is scanned)
    logger.info(f'Use reconstructions is {options_.use_reconstructions}')
    if 'random' in options_.policy:
        policy = util.rl.simple_baselines.RandomPolicy(range(env.action_space.n))
    elif 'lowfirst' in options_.policy:
        policy = util.rl.simple_baselines.NextIndexPolicy(
            range(env.action_space.n), not env.conjugate_symmetry)
    elif 'evaluator_net' in options_.policy:
        assert options_.obs_type == 'image_space'
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
    if options_.normalize_rewards_on_val:
        logger.info('Running random policy to get reference point for reward.')
        random_policy = util.rl.simple_baselines.RandomPolicy(range(env.action_space.n))
        env.set_testing()
        _, statistics = test_policy(
            env, random_policy, None, None, 0, options_, leave_no_trace=True)
        logger.info('Done computing reference.')
        env.set_reference_point_for_rewards(statistics)
    options_.mask_embedding_dim = env.metadata['mask_embed_dim']
    options_.image_width = env.image_width
    env.set_training()
    logger.info(f'Created environment with {env.action_space.n} actions')
    policy = get_policy(env, writer, logger, options_)  # Trains if necessary
    env.set_testing()
    test_policy(env, policy, writer, logger, 0, options_)


if __name__ == '__main__':
    # Reading options
    opts = options.rl_options.RLOptions().parse()
    opts.batchSize = 1
    opts.masks_dir = None  # Ignored, only here for compatibility with loader

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
