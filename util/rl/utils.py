import os

import numpy as np
import torch


def update_statistics(value, episode_step, statistics):
    """ Updates a running mean and standard deviation for `episode_step`, given `value`.
        https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford's_online_algorithm
    """
    assert len(value) == len(statistics)
    for k, v in statistics.items():
        if episode_step not in statistics[k]:
            statistics[k][episode_step] = {'mean': 0, 'm2': 0, 'count': 0, 'all': []}
        value_k = value[k].item() if isinstance(value[k], torch.Tensor) else value[k]
        statistics[k][episode_step]['all'].append(value_k)
        statistics[k][episode_step]['count'] += 1
        delta = value_k - statistics[k][episode_step]['mean']
        statistics[k][episode_step]['mean'] += delta / statistics[k][episode_step]['count']
        delta2 = value_k - statistics[k][episode_step]['mean']
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
                leave_no_trace=False):
    """ Evaluates a given policy for the environment on the test set. """
    env.set_testing(use_training_set=test_on_train)
    old_budget = env.options.budget
    env.options.budget = env.action_space.n if options_.test_budget is None \
        else options_.test_budget
    logger.info(f'Starting test iterations. Test budget set to {env.options.budget}.')
    episode = 0
    statistics = {'mse': {}, 'ssim': {}, 'psnr': {}, 'rewards': {}}
    import time
    start = time.time()
    all_actions = []
    while True:  # Will loop over the complete test set (indicated when env.reset() returns None)
        obs, _ = env.reset(start_with_initial_mask=True)
        policy.init_episode()
        if obs is None:
            if not leave_no_trace:
                save_statistics_and_actions(statistics, all_actions, episode, logger, options_)
            break
        episode += 1
        done = False
        total_reward_episode = 0
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
            total_reward_episode += reward
            obs = next_obs
            episode_step += 1
            reconstruction_results = env.compute_score(
                use_current_score=True, use_zz_score=options_.eval_with_zz_score)[0]
            reconstruction_results['rewards'] = reward
            update_statistics(reconstruction_results, episode_step, statistics)
        all_actions.append(actions)
        if not leave_no_trace:
            logger.debug('Actions and reward: {}, {}'.format(actions, total_reward_episode))
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
    env.options.budget = old_budget

    if options_.reward_metric == 'mse':  # DQN maximizes but we want to minimize MSE
        test_score = -test_score

    logger.info(f'Completed test iterations. Test budget set back to {env.options.budget}.')
    return test_score, statistics
