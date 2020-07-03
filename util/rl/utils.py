import os

import numpy as np
import sklearn.metrics
import torch


def update_statistics(value, episode_step, statistics):
    """ Updates a running mean and standard deviation for `episode_step`, given `value`.
        https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford's_online_algorithm
    """
    assert len(value) == len(statistics)
    for k, v in statistics.items():
        if episode_step not in statistics[k]:
            statistics[k][episode_step] = {"mean": 0, "m2": 0, "count": 0, "all": []}
        value_k = value[k].item() if isinstance(value[k], torch.Tensor) else value[k]
        statistics[k][episode_step]["all"].append(value_k)
        statistics[k][episode_step]["count"] += 1
        delta = value_k - statistics[k][episode_step]["mean"]
        statistics[k][episode_step]["mean"] += (
            delta / statistics[k][episode_step]["count"]
        )
        delta2 = value_k - statistics[k][episode_step]["mean"]
        statistics[k][episode_step]["m2"] += delta * delta2


def compute_test_score_from_stats(statistics):
    """ Computes a single-value score from a set of time step statistics. """
    score = 0
    for episode_step, step_stats in statistics.items():
        score += step_stats["mean"]
    return score


def save_statistics_and_actions(statistics, all_actions, episode, logger, options_):
    logger.info(f"Episode {episode}. Saving statistics to {options_.checkpoints_dir}.")
    np.save(
        os.path.join(options_.checkpoints_dir, "test_stats_mse_{}".format(episode)),
        statistics["mse"],
    )
    np.save(
        os.path.join(options_.checkpoints_dir, "test_stats_nmse_{}".format(episode)),
        statistics["nmse"],
    )
    np.save(
        os.path.join(options_.checkpoints_dir, "test_stats_ssim_{}".format(episode)),
        statistics["ssim"],
    )
    np.save(
        os.path.join(options_.checkpoints_dir, "test_stats_psnr_{}".format(episode)),
        statistics["psnr"],
    )
    np.save(
        os.path.join(options_.checkpoints_dir, "all_actions"), np.array(all_actions)
    )


def test_policy(
    env,
    policy,
    writer,
    logger,
    step,
    options_,
    test_on_train=False,
    leave_no_trace=False,
):
    """ Evaluates a given policy for the environment on the test set. """
    env.set_testing(use_training_set=test_on_train)
    cols_cutoff = env.options.test_num_cols_cutoff
    if cols_cutoff is None:
        cols_cutoff = env.action_space.n
    logger.info(
        f"Starting test iterations. Max. num lines for test set " f"to {cols_cutoff}."
    )
    episode = 0
    statistics = {"mse": {}, "nmse": {}, "ssim": {}, "psnr": {}, "rewards": {}}
    import time

    start = time.time()
    all_actions = []
    all_auc = []
    while (
        True
    ):  # Will loop over the complete test set (indicated when env.reset() returns None)
        obs, _ = env.reset(start_with_initial_mask=True)
        policy.init_episode()
        if obs is None:
            if not leave_no_trace:
                save_statistics_and_actions(
                    statistics, all_actions, episode, logger, options_
                )
            break
        episode += 1
        done = False
        total_reward_episode = 0
        episode_actions = []
        episode_accelerations = []
        episode_scores = []
        episode_step = 0
        reconstruction_results = env.compute_score(
            options_.use_reconstructions, use_current_score=True
        )[0]
        episode_accelerations.append(
            env.convert_num_cols_to_acceleration(
                env.get_num_active_columns_in_obs(obs), options_.dataroot
            )
        )
        episode_scores.append(reconstruction_results[options_.reward_metric])
        reconstruction_results["rewards"] = 0
        update_statistics(reconstruction_results, episode_step, statistics)
        while not done:
            action = policy.get_action(obs, 0.0, episode_actions)
            episode_actions.append(action)
            next_obs, reward, done, _ = env.step(action)
            total_reward_episode += reward
            obs = next_obs
            episode_step += 1
            reconstruction_results = env.compute_score(use_current_score=True)[0]
            reconstruction_results["rewards"] = reward
            episode_accelerations.append(
                env.convert_num_cols_to_acceleration(
                    env.get_num_active_columns_in_obs(obs), options_.dataroot
                )
            )
            episode_scores.append(reconstruction_results[options_.reward_metric])
            update_statistics(reconstruction_results, episode_step, statistics)
        all_actions.append(episode_actions)
        all_auc.append(sklearn.metrics.auc(episode_accelerations, episode_scores))
        if not leave_no_trace:
            logger.debug(
                "Actions and reward: {}, {}".format(
                    episode_actions, total_reward_episode
                )
            )
        if (
            not test_on_train
            and not leave_no_trace
            and (episode % options_.freq_save_test_stats == 0)
        ):
            save_statistics_and_actions(
                statistics, all_actions, episode, logger, options_
            )
    end = time.time()
    if not leave_no_trace:
        logger.debug("Test run lasted {} seconds.".format(end - start))
    test_score = np.mean(all_auc)
    split = "train" if test_on_train else "test"
    if not leave_no_trace:
        writer.add_scalar(
            f"eval/{split}_score__{options_.reward_metric}_auc", test_score, step
        )

    # DQN maximizes but we want to minimize MSE
    if options_.reward_metric == "mse" or options_.reward_metric == "nmse":
        test_score = -test_score

    logger.info(
        f"Completed test iterations. Test budget set back to {env.options.budget}."
    )

    env.set_training()
    return test_score, statistics
