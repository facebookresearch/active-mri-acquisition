import logging
import os
import types

import numpy as np
import sklearn.metrics
import tensorboardX
import torch

from . import Policy

from typing import Dict, Tuple


def _update_statistics(value, episode_step, statistics):
    # Updates a running mean and standard deviation for `episode_step`, given `value`.
    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford's_online_algorithm

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


def _save_statistics_and_actions(statistics, all_actions, episode, logger, options_):
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
    env: "rl_env.ReconstructionEnv",
    policy: Policy,
    writer: tensorboardX.SummaryWriter,
    logger: logging.Logger,
    step: int,
    options_: types.SimpleNamespace,
    test_on_train: bool = False,
    silent: bool = False,
) -> Tuple[float, Dict]:
    """ Evaluates a given policy for the given environment on its test set.

        Args:
            env(rl_env.ReconstructionEnv): The environment where the policy will be run on. The
                    evaluation will be done over the test set, as configured when creating the env.
            policy(Policy): The policy to evaluate.
            writer(tensorboardX.SummaryWriter): A tensorboard writer for results.
            logger(logging.Logger): The logger to which to write outputs.
            options_(types.SimpleNamespace): See :class:rl_env.ReconstructionEnv for fields.
            test_on_train(bool): If ``True`` the policy will be evaluated on the environments'
                    train set. Useful for debugging. Defaults to ``False``.
            silent(bool): If ``True`` nothing will be sent to writer nor to logger, and no
                    statistics will be saved. Defaults to ``False``.

        Returns:
            Tuple(float,Dict): The first element is the score obtained by the policy. The second
            one is a dictionary storing statistics for all reward metrics, as well as the
            actions chosen at each episode and time step (key "all_actions").\n

            The dictionary will contain a key for each metric (mse, nmse, ssim, psnr). Each
            of these dictionaries will contain the following entries:
                \t"all" - storing all observed metric values per episode per time.\n
                \t"count" - the count of values over which mean and m2 where computed.
                \t"mean" - the mean at each time step across all episodes.\n
                \t"m2" - the sum of squared deviations at each time step across all episodes.
                        The estimated variance can be then obtained as m2 / (count - 1).

    """
    env.set_testing(use_training_set=test_on_train)
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
            if not silent:
                _save_statistics_and_actions(
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
            options_.use_reconstructions,
            use_current_score=True,
            keep_prev_reconstruction=options_.keep_prev_reconstruction,
        )[0]
        episode_accelerations.append(
            env.convert_num_cols_to_acceleration(
                env.get_num_active_columns_in_obs(obs), options_.dataroot
            )
        )
        episode_scores.append(reconstruction_results[options_.reward_metric])
        reconstruction_results["rewards"] = 0
        _update_statistics(reconstruction_results, episode_step, statistics)
        while not done:
            action = policy.get_action(obs, 0.0, episode_actions)
            episode_actions.append(action)
            next_obs, reward, done, _ = env.step(action)
            total_reward_episode += reward
            obs = next_obs
            episode_step += 1
            reconstruction_results = env.compute_score(
                use_current_score=True,
                keep_prev_reconstruction=options_.keep_prev_reconstruction,
            )[0]
            reconstruction_results["rewards"] = reward
            episode_accelerations.append(
                env.convert_num_cols_to_acceleration(
                    env.get_num_active_columns_in_obs(obs), options_.dataroot
                )
            )
            episode_scores.append(reconstruction_results[options_.reward_metric])
            _update_statistics(reconstruction_results, episode_step, statistics)
        all_actions.append(episode_actions)
        all_auc.append(sklearn.metrics.auc(episode_accelerations, episode_scores))
        if not silent:
            logger.debug(
                "Actions and reward: {}, {}".format(
                    episode_actions, total_reward_episode
                )
            )
        if (
            not test_on_train
            and not silent
            and (episode % options_.freq_save_test_stats == 0)
        ):
            _save_statistics_and_actions(
                statistics, all_actions, episode, logger, options_
            )
    end = time.time()
    if not silent:
        logger.debug("Test run lasted {} seconds.".format(end - start))
    test_score = np.mean(all_auc)
    split = "train" if test_on_train else "test"
    if not silent:
        writer.add_scalar(
            f"eval/{split}_score__{options_.reward_metric}_auc", test_score, step
        )

    # DQN maximizes but we want to minimize MSE
    if options_.reward_metric == "mse" or options_.reward_metric == "nmse":
        test_score = -test_score

    logger.info(f"Completed test iterations.")

    env.set_training()
    return test_score, statistics
