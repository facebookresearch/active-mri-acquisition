import argparse
import logging
import os
import pickle
import sys
from typing import List

import numpy as np
import torch

import rl_env
import util.rl.dqn
import util.rl.evaluator_plus_plus
import util.rl.replay_buffer
import util.rl.simple_baselines


def get_num_lines(min_lowf_lines: int = 3,
                  max_lowf_lines: int = 8,
                  highf_beta_alpha: int = 1,
                  highf_beta_beta: int = 5,
                  num_cols: int = 64):
    num_low_freqs = rng.choice(range(min_lowf_lines, max_lowf_lines))
    num_high_freqs = int(rng.beta(highf_beta_alpha, highf_beta_beta) * (num_cols - num_low_freqs))
    return num_low_freqs, num_high_freqs


# noinspection PyProtectedMember
def generate_masks(env: rl_env.ReconstructionEnv,
                   evaluator_policy: util.rl.simple_baselines.EvaluatorNetwork,
                   how_many_images: int) -> List[np.ndarray]:
    import time
    start = time.time()
    masks = []
    for episode in range(how_many_images):
        obs, info = env.reset()
        evaluator_policy.init_episode()
        episode += 1
        initial_num_lines_per_side, additional_lines = get_num_lines()
        logging.info(
            f'Will generate mask with {2 * initial_num_lines_per_side} low freq. lines and '
            f'{2 * additional_lines} high freq. lines.')
        mask = np.zeros(rl_env.IMAGE_WIDTH)
        mask[:initial_num_lines_per_side] = mask[-initial_num_lines_per_side:] = 1
        env._current_mask = torch.from_numpy(mask).float().to(rl_env.device).view(
            1, 1, 1, rl_env.IMAGE_WIDTH)
        env.options.initial_num_lines_per_side = initial_num_lines_per_side
        for i in range(additional_lines):
            action = evaluator_policy.get_action(obs, None, None)
            assert mask[initial_num_lines_per_side + action] == 0
            assert mask[rl_env.IMAGE_WIDTH - initial_num_lines_per_side - action - 1] == 0
            mask[initial_num_lines_per_side + action] = 1
            mask[rl_env.IMAGE_WIDTH - initial_num_lines_per_side - action - 1] = 1
            next_obs, _, done, _ = env.step(action)
            obs = next_obs
            if done:
                break
        if (episode + 1) % 500 == 0:
            logging.info(f"Processed image {info['image_idx']}")
        if info['image_idx'] == len(env._dataset_test) - 1:
            break  # Don't go over the last image, because env loops back to index 0
        assert np.sum(mask) == (initial_num_lines_per_side + additional_lines) * 2
        masks.append(mask)
    end = time.time()
    logging.debug(f'Generated {len(masks)} masks. Total time was {end - start} seconds.')
    return masks


# noinspection PyProtectedMember
def main(options: argparse.Namespace):
    env = rl_env.ReconstructionEnv(options)
    env._test_order = range(len(env._dataset_test))
    env.set_testing()
    env._image_idx_test = options.initial_index
    logging.info('Created environment with {} actions'.format(env.action_space.n))
    policy = util.rl.simple_baselines.EvaluatorNetwork(env)
    masks = generate_masks(env, policy, options.how_many_images)
    masks = np.stack(masks)
    filename = os.path.join(options.dataset_dir,
                            f'masks_{options.initial_index}-{options.initial_index + len(masks)}.p')
    with open(filename, 'wb') as f:
        pickle.dump(masks, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--reconstructor_dir', type=str, required=True)
    parser.add_argument('--evaluator_dir', type=str, required=True)
    parser.add_argument('--dataset_dir', type=str, required=True)
    parser.add_argument('--initial_num_lines_per_side', type=int, default=5)
    parser.add_argument('--dataroot', type=str, default='KNEE')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--initial_index', type=int, default=0)
    parser.add_argument('--how_many_images', type=int, default=15000)
    parser.add_argument('--test_set', choices=['train', 'test', 'valid'], default='train')
    options_ = parser.parse_args()

    options_.dataset_dir = os.path.join(options_.dataset_dir, options_.test_set)
    if not os.path.exists(options_.dataset_dir):
        os.makedirs(options_.dataset_dir)

    # Adding config options expected by `rl_env.ReconstructionEnv`
    options_.budget = 368 if options_.dataroot == 'KNEE_RAW' else 128
    options_.obs_type = 'image_space'
    options_.num_train_images = 2000000
    options_.num_test_images = 2000000
    options_.batchSize = 1
    options_.nThreads = 1
    # All of the params below are ignored
    # only here because they are expected, but not used by this script
    options_.mask_type = 'basic'
    options_.rl_env_train_no_seed = False
    options_.use_score_as_reward = False

    # Initializing logger
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.DEBUG)
    fh = logging.FileHandler(
        os.path.join(options_.dataset_dir, f'generation_{options_.initial_index}.log'))
    fh.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(threadName)s - %(levelname)s: %(message)s')
    ch.setFormatter(formatter)
    fh.setFormatter(formatter)
    logger.addHandler(ch)
    logger.addHandler(fh)

    rng = np.random.RandomState(options_.seed + options_.initial_index)

    main(options_)
