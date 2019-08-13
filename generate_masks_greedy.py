import argparse
import logging
import os
import sys

import numpy as np

import rl_env
import util.rl.dqn
import util.rl.evaluator_plus_plus
import util.rl.replay_buffer
import util.rl.simple_baselines


def generate_masks(env: rl_env.ReconstructionEnv,
                   greedy_policy: util.rl.simple_baselines.FullGreedy):
    import time
    start = time.time()
    initial_num_lines = env.options.initial_num_lines
    num_non_scanned_lines = (rl_env.IMAGE_WIDTH // 2) - initial_num_lines
    for episode in range(env.num_train_images):
        obs, info = env.reset()
        greedy_policy.init_episode()
        episode += 1
        done = False
        how_many_lines = int(num_non_scanned_lines * np.random.beta(1, 4))
        mask = np.zeros(rl_env.IMAGE_WIDTH)
        mask[:initial_num_lines] = mask[-initial_num_lines:] = 1
        while not done and how_many_lines > 0:
            action = greedy_policy.get_action(obs, None, None)
            next_obs, _, done, _ = env.step(action)
            obs = next_obs
            how_many_lines -= 1
            mask[initial_num_lines + action] = 1
        if (episode + 1) % 500 == 0:
            logging.info(f"Processed image {info['image_idx']}")
    end = time.time()
    logging.debug('Generated all masks. Total time was {} seconds.'.format(end - start))


def main(options: argparse.Namespace):
    env = rl_env.ReconstructionEnv(rl_env.generate_initial_mask(options.initial_num_lines), options)
    env.set_training()
    logging.info('Created environment with {} actions'.format(env.action_space.n))
    policy = util.rl.simple_baselines.FullGreedy(
        env, num_steps=1, use_ground_truth=True, use_reconstructions=True)
    generate_masks(env, policy)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoints_dir', type=str, required=True)
    parser.add_argument('--dataset_dir', type=str, required=True)
    parser.add_argument('--initial_num_lines', type=int, default=10)
    parser.add_argument('--dataroot', type=str, default='KNEE')
    options_ = parser.parse_args()

    # Adding config options expected by `rl_env.ReconstructionEnv`
    options_.sequential_images = True
    options_.budget = rl_env.IMAGE_WIDTH
    options_.obs_type = 'two_streams'
    options_.num_train_images = None
    options_.num_test_images = None
    options_.batchSize = 1
    options_.nThreads = 1

    # Initializing logger
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.DEBUG)
    fh = logging.FileHandler(os.path.join(options_.dataset_dir, 'generation.log'))
    fh.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(threadName)s - %(levelname)s: %(message)s')
    ch.setFormatter(formatter)
    fh.setFormatter(formatter)
    logger.addHandler(ch)
    logger.addHandler(fh)

    main(options_)
