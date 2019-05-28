import logging
import numpy as np
import os
import random
import sys
import torch

from options.rl_options import RLOptions
from tensorboardX import SummaryWriter
from util.rl.dqn import DDQN, get_epsilon
from util.rl.simple_baselines import RandomPolicy, NextIndexPolicy, GreedyMC
from util.rl.replay_buffer import ReplayMemory

from rl_env import ReconstrunctionEnv, device, generate_initial_mask, CONJUGATE_SYMMETRIC


def update_statisics(value, episode_step, statistics):
    if episode_step not in statistics:
        statistics[episode_step] = {'mean': 0, 'm2': 0, 'count': 0}
    statistics[episode_step]['count'] += 1
    delta = value - statistics[episode_step]['mean']
    statistics[episode_step]['mean'] += delta / statistics[episode_step]['count']
    delta2 = value - statistics[episode_step]['mean']
    statistics[episode_step]['m2'] += delta * delta2


def test_policy(env, policy, writer, num_episodes, step, opts):
    average_total_reward = 0
    episode = 0
    statistics = {}
    import time
    start = time.time()
    while True:
        episode += 1
        obs = env.reset()
        policy.init_episode()
        if episode == num_episodes or obs is None:
            break
        done = False
        total_reward = 0
        actions = []
        episode_step = 0
        update_statisics(env.compute_score(opts.use_reconstructions), episode_step, statistics)
        while not done:
            action = policy.get_action(obs, 0., actions)
            actions.append(action)
            next_obs, reward, done, _ = env.step(action)
            if done:
                next_obs = None
            total_reward += reward
            obs = next_obs
            episode_step += 1
            update_statisics(env.compute_score(opts.use_reconstructions), episode_step, statistics)
        average_total_reward += total_reward
        if episode == 0:
            logging.debug(actions)
        if episode % 100 == 0:
            logging.info('Episode {}. Saving statistics'.format(episode))
            np.save(os.path.join(opts.tb_logs_dir, 'test_stats_{}'.format(episode)), statistics)
    end = time.time()
    logging.debug('Test run lasted {} seconds.'.format(end - start))
    writer.add_scalar('eval/average_reward', average_total_reward / episode, step)


def train_policy(env, policy, target_net, writer, opts):
    steps = 0
    for episode in range(opts.num_episodes):
        logging.info('Episode {}'.format(episode))
        obs = env.reset()
        done = False
        total_reward = 0
        episode_actions = []
        cnt_repeated_actions = 0
        while not done:
            epsilon = get_epsilon(steps, opts)
            action = policy.get_action(obs, epsilon, episode_actions)
            next_obs, reward, done, _ = env.step(action)
            steps += 1
            is_zero_target = (done or action in episode_actions) if opts.no_replacement_policy else done
            policy.add_experience(obs, action, next_obs, reward, is_zero_target)
            loss, grad_norm, mean_q_values, std_q_values = policy.update_parameters(target_net)

            if steps % opts.target_net_update_freq == 0:
                logging.info('Updating target network.')
                target_net.load_state_dict(policy.state_dict())

            # Adding per-step tensorboard logs
            writer.add_scalar('epsilon', epsilon, steps)
            if loss is not None:
                writer.add_scalar('loss', loss, steps)
                writer.add_scalar('grad_norm', grad_norm, steps)
                writer.add_scalar('mean_q_values', mean_q_values, steps)
                writer.add_scalar('std_q_values', std_q_values, steps)

            total_reward += reward
            obs = next_obs
            cnt_repeated_actions += int(action in episode_actions)
            episode_actions.append(action)

        # Adding per-episode tensorboard logs
        writer.add_scalar('episode_reward', total_reward, episode)
        writer.add_scalar('cnt_repeated_actions', cnt_repeated_actions, episode)

        # Evaluate the current policy
        if (episode + 1) % opts.agent_test_episode_freq == 0:
            test_policy(env, policy, writer, 1, episode)


def get_experiment_str(opts):
    if opts.policy == 'dqn':
        policy_str = '{}.bu{}.tupd{}.bs{}.edecay{}.gamma{}.norepl{}.npetr{}_'.format(
            opts.rl_model_type, opts.budget, opts.target_net_update_freq, opts.rl_batch_size, opts.epsilon_decay,
            opts.gamma, int(opts.no_replacement_policy), opts.num_episodes)
    else:
        policy_str = opts.policy
        if 'greedymc' in opts.policy:
            policy_str = '{}.nsam{}.hor{}_'.format(policy_str, opts.greedymc_num_samples, opts.greedymc_horizon)
    return '{}_bu{}_seed{}_neptest{}'.format(policy_str, opts.budget, opts.seed, opts.num_test_episodes)


def main(opts):
    writer = SummaryWriter(opts.tb_logs_dir)

    env = ReconstrunctionEnv(generate_initial_mask(opts.initial_num_lines), opts)
    env.set_training()

    logging.info('Created environment with {} actions'.format(env.action_space.n))

    if opts.policy == 'random':
        policy = RandomPolicy(range(env.action_space.n))
        opts.use_reconstructions = False
    elif opts.policy == 'random_r':
        policy = RandomPolicy(range(env.action_space.n))
        opts.use_reconstructions = True
    elif opts.policy == 'lowfirst':
        assert CONJUGATE_SYMMETRIC
        policy = NextIndexPolicy(range(env.action_space.n))
        opts.use_reconstructions = False
    elif opts.policy == 'lowfirst_r':
        assert CONJUGATE_SYMMETRIC
        policy = NextIndexPolicy(range(env.action_space.n))
        opts.use_reconstructions = True
    elif opts.policy == 'greedymc':
        policy = GreedyMC(env, samples=opts.greedymc_num_samples, horizon=opts.greedymc_horizon)
        opts.use_reconstructions = True
    elif opts.policy == 'greedymc_gt':
        policy = GreedyMC(env, samples=opts.greedymc_num_samples, horizon=opts.greedymc_horizon, use_ground_truth=True)
        opts.use_reconstructions = True
    elif opts.policy == 'dqn':
        replay_memory = ReplayMemory(opts.size_replay_buffer, env.observation_space.shape)
        policy = DDQN(env.action_space.n, device, replay_memory, opts).to(device)
        target_net = DDQN(env.action_space.n, device, None, opts).to(device)
        train_policy(env, policy, target_net, writer, opts)
    else:
        raise ValueError

    env.set_testing()
    test_policy(env, policy, writer, None, 0, opts)


if __name__ == '__main__':
    # Reading options
    opts = RLOptions().parse()
    opts.batchSize = 1
    opts.results_dir = opts.checkpoints_dir

    random.seed(opts.seed)
    np.random.seed(opts.seed)
    torch.manual_seed(opts.seed)

    experiment_str = get_experiment_str(opts)
    opts.tb_logs_dir = os.path.join(opts.results_dir, opts.rl_logs_subdir, experiment_str)
    if not os.path.isdir(opts.tb_logs_dir):
        os.makedirs(opts.tb_logs_dir)

    # Initializing logger
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(threadName)s - %(levelname)s: %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    main(opts)
