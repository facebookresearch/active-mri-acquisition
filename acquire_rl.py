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


def test_policy(env, policy, writer, num_episodes, step, opts):
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
            env.compute_score(opts.use_reconstructions, kind='mse')[0], episode_step,
            statistics_mse)
        update_statistics(
            env.compute_score(opts.use_reconstructions, kind='ssim')[0], episode_step,
            statistics_ssim)
        update_statistics(
            env.compute_score(opts.use_reconstructions, kind='psnr')[0], episode_step,
            statistics_psnr)
        while not done:
            action = policy.get_action(obs, 0., actions)
            actions.append(action)
            next_obs, reward, done, _ = env.step(action)
            total_reward += reward
            obs = next_obs
            episode_step += 1
            update_statistics(
                env.compute_score(opts.use_reconstructions, kind='mse')[0], episode_step,
                statistics_mse)
            update_statistics(
                env.compute_score(opts.use_reconstructions, kind='ssim')[0], episode_step,
                statistics_ssim)
            update_statistics(
                env.compute_score(opts.use_reconstructions, kind='psnr')[0], episode_step,
                statistics_psnr)
        average_total_reward += total_reward
        all_actions.append(actions)
        logging.debug('Actions and reward: {}, {}'.format(actions, total_reward))
        if episode % opts.freq_save_test_stats == 0 or episode == num_episodes:
            logging.info(f'Episode {episode}. Saving statistics to {opts.tb_logs_dir}.')
            np.save(
                os.path.join(opts.tb_logs_dir, 'test_stats_mse_{}'.format(episode)), statistics_mse)
            np.save(
                os.path.join(opts.tb_logs_dir, 'test_stats_ssim_{}'.format(episode)),
                statistics_ssim)
            np.save(
                os.path.join(opts.tb_logs_dir, 'test_stats_psnr_{}'.format(episode)),
                statistics_ssim)
            np.save(os.path.join(opts.tb_logs_dir, 'all_actions'), np.array(all_actions))
    end = time.time()
    logging.debug('Test run lasted {} seconds.'.format(end - start))
    writer.add_scalar('eval/average_reward', average_total_reward / episode, step)
    env.set_training()

    return compute_test_score_from_stats(statistics_mse)


def train_policy(env, policy, target_net, writer, opts):
    steps = 0
    best_test_score = np.inf
    for episode in range(opts.num_train_episodes):
        logging.info('Episode {}'.format(episode + 1))
        obs, _ = env.reset()
        done = False
        total_reward = 0
        episode_actions = []
        cnt_repeated_actions = 0
        while not done:
            epsilon = util.rl.dqn.get_epsilon(steps, opts)
            action = policy.get_action(obs, epsilon, episode_actions)
            next_obs, reward, done, _ = env.step(action)
            steps += 1
            is_zero_target = (done or
                              action in episode_actions) if opts.no_replacement_policy else done
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
            test_score = test_policy(env, policy, writer, None, episode, opts)
            if test_score < best_test_score:
                policy.save(os.path.join(opts.tb_logs_dir, 'policy_best.pt'))
                best_test_score = test_score
                logging.info('Saved model to {}'.format(
                    os.path.join(opts.tb_logs_dir, 'policy_checkpoint.pt')))


def get_experiment_str(opts):
    if opts.policy == 'dqn':
        policy_str = '{}.bu{}.tupd{}.bs{}.edecay{}.gamma{}.norepl{}.nimgtr{}.nimgtest{}_'.format(
            opts.obs_type, opts.budget,
            opts.target_net_update_freq, opts.rl_batch_size, opts.epsilon_decay, opts.gamma,
            int(opts.no_replacement_policy), opts.num_train_images, opts.num_test_images)
    else:
        policy_str = opts.policy
        if 'greedymc' in opts.policy:
            policy_str = '{}.nsam{}.hor{}_'.format(policy_str, opts.greedymc_num_samples,
                                                   opts.greedymc_horizon)
    return '{}_bu{}_seed{}_neptest{}'.format(policy_str, opts.budget, opts.seed,
                                             opts.num_test_images)


# Not a great policy specification protocol, but works for now.
def get_policy(env, writer, opts):
    # This options affects how the score is computed by the environment
    # (whether it passes through reconstruction network or not after a new col. is scanned)
    opts.use_reconstructions = (opts.policy[-2:] == '_r')
    logging.info('Use reconstructions is {}'.format(opts.use_reconstructions))
    if 'random' in opts.policy:
        policy = util.rl.simple_baselines.RandomPolicy(range(env.action_space.n))
    elif 'lowfirst' in opts.policy:
        assert rl_env.CONJUGATE_SYMMETRIC
        policy = util.rl.simple_baselines.NextIndexPolicy(range(env.action_space.n))
    elif 'greedymc' in opts.policy:
        policy = util.rl.simple_baselines.GreedyMC(
            env,
            samples=opts.greedymc_num_samples,
            horizon=opts.greedymc_horizon,
            use_reconstructions=opts.use_reconstructions,
            use_ground_truth='_gt' in opts.policy)
    elif 'greedyfull1' in opts.policy:
        assert opts.use_reconstructions
        policy = util.rl.simple_baselines.FullGreedy(
            env, num_steps=1, use_ground_truth='_gt' in opts.policy, use_reconstructions=True)
    elif 'greedyzero' in opts.policy:
        assert opts.use_reconstructions
        policy = util.rl.simple_baselines.ZeroStepGreedy(env)
    elif 'evaluator_net' in opts.policy:
        policy = util.rl.simple_baselines.EvaluatorNetwork(env)
    elif 'evaluator_net_offp' in opts.policy:
        assert opts.evaluator_name is not None and opts.evaluator_name != opts.name
        policy = util.rl.simple_baselines.EvaluatorNetwork(env)
    elif 'evaluator++' in opts.policy:
        assert opts.obs_type == 'concatenate_mask'
        policy = util.rl.evaluator_plus_plus.EvaluatorPlusPlusPolicy(
            os.path.join(opts.checkpoints_dir, opts.evaluator_pp_path), opts.initial_num_lines,
            rl_env.device)
    elif opts.policy == 'dqn':

        replay_memory = util.rl.replay_buffer.ReplayMemory(opts.replay_buffer_size,
                                                           env.observation_space.shape,
                                                           opts.rl_batch_size, opts.rl_burn_in)
        policy = util.rl.dqn.DDQN(env.action_space.n, rl_env.device, replay_memory, opts).to(
            rl_env.device)
        target_net = util.rl.dqn.DDQN(env.action_space.n, rl_env.device, None, opts).to(
            rl_env.device)

        if opts.dqn_resume:
            # TODO to be able to resume training need some code to store the replay buffer
            raise NotImplementedError
        if opts.dqn_only_test:
            policy_path = os.path.join(opts.dqn_load_dir, 'policy_best.pt')
            if os.path.isfile(policy_path):
                policy.load(policy_path)
                logging.info(f'Policy found in {policy_path}.')
            else:
                logging.warning(f'No policy found in {policy_path}.')

        else:
            train_policy(env, policy, target_net, writer, opts)
    else:
        raise ValueError

    return policy


def main(options):
    writer = tensorboardX.SummaryWriter(options.tb_logs_dir)
    env = rl_env.ReconstructionEnv(rl_env.generate_initial_mask(options.initial_num_lines), options)
    env.set_training()
    logging.info(f'Created environment with {env.action_space.n} actions')
    policy = get_policy(env, writer, options)  # Trains if necessary
    env.set_testing()
    test_policy(env, policy, writer, None, 0, options)


if __name__ == '__main__':
    # Reading options
    opts = options.rl_options.RLOptions().parse()
    opts.batchSize = 1
    if opts.results_dir is None:
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
    fh = logging.FileHandler(os.path.join(opts.tb_logs_dir, 'train.log'))
    fh.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(threadName)s - %(levelname)s: %(message)s')
    ch.setFormatter(formatter)
    fh.setFormatter(formatter)
    logger.addHandler(ch)
    logger.addHandler(fh)

    logging.info(f'Results will be saved at {opts.tb_logs_dir}.')

    main(opts)
