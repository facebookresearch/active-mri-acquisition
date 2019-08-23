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
            logger.info(f'Episode {episode}. Saving statistics to {options_.tb_logs_dir}.')
            np.save(
                os.path.join(options_.tb_logs_dir, 'test_stats_mse_{}'.format(episode)),
                statistics_mse)
            np.save(
                os.path.join(options_.tb_logs_dir, 'test_stats_ssim_{}'.format(episode)),
                statistics_ssim)
            np.save(
                os.path.join(options_.tb_logs_dir, 'test_stats_psnr_{}'.format(episode)),
                statistics_ssim)
            np.save(os.path.join(options_.tb_logs_dir, 'all_actions'), np.array(all_actions))
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
    return '{}_bu{}_seed{}_neptest{}'.format(policy_str, options_.budget, options_.seed,
                                             options_.num_test_images)


class DQNTrainer:

    def __init__(self, options_, env=None, writer=None, logger=None):
        self.options = options_
        self.env = env

        if self.env is not None:
            self.env = env
            self.writer = writer
            self.logger = logger

            max_num_steps = options_.num_train_episodes * self.env.action_space.n
            replay_memory_size = min(max_num_steps, options_.replay_buffer_size)
            replay_memory = util.rl.replay_buffer.ReplayMemory(
                replay_memory_size, self.env.observation_space.shape, options_.rl_batch_size,
                options_.rl_burn_in)
            self.policy = util.rl.dqn.DDQN(self.env.action_space.n, rl_env.device, replay_memory,
                                           options_).to(rl_env.device)
            self.target_net = util.rl.dqn.DDQN(self.env.action_space.n, rl_env.device, None,
                                               options_).to(rl_env.device)

    def _train_dqn_policy(self):
        """ Trains the DQN policy. """
        steps = 0
        best_test_score = np.inf
        for episode in range(self.options.num_train_episodes):
            self.logger.info('Episode {}'.format(episode + 1))
            obs, _ = self.env.reset()
            done = False
            total_reward = 0
            episode_actions = []
            cnt_repeated_actions = 0
            while not done:
                epsilon = util.rl.dqn.get_epsilon(steps, self.options)
                action = self.policy.get_action(obs, epsilon, episode_actions)
                next_obs, reward, done, _ = self.env.step(action)
                steps += 1
                is_zero_target = (
                    done or
                    action in episode_actions) if self.options.no_replacement_policy else done
                self.policy.add_experience(obs, action, next_obs, reward, is_zero_target)
                loss, grad_norm, mean_q_values, std_q_values = self.policy.update_parameters(
                    self.target_net)

                if steps % self.options.target_net_update_freq == 0:
                    self.logger.info('Updating target network.')
                    self.target_net.load_state_dict(self.policy.state_dict())

                # Adding per-step tensorboard logs
                self.writer.add_scalar('epsilon', epsilon, steps)
                if loss is not None:
                    self.writer.add_scalar('loss', loss, steps)
                    self.writer.add_scalar('grad_norm', grad_norm, steps)
                    self.writer.add_scalar('mean_q_values', mean_q_values, steps)
                    self.writer.add_scalar('std_q_values', std_q_values, steps)

                total_reward += reward
                obs = next_obs
                cnt_repeated_actions += int(action in episode_actions)
                episode_actions.append(action)

            # Adding per-episode tensorboard logs
            self.writer.add_scalar('episode_reward', total_reward, episode)
            self.writer.add_scalar('cnt_repeated_actions', cnt_repeated_actions, episode)

            # Evaluate the current policy
            if (episode + 1) % self.options.agent_test_episode_freq == 0:
                test_score = test_policy(self.env, self.policy, self.writer, self.logger, None,
                                         episode, self.options)
                if test_score < best_test_score:
                    self.policy.save(os.path.join(self.options.tb_logs_dir, 'policy_best.pt'))
                    best_test_score = test_score
                    self.logger.info('Saved model to {}'.format(
                        os.path.join(self.options.tb_logs_dir, 'policy_checkpoint.pt')))

        return best_test_score

    def __call__(self):
        if self.env is None:
            # Initialize everything
            self.writer = tensorboardX.SummaryWriter(
                os.path.join(self.options.checkpoints_dir, 'tb_logs'))
            self.logger = logging.getLogger()
            self.logger.setLevel(logging.DEBUG)
            fh = logging.FileHandler(os.path.join(self.options.checkpoints_dir, 'train.log'))
            fh.setLevel(logging.DEBUG)
            formatter = logging.Formatter(
                '%(asctime)s - %(threadName)s - %(levelname)s: %(message)s')
            fh.setFormatter(formatter)
            self.logger.addHandler(fh)
            self.env = rl_env.ReconstructionEnv(
                rl_env.generate_initial_mask(self.options.initial_num_lines), self.options)
            self.env.set_training()

            self.logger.info(f'Created environment with {self.env.action_space.n} actions')
            self.policy = get_policy(self.env, self.writer, self.logger, self.options)

            max_num_steps = self.options.num_train_episodes * self.env.action_space.n
            replay_memory_size = min(max_num_steps, self.options.replay_buffer_size)
            replay_memory = util.rl.replay_buffer.ReplayMemory(
                replay_memory_size, self.env.observation_space.shape, self.options.rl_batch_size,
                self.options.rl_burn_in)
            self.policy = util.rl.dqn.DDQN(self.env.action_space.n, rl_env.device, replay_memory,
                                           self.options).to(rl_env.device)
            self.target_net = util.rl.dqn.DDQN(self.env.action_space.n, rl_env.device, None,
                                               self.options).to(rl_env.device)

        if self.options.dqn_resume:
            # TODO to be able to resume training need some code to store the replay buffer
            raise NotImplementedError
        if self.options.dqn_only_test:
            policy_path = os.path.join(self.options.dqn_load_dir, 'policy_best.pt')
            if os.path.isfile(policy_path):
                self.policy.load(policy_path)
                self.logger.info(f'Policy found in {policy_path}.')
            else:
                self.logger.warning(f'No policy found in {policy_path}.')

        else:
            return -self._train_dqn_policy()  # Hyperparameter tuner tries to maximize


# Not a great policy specification protocol, but works for now.
def get_policy(env, writer, logger, options_):
    # This options affects how the score is computed by the environment
    # (whether it passes through reconstruction network or not after a new col. is scanned)
    options_.use_reconstructions = (options_.policy[-2:] == '_r')
    logger.info('Use reconstructions is {}'.format(options_.use_reconstructions))
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
        dqn_trainer = DQNTrainer(options_, env, writer, logger)
        dqn_trainer()
        policy = dqn_trainer.policy
    else:
        raise ValueError

    return policy


def main(options_, logger):
    writer = tensorboardX.SummaryWriter(options_.tb_logs_dir)
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
    opts.tb_logs_dir = os.path.join(opts.checkpoints_dir, opts.rl_logs_subdir, experiment_str)
    if not os.path.isdir(opts.tb_logs_dir):
        os.makedirs(opts.tb_logs_dir)

    # Initializing logger
    logger_ = logging.getLogger()
    logger_.setLevel(logging.DEBUG)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.DEBUG)
    fh = logging.FileHandler(os.path.join(opts.tb_logs_dir, 'train.log'))
    fh.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(threadName)s - %(levelname)s: %(message)s')
    ch.setFormatter(formatter)
    fh.setFormatter(formatter)
    logger_.addHandler(ch)
    logger_.addHandler(fh)

    logger_.info(f'Results will be saved at {opts.tb_logs_dir}.')

    main(opts, logger_)
