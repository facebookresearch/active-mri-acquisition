import logging
import math
import os
import random

import numpy as np
import submitit
import tensorboardX
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import acquire_rl
import models.evaluator
import rl_env
import util.rl.replay_buffer


def get_epsilon(steps_done, opts):
    return opts.epsilon_end + (opts.epsilon_start - opts.epsilon_end) * \
        math.exp(-1. * steps_done / opts.epsilon_decay)


class Flatten(nn.Module):

    def forward(self, x):
        return x.view(x.size(0), -1)


class EvaluatorBasedValueNetwork(nn.Module):

    def __init__(self, image_width, initial_num_lines_per_side, mask_embed_dim):
        super(EvaluatorBasedValueNetwork, self).__init__()
        num_actions = image_width - 2 * initial_num_lines_per_side
        self.evaluator = models.evaluator.EvaluatorNetwork(
            number_of_filters=256,
            number_of_conv_layers=4,
            use_sigmoid=False,
            width=image_width,
            mask_embed_dim=mask_embed_dim,
            num_output_channels=num_actions)
        self.mask_embed_dim = mask_embed_dim
        self.initial_num_lines_per_side = initial_num_lines_per_side

    def forward(self, x):
        reconstruction = x[..., :-2, :]
        mask = x[:, 0, -2, self.initial_num_lines_per_side:-self.initial_num_lines_per_side]
        mask_embedding = x[0, 0, -1, :self.mask_embed_dim].view(1, -1, 1, 1)
        mask_embedding = mask_embedding.repeat(reconstruction.shape[0], 1, reconstruction.shape[2],
                                               reconstruction.shape[3])
        value = self.evaluator(reconstruction, mask_embedding)
        # This makes the DQN max over the target values consider only non repeated actions
        value = value - 1e10 * mask
        return value


class BasicValueNetwork(nn.Module):
    """ The input to this model includes the reconstruction and the current mask. The
        reconstruction is passed through a convolutional path and the mask through a few MLP layers.
        Then both outputs are combined to produce the final value.
    """

    def __init__(self, num_actions):
        super(BasicValueNetwork, self).__init__()

        self.conv_image = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=8, stride=4, padding=0), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0), nn.ReLU(), Flatten())

        self.conv_fft = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=8, stride=4, padding=0), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0), nn.ReLU(), Flatten())

        self.fc = nn.Sequential(
            nn.Linear(2 * 12 * 12 * 64, 512), nn.ReLU(), nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, num_actions))

    def forward(self, x):
        reconstructions = x[:, :2, :, :]
        masked_rffts = x[:, 2:, :, :]
        image_encoding = self.conv_image(reconstructions)
        rfft_encoding = self.conv_image(masked_rffts)
        return self.fc(torch.cat((image_encoding, rfft_encoding), dim=1))


def get_model(num_actions, options):
    if options.dqn_model_type == 'basic':
        return BasicValueNetwork(num_actions)
    if options.dqn_model_type == 'evaluator':
        return EvaluatorBasedValueNetwork(options.image_width, options.initial_num_lines_per_side,
                                          options.mask_embedding_dim)
    raise ValueError('Unknown model specified for DQN.')


class DDQN(nn.Module):

    def __init__(self, num_actions, device, memory, opts):
        super(DDQN, self).__init__()
        self.model = get_model(num_actions, opts)
        self.memory = memory
        self.optimizer = optim.Adam(self.parameters(), lr=opts.dqn_learning_rate)
        self.num_actions = num_actions
        self.opts = opts
        self.device = device

    def forward(self, x):
        return self.model(x)

    def _get_action_no_replacement(self, observation, eps_threshold):
        sample = random.random()
        previous_actions = np.nonzero(observation[0, -2, self.opts.initial_num_lines_per_side:
                                                  -self.opts.initial_num_lines_per_side])[0]
        if sample < eps_threshold:
            return random.choice([x for x in range(self.num_actions) if x not in previous_actions])
        with torch.no_grad():
            q_values = self(torch.from_numpy(observation).unsqueeze(0).to(self.device))
        return torch.argmax(q_values, dim=1).item()

    def _get_action_standard_e_greedy(self, observation, eps_threshold):
        sample = random.random()
        if sample < eps_threshold:
            return random.randrange(self.num_actions)
        with torch.no_grad():
            q_values = self(torch.from_numpy(observation).unsqueeze(0).to(self.device))
        return torch.argmax(q_values, dim=1).item()

    def get_action(self, observation, eps_threshold, _):
        if self.opts.no_replacement_policy:
            return self._get_action_no_replacement(observation, eps_threshold)
        else:
            return self._get_action_standard_e_greedy(observation, eps_threshold)

    def add_experience(self, observation, action, next_observation, reward, done):
        self.memory.push(observation, action, next_observation, reward, done)

    def update_parameters(self, target_net):
        batch = self.memory.sample()
        if batch is None:
            return None, None, None, None
        observations = batch['observations'].to(self.device)
        next_observations = batch['next_observations'].to(self.device)
        actions = batch['actions'].to(self.device)
        rewards = batch['rewards'].to(self.device).squeeze()
        dones = batch['dones'].to(self.device)

        not_done_mask = (1 - dones).squeeze()

        # Compute Q-values and get best action according to online network
        all_q_values_cur = self.forward(observations)
        q_values = all_q_values_cur.gather(1, actions.unsqueeze(1))

        # Compute target values using the best action found
        with torch.no_grad():
            all_q_values_next = self.forward(next_observations)
            target_values = torch.zeros(observations.shape[0], device=self.device)
            if not_done_mask.any().item() != 0:
                best_actions = all_q_values_next.detach().max(1)[1]
                target_values[not_done_mask] = target_net(next_observations)\
                    .gather(1, best_actions.unsqueeze(1))[not_done_mask].squeeze().detach()

            target_values = self.opts.gamma * target_values + rewards

        # loss = F.mse_loss(q_values, target_values.unsqueeze(1))
        loss = F.smooth_l1_loss(q_values, target_values.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()

        # Compute total gradient norm (for logging purposes) and then clip gradients
        grad_norm = 0
        for p in list(filter(lambda p: p.grad is not None, self.parameters())):
            grad_norm += (p.grad.data.norm(2).item()**2)
        grad_norm = grad_norm**.5
        # grad_norm = torch.nn.utils.clip_grad_norm_(self.parameters(), 100)
        torch.nn.utils.clip_grad_value_(self.parameters(), 1)

        self.optimizer.step()

        return loss, grad_norm, \
               q_values.detach().mean().cpu().numpy(), \
               q_values.detach().std().cpu().numpy()

    def init_episode(self):
        pass


class DQNTrainer:

    def __init__(self, options_, env=None, writer=None, logger=None, load_replay_mem=True):
        self.options = options_
        self.env = env
        self.steps = 0
        self.episode = 0
        self.best_test_score = np.inf

        self.load_replay_mem = options_.dqn_resume and load_replay_mem

        if self.env is not None:
            self.env = env
            self.writer = writer
            self.logger = logger

            # If replay will be loaded set capacity to zero (defer allocation to `__call__()`)
            mem_capacity = 0 if self.load_replay_mem or options_.dqn_only_test \
                else self._max_replay_buffer_size()
            self.replay_memory = util.rl.replay_buffer.ReplayMemory(
                mem_capacity,
                self.env.observation_space.shape,
                options_.rl_batch_size,
                options_.dqn_burn_in,
                use_normalization=options_.dqn_normalize)

            self.policy = DDQN(self.env.action_space.n, rl_env.device, self.replay_memory,
                               options_).to(rl_env.device)
            self.target_net = DDQN(self.env.action_space.n, rl_env.device, None, options_).to(
                rl_env.device)

    def _max_replay_buffer_size(self):
        return min(self.options.num_train_steps, self.options.replay_buffer_size)

    def _init_all(self):
        # This is here so that it appears in SLURM stdout logs
        print(f'Checkpoint dir for this job is {self.options.checkpoints_dir}', flush=True)

        # Initialize writer and logger
        self.writer = tensorboardX.SummaryWriter(
            os.path.join(self.options.checkpoints_dir, 'tb_logs'))
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.DEBUG)
        fh = logging.FileHandler(os.path.join(self.options.checkpoints_dir, 'train.log'))
        fh.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(threadName)s - %(levelname)s: %(message)s')
        fh.setFormatter(formatter)
        self.logger.addHandler(fh)

        # Initialize environment
        self.env = rl_env.ReconstructionEnv(self.options)
        self.env.set_training()
        self.logger.info(f'Created environment with {self.env.action_space.n} actions')

        # If replay will be loaded, set capacity to zero (defer allocation to `__call__()`)
        mem_capacity = 0 if self.load_replay_mem or self.options.dqn_only_test \
            else self._max_replay_buffer_size()
        self.replay_memory = util.rl.replay_buffer.ReplayMemory(
            mem_capacity,
            self.env.observation_space.shape,
            self.options.rl_batch_size,
            self.options.dqn_burn_in,
            use_normalization=self.options.dqn_normalize)

        # Initialize policy
        self.policy = DDQN(self.env.action_space.n, rl_env.device, self.replay_memory,
                           self.options).to(rl_env.device)
        self.target_net = DDQN(self.env.action_space.n, rl_env.device, None, self.options).to(
            rl_env.device)

    def load_checkpoint_if_needed(self):
        if self.options.dqn_only_test:
            policy_path = os.path.join(self.options.dqn_weights_dir, 'policy_best.pt')
            if os.path.isfile(policy_path):
                self.load(policy_path)
                self.logger.info(f'Loaded DQN policy found at {policy_path}.')
            else:
                self.logger.error(f'No DQN policy found at {policy_path}.')
                raise FileNotFoundError

        elif self.options.dqn_resume:
            policy_path = os.path.join(self.options.checkpoints_dir, 'policy_checkpoint.pt')
            if os.path.isfile(policy_path):
                self.load(policy_path)
                self.logger.info(f'Loaded DQN policy found at {policy_path}. Steps was set to '
                                 f'{self.steps}. Episodes set to {self.episode}.')
            else:
                self.logger.info(f'No policy found at {policy_path}, continue without checkpoint.')
            if self.load_replay_mem:
                memory_path = os.path.join(self.options.checkpoints_dir, 'replay_buffer.pt')
                if os.path.isfile(memory_path):
                    old_len = self.replay_memory.load(
                        memory_path, capacity=self._max_replay_buffer_size())
                    self.logger.info(f'Loaded replay memory from {memory_path}. '
                                     f'Capacity was set to {self._max_replay_buffer_size()}. '
                                     f'Previous capacity was {old_len}.')
                else:
                    mem_capacity = self._max_replay_buffer_size()
                    self.logger.warning(f'load_replay_mem was True, but no checkpoint was found '
                                        f'at {memory_path}. Allocating a new buffer with '
                                        f'capacity {mem_capacity}.')

                    self.replay_memory = util.rl.replay_buffer.ReplayMemory(
                        mem_capacity,
                        self.env.observation_space.shape,
                        self.options.rl_batch_size,
                        self.options.dqn_burn_in,
                        use_normalization=self.options.dqn_normalize)
                    self.logger.info('Finished allocating replay buffer.')
                    self.policy.memory = self.replay_memory

    def _train_dqn_policy(self):
        """ Trains the DQN policy. """
        self.logger.info(f'Starting training at step {self.steps}/{self.options.num_train_steps}. '
                         f'Best score so far is {self.best_test_score}.')
        while self.steps < self.options.num_train_steps:
            self.logger.info('Episode {}'.format(self.episode + 1))
            obs, _ = self.env.reset()
            done = False
            total_reward = 0
            cnt_repeated_actions = 0
            while not done:
                epsilon = get_epsilon(self.steps, self.options)
                action = self.policy.get_action(obs, epsilon, None)
                is_repeated_action = bool(
                    obs[0, -2, action + self.options.initial_num_lines_per_side])

                assert not (self.options.no_replacement_policy and is_repeated_action)

                next_obs, reward, done, _ = self.env.step(action)
                self.steps += 1
                self.policy.add_experience(obs, action, next_obs, reward, done)
                loss, grad_norm, mean_q_values, std_q_values = self.policy.update_parameters(
                    self.target_net)

                if self.steps % self.options.target_net_update_freq == 0:
                    self.logger.info('Updating target network.')
                    self.target_net.load_state_dict(self.policy.state_dict())

                # Adding per-step tensorboard logs
                if self.steps % 50 == 0:
                    self.writer.add_scalar('epsilon', epsilon, self.steps)
                    if loss is not None:
                        self.writer.add_scalar('loss', loss, self.steps)
                        self.writer.add_scalar('grad_norm', grad_norm, self.steps)
                        self.writer.add_scalar('mean_q_values', mean_q_values, self.steps)
                        self.writer.add_scalar('std_q_values', std_q_values, self.steps)

                total_reward += reward
                obs = next_obs

                cnt_repeated_actions += int(is_repeated_action)

            # Adding per-episode tensorboard logs
            self.writer.add_scalar('episode_reward', total_reward, self.episode)
            self.writer.add_scalar('cnt_repeated_actions', cnt_repeated_actions, self.episode)

            # Evaluate the current policy
            if (self.episode + 1) % self.options.dqn_test_episode_freq == 0:
                test_score = acquire_rl.test_policy(self.env, self.policy, self.writer, self.logger,
                                                    None, self.episode, self.options)
                if test_score < self.best_test_score:
                    policy_path = os.path.join(self.options.checkpoints_dir, 'policy_best.pt')
                    self.save(policy_path)
                    self.best_test_score = test_score
                    self.logger.info(
                        f'Saved DQN model with score {self.best_test_score} to {policy_path}.')

            self.episode += 1

        self.checkpoint(submit_job=False)
        return self.best_test_score

    def __call__(self):
        if self.env is None:
            self._init_all()
        self.load_checkpoint_if_needed()
        if self.options.dqn_only_test:
            return None
        # Hyperparameter tuner tries to maximize, but `train_dqn_policy`
        # returns MSE over validation set (which we want to minimize)
        return -self._train_dqn_policy()

    def checkpoint(self, submit_job=True):
        self.logger.info('Received preemption signal.')
        policy_path = os.path.join(self.options.checkpoints_dir, 'policy_checkpoint.pt')
        self.save(policy_path)
        self.logger.info(f'Saved DQN checkpoint to {policy_path}. Now saving replay memory.')
        memory_path = self.replay_memory.save(self.options.checkpoints_dir, 'replay_buffer.pt')
        self.logger.info(f'Saved replay buffer to {memory_path}.')
        trainer = DQNTrainer(self.options, load_replay_mem=True)
        if submit_job:
            return submitit.helpers.DelayedSubmission(trainer)

    def save(self, path):
        torch.save({
            'dqn_weights': self.policy.state_dict(),
            'target_weights': self.target_net.state_dict(),
            'episode': self.episode,
            'steps': self.steps,
            'best_test_score': self.best_test_score,
        }, path)

    def load(self, path):
        checkpoint = torch.load(path)
        self.policy.load_state_dict(checkpoint['dqn_weights'])
        self.target_net.load_state_dict(checkpoint['target_weights'])
        self.steps = checkpoint['steps']
        self.episode = checkpoint['episode']
        self.best_test_score = checkpoint['best_test_score']
