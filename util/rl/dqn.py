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
import models.fft_utils
import rl_env
import util.rl.replay_buffer

from typing import Any, Dict, Optional


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


# noinspection PyProtectedMember
class DDQN(nn.Module):

    # TODO add code to freeze reconstructor for some number of steps
    def __init__(self, num_actions, device, memory, opts, env=None):
        # `env` != None indicates that the environment's reconstructor needs to be updated also
        super(DDQN, self).__init__()
        self.model = get_model(num_actions, opts)
        self.memory = memory
        self.optimizer = optim.Adam(self.parameters(), lr=opts.dqn_learning_rate)
        self.optimizer_reconstructor = None
        self.env = None
        if env is not None:
            self.env = env
            self.optimizer_reconstructor = optim.Adam(
                env._reconstructor.parameters(), lr=opts.reconstructor_lr)
        self.num_actions = num_actions
        self.opts = opts
        self.device = device

    def forward(self, x):
        return self.model(x)

    def _get_action_no_replacement(self, observation, eps_threshold):
        sample = random.random()
        # See comment in DQNTrainer._train_dqn_policy for observation format,
        # Index -2 recovers the mask associated to this observation
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
        self.model.eval()
        if self.opts.no_replacement_policy:
            return self._get_action_no_replacement(observation, eps_threshold)
        else:
            return self._get_action_standard_e_greedy(observation, eps_threshold)

    def add_experience(self, observation, action, next_observation, reward, done):
        self.memory.push(observation, action, next_observation, reward, done)

    def _update_dqn_parameters(self, dqn_batch, target_net):
        self.model.train()
        self.optimizer.zero_grad()
        observations = dqn_batch['observations'].to(self.device)
        next_observations = dqn_batch['next_observations'].to(self.device)
        actions = dqn_batch['actions'].to(self.device)
        rewards = dqn_batch['rewards'].to(self.device).squeeze()
        dones = dqn_batch['dones'].to(self.device)

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
                target_values[not_done_mask] = target_net.forward(next_observations)\
                    .gather(1, best_actions.unsqueeze(1))[not_done_mask].squeeze().detach()

            target_values = self.opts.gamma * target_values + rewards

        # loss = F.mse_loss(q_values, target_values.unsqueeze(1))
        loss = F.smooth_l1_loss(q_values, target_values.unsqueeze(1))

        loss.backward()

        # Compute total gradient norm (for logging purposes) and then clip gradients
        grad_norm = 0
        for p in list(filter(lambda p: p.grad is not None, self.parameters())):
            grad_norm += (p.grad.data.norm(2).item()**2)
        grad_norm = grad_norm**.5
        # grad_norm = torch.nn.utils.clip_grad_norm_(self.parameters(), 100)
        torch.nn.utils.clip_grad_value_(self.parameters(), 1)

        self.optimizer.step()

        return {
            'loss': loss,
            'grad_norm': grad_norm,
            'mean_q_values': q_values.detach().mean().cpu().numpy(),
            'std_q_values': q_values.detach().std().cpu().numpy()
        }

    def update_parameters(self, target_net) -> Optional[Dict[str, Any]]:
        batch = self.memory.sample()
        if batch is None:
            return None
        if self.opts.dqn_alternate_opt:
            assert self.env is not None
            assert self.opts.use_reconstructions
            dqn_batch = self._update_reconstructor_and_get_dqn_batch(batch)
        else:
            dqn_batch = {}
            for key, tensor in batch.items():
                dqn_batch[key] = tensor.to(self.device)
        return self._update_dqn_parameters(dqn_batch, target_net)

    def _update_reconstructor_and_get_dqn_batch(self, batch):
        assert self.env is not None and self.opts.dqn_alternate_opt
        # When using DQN alternate optimization, the replay buffer only stores the image index
        # and the mask used during the call to step. This function needs to retrieve the image
        # from the environment's training dataset, compute reconstructions with the current
        # reconstructor, update it, and create observations (i.e., reconstructions) for the DQN
        prev_image_idx_and_mask = batch['observations'].to(self.device)
        next_image_idx_and_mask = batch['next_observations'].to(self.device)

        images, masks = [], []
        for tensor in prev_image_idx_and_mask:
            tmp = self.env._dataset_train.__getitem__(tensor[0])
            masks.append(tensor[1:])
            images.append(tmp[1])
        for tensor in next_image_idx_and_mask:
            tmp = self.env._dataset_train.__getitem__(tensor[0])
            masks.append(tensor[1:])
            images.append(tmp[1])

        # Update for reconstructor network
        self.env._reconstructor.train()
        self.optimizer_reconstructor.zero_grad()
        batch_for_reconstructor = (torch.stack(masks, dim=0).view(len(masks), 1, 1, -1),
                                   torch.stack(images, dim=0))
        zero_filled_image, target, mask = models.fft_utils.preprocess_inputs(
            batch_for_reconstructor, self.env.options.dataroot, self.device)
        reconstructed_image, uncertainty_map, mask_embedding = self.env._reconstructor(
            zero_filled_image, mask)
        loss = models.fft_utils.gaussian_nll_loss(reconstructed_image, target, uncertainty_map,
                                                  self.env.options.dataroot).mean()
        loss.backward()
        self.optimizer_reconstructor.step()
        self.env._reconstructor.eval()

        # Now create a batch to update DQN model (obs, next_obs, rewards)
        all_scores = torch.tensor([
            self.env._compute_score(reconstructed_image[i].unsqueeze(0).detach(),
                                    target[i].unsqueeze(0).detach(), False)[self.opts.reward_metric]
            for i in range(len(reconstructed_image))
        ])
        batch_size = self.opts.rl_batch_size
        observations = self.env.pack_reconstruction_tensors_to_obs_format(
            reconstructed_image[:batch_size, :, :, :].detach(), mask[:batch_size, :, :, :].detach(),
            mask_embedding[:batch_size, :, :, :].detach())
        next_observations = self.env.pack_reconstruction_tensors_to_obs_format(
            reconstructed_image[batch_size:, :, :, :].detach(), mask[batch_size:, :, :, :].detach(),
            mask_embedding[batch_size:, :, :, :].detach())
        dqn_batch = {
            'observations': observations,
            'next_observations': next_observations,
            'actions': batch['actions'].to(self.device),
            'rewards': (all_scores[batch_size:] - all_scores[:batch_size]).to(self.device),
            'dones': batch['dones'].to(self.device)
        }

        return dqn_batch

    def init_episode(self):
        pass


# noinspection PyProtectedMember
class DQNTrainer:

    def __init__(self, options_, env=None, writer=None, logger=None, load_replay_mem=True):
        self.options = options_
        self.env = env
        self.steps = 0
        self.episode = 0
        self.best_test_score = -np.inf

        self.load_replay_mem = options_.dqn_resume and load_replay_mem

        if self.env is not None:
            self.env = env
            self.writer = writer
            self.logger = logger

            self.logger.info('Creating DDQN model.')

            # If replay will be loaded set capacity to zero (defer allocation to `__call__()`)
            mem_capacity = 0 if self.load_replay_mem or options_.dqn_only_test \
                else self._max_replay_buffer_size()
            self.logger.info(f'Creating replay buffer with capacity {mem_capacity}.')
            # For alternate optimization, obs_shape is just image index plus mask
            self.replay_memory = util.rl.replay_buffer.ReplayMemory(
                mem_capacity, (1 + self.env.image_width,)
                if self.options.dqn_alternate_opt else self.env.observation_space.shape,
                options_.rl_batch_size,
                options_.dqn_burn_in,
                use_normalization=options_.dqn_normalize)
            self.logger.info('Created replay buffer.')

            self.policy = DDQN(self.env.action_space.n, rl_env.device, self.replay_memory, options_,
                               self.env if options_.dqn_alternate_opt else None).to(rl_env.device)
            self.target_net = DDQN(self.env.action_space.n, rl_env.device, None, options_, None).to(
                rl_env.device)
            self.target_net.eval()
            self.logger.info(f'Created neural networks with {self.env.action_space.n} outputs.')

            self.window_size = min(self.options.num_train_images, 5000)
            self.reward_images_in_window = np.zeros(self.window_size)
            self.current_score_auc_window = np.zeros(self.window_size)

    def _max_replay_buffer_size(self):
        return min(self.options.num_train_steps, self.options.replay_buffer_size)

    def _init_all(self):
        # This is here so that it appears in SLURM stdout logs
        print(f'Checkpoint dir for this job is {self.options.checkpoints_dir}', flush=True)

        # Initialize writer and logger
        self.writer = tensorboardX.SummaryWriter(os.path.join(self.options.checkpoints_dir))
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.DEBUG)
        fh = logging.FileHandler(os.path.join(self.options.checkpoints_dir, 'train.log'))
        fh.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(threadName)s - %(levelname)s: %(message)s')
        fh.setFormatter(formatter)
        self.logger.addHandler(fh)

        # Initialize environment
        self.env = rl_env.ReconstructionEnv(self.options)
        self.options.mask_embedding_dim = self.env.metadata['mask_embed_dim']
        self.options.image_width = self.env.image_width
        self.env.set_training()
        self.logger.info(f'Created environment with {self.env.action_space.n} actions')

        # If replay will be loaded, set capacity to zero (defer allocation to `__call__()`)
        mem_capacity = 0 if self.load_replay_mem or self.options.dqn_only_test \
            else self._max_replay_buffer_size()
        # For alternate optimization, obs_shape is just image index plus mask
        self.replay_memory = util.rl.replay_buffer.ReplayMemory(
            mem_capacity, (1 + self.env.image_width,)
            if self.options.dqn_alternate_opt else self.env.observation_space.shape,
            self.options.rl_batch_size,
            self.options.dqn_burn_in,
            use_normalization=self.options.dqn_normalize)

        # Initialize policy
        self.policy = DDQN(self.env.action_space.n, rl_env.device, self.replay_memory, self.options,
                           self.env if self.options.dqn_alternate_opt else None).to(rl_env.device)
        self.target_net = DDQN(self.env.action_space.n, rl_env.device, None, self.options, None).to(
            rl_env.device)

        self.window_size = min(self.options.num_train_images, 5000)
        self.reward_images_in_window = np.zeros(self.window_size)
        self.current_score_auc_window = np.zeros(self.window_size)

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
                self.env._image_idx_train = self.episode + 1
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

                    # For alternate optimization, obs_shape is just image index plus mask
                    self.replay_memory = util.rl.replay_buffer.ReplayMemory(
                        mem_capacity, (1 + self.env.image_width,)
                        if self.options.dqn_alternate_opt else self.env.observation_space.shape,
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

            # Evaluate the current policy
            if self.episode % self.options.dqn_test_episode_freq == 0:
                test_score = acquire_rl.test_policy(
                    self.env,
                    self.policy,
                    self.writer,
                    self.logger,
                    self.episode,
                    self.options,
                    test_with_full_budget=True)
                if test_score > self.best_test_score:
                    policy_path = os.path.join(self.options.checkpoints_dir, 'policy_best.pt')
                    self.save(policy_path)
                    self.best_test_score = test_score
                    self.logger.info(
                        f'Saved DQN model with score {self.best_test_score} to {policy_path}.')

            # Evaluate the current policy on training set
            if self.episode % self.options.dqn_eval_train_set_episode_freq == 0 \
                    and self.options.num_train_images <= 5000:
                acquire_rl.test_policy(
                    self.env,
                    self.policy,
                    self.writer,
                    self.logger,
                    self.episode,
                    self.options,
                    test_on_train=True,
                    test_with_full_budget=True)

            # Run an episode and update model
            obs, info = self.env.reset()
            image_and_mask_obs = np.array([info['image_idx']] + list(obs[0, -2, :]))
            done = False
            total_reward = 0
            cnt_repeated_actions = 0
            auc_score = self.env.compute_score(
                True, use_current_score=True)[0][self.options.reward_metric]
            while not done:
                epsilon = get_epsilon(self.steps, self.options)
                action = self.policy.get_action(obs, epsilon, None)
                # Format of observation is [bs, img_height + 2, img_width], where first
                # img_height rows are reconstruction, next line is the mask, and final line is
                # the mask embedding
                is_repeated_action = bool(
                    obs[0, -2, action + self.options.initial_num_lines_per_side])

                assert not (self.options.no_replacement_policy and is_repeated_action)

                next_obs, reward, done, _ = self.env.step(action)
                # See comment above for format of obs and how to obtain mask from it
                image_and_mask_next_obs = np.array([info['image_idx']] + list(next_obs[0, -2, :]))
                self.steps += 1
                if self.options.dqn_alternate_opt:
                    self.policy.add_experience(image_and_mask_obs, action, image_and_mask_next_obs,
                                               -1, done)
                else:
                    self.policy.add_experience(obs, action, next_obs, reward, done)
                update_results = self.policy.update_parameters(self.target_net)

                if self.steps % self.options.target_net_update_freq == 0:
                    self.logger.info('Updating target network.')
                    self.target_net.load_state_dict(self.policy.state_dict())

                # Adding per-step tensorboard logs
                if self.steps % 50 == 0:
                    self.writer.add_scalar('epsilon', epsilon, self.steps)
                    if update_results is not None:
                        self.writer.add_scalar('loss', update_results['loss'], self.steps)
                        self.writer.add_scalar('grad_norm', update_results['grad_norm'], self.steps)
                        self.writer.add_scalar('mean_q_values', update_results['mean_q_values'],
                                               self.steps)
                        self.writer.add_scalar('std_q_values', update_results['std_q_values'],
                                               self.steps)

                total_reward += reward
                auc_score += self.env.compute_score(
                    True, use_current_score=True)[0][self.options.reward_metric]
                obs = next_obs
                image_and_mask_obs = image_and_mask_next_obs
                cnt_repeated_actions += int(is_repeated_action)

            # Adding per-episode tensorboard logs
            self.reward_images_in_window[self.episode % self.window_size] = total_reward
            self.current_score_auc_window[self.episode % self.window_size] = auc_score
            self.writer.add_scalar('episode_reward', total_reward, self.episode)
            self.writer.add_scalar(
                'average_reward_images_in_window',
                np.sum(self.reward_images_in_window) / min(self.episode + 1, self.window_size),
                self.episode)
            self.writer.add_scalar(
                'average_auc_score_in_window',
                np.sum(self.current_score_auc_window) / min(self.episode + 1, self.window_size),
                self.episode)

            self.episode += 1

            if self.episode % self.options.freq_dqn_checkpoint_save == 0:
                self.checkpoint(submit_job=False)

        self.checkpoint(submit_job=False)
        return self.best_test_score

    def __call__(self):
        if self.env is None:
            self._init_all()
        self.load_checkpoint_if_needed()
        if self.options.dqn_only_test:
            return None

        return self._train_dqn_policy()

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
        reconstructor = self.env._reconstructor.state_dict() if self.options.dqn_alternate_opt \
            else None
        torch.save({
            'dqn_weights': self.policy.state_dict(),
            'target_weights': self.target_net.state_dict(),
            'episode': self.episode,
            'steps': self.steps,
            'best_test_score': self.best_test_score,
            'reward_images_in_window': self.reward_images_in_window,
            'current_score_auc_window': self.current_score_auc_window,
            'reconstructor': reconstructor
        }, path)

    def load(self, path):
        checkpoint = torch.load(path)
        self.policy.load_state_dict(checkpoint['dqn_weights'])
        self.target_net.load_state_dict(checkpoint['target_weights'])
        self.steps = checkpoint['steps']
        self.episode = checkpoint['episode'] + 1
        self.best_test_score = checkpoint['best_test_score']
        self.reward_images_in_window = checkpoint['reward_images_in_window']
        self.current_score_auc_window = checkpoint['current_score_auc_window']
        if checkpoint['reconstructor'] is not None:
            self.env._reconstructor.load_state_dict(checkpoint['reconstructor'])
