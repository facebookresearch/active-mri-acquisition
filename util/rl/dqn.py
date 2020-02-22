import logging
import math
import os
import pickle
import random
import sys
import time

import filelock
import numpy as np
import submitit
import tensorboardX
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import models.evaluator
import rl_env
import util.rl.replay_buffer
import util.rl.utils


def get_epsilon(steps_done, opts):
    return opts.epsilon_end + (opts.epsilon_start - opts.epsilon_end) * \
        math.exp(-1. * steps_done / opts.epsilon_decay)


class Flatten(nn.Module):

    def forward(self, x):
        return x.view(x.size(0), -1)


class SimpleMLP(nn.Module):

    def __init__(self,
                 budget,
                 image_width,
                 initial_num_lines_per_side,
                 num_hidden_layers=2,
                 num_hidden=32,
                 ignore_mask=True,
                 symmetric=True):
        super(SimpleMLP, self).__init__()
        self.initial_num_lines_per_side = initial_num_lines_per_side
        self.ignore_mask = ignore_mask
        self.symmetric = symmetric
        self.num_inputs = budget if self.ignore_mask else self.image_width
        num_actions = image_width - 2 * initial_num_lines_per_side
        self.linear1 = nn.Sequential(nn.Linear(self.num_inputs, num_hidden), nn.ReLU())
        hidden_layers = []
        for i in range(num_hidden_layers):
            hidden_layers.append(nn.Sequential(nn.Linear(num_hidden, num_hidden), nn.ReLU()))
        self.hidden = nn.Sequential(*hidden_layers)
        self.output = nn.Linear(num_hidden, num_actions)
        self.model = nn.Sequential(self.linear1, self.hidden, self.output)

    def forward(self, x):
        previous_actions = x[:, self.initial_num_lines_per_side:-self.initial_num_lines_per_side]
        if self.ignore_mask:
            input_tensor = torch.zeros(x.shape[0], self.num_inputs).to(x.device)
            time_steps = previous_actions.sum(1).unsqueeze(1)
            if self.symmetric:
                time_steps //= 2
            # We allow the model to receive observations that are over budget during test
            # Code below randomizes the input to the model for these observations
            index_over_budget = (time_steps >= self.num_inputs).squeeze()
            time_steps = time_steps.clamp(0, self.num_inputs - 1)
            input_tensor.scatter_(1, time_steps.long(), 1)
            input_tensor[index_over_budget] = torch.randn_like(input_tensor[index_over_budget])
        else:
            input_tensor = x
        value = self.model(input_tensor)
        value = value - 1e10 * previous_actions
        return {'qvalue': value, 'value': None, 'advantage': None}


class EvaluatorBasedValueNetwork(nn.Module):

    def __init__(self, image_width, initial_num_lines_per_side, mask_embed_dim, use_dueling=False):
        super(EvaluatorBasedValueNetwork, self).__init__()
        num_actions = image_width - 2 * initial_num_lines_per_side
        self.evaluator = models.evaluator.EvaluatorNetwork(
            number_of_filters=128,
            number_of_conv_layers=4,
            use_sigmoid=False,
            width=image_width,
            mask_embed_dim=mask_embed_dim,
            num_output_channels=num_actions)
        self.mask_embed_dim = mask_embed_dim
        self.initial_num_lines_per_side = initial_num_lines_per_side
        self.use_dueling = use_dueling

        self.value_stream = None
        self.advantage_stream = None
        if self.use_dueling:
            self.value_stream = nn.Sequential(
                nn.Linear(num_actions, num_actions), nn.ReLU(), nn.Linear(num_actions, 1))
            self.advantage_stream = nn.Sequential(
                nn.Linear(num_actions, num_actions), nn.ReLU(), nn.Linear(num_actions, num_actions))

    def forward(self, x):
        reconstruction = x[..., :-2, :]
        bs = x.shape[0]
        if torch.isnan(x[0, 0, -1, 0]).item() == 1:
            assert self.mask_embed_dim == 0
            mask_embedding = None
            mask = None
        else:
            mask_embedding = x[:, 0, -1, :self.mask_embed_dim].view(bs, -1, 1, 1)
            mask_embedding = mask_embedding.repeat(1, 1, reconstruction.shape[2],
                                                   reconstruction.shape[3])
            mask = x[:, 0, -2, :]
            mask = mask.contiguous().view(bs, 1, 1, -1)

        value = None
        advantage = None
        if self.use_dueling:
            features = self.evaluator(reconstruction, mask_embedding, mask)
            value = self.value_stream(features)
            advantage = self.advantage_stream(features)
            qvalue = value + (advantage - advantage.mean(dim=1).unsqueeze(1))
        else:
            qvalue = self.evaluator(reconstruction, mask_embedding, mask)
        # This makes the DQN max over the target values consider only non repeated actions
        filtered_mask = x[:, 0, -2, self.initial_num_lines_per_side:
                          -self.initial_num_lines_per_side]
        qvalue = qvalue - 1e10 * filtered_mask
        return {'qvalue': qvalue, 'value': value, 'advantage': advantage}


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
        return {'qvalue': self.fc(torch.cat((image_encoding, rfft_encoding), dim=1))}


def get_model(num_actions, options):
    if options.dqn_model_type == 'simple_mlp':
        if options.use_dueling_dqn:
            raise NotImplementedError('Dueling DQN only implemented with dqn_model_type=evaluator.')
        return SimpleMLP(options.budget, options.image_width, options.initial_num_lines_per_side)
    if options.dqn_model_type == 'basic':
        if options.use_dueling_dqn:
            raise NotImplementedError('Dueling DQN only implemented with dqn_model_type=evaluator.')
        return BasicValueNetwork(num_actions)
    if options.dqn_model_type == 'evaluator':
        return EvaluatorBasedValueNetwork(
            options.image_width,
            options.initial_num_lines_per_side,
            options.mask_embedding_dim,
            use_dueling=options.use_dueling_dqn)
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
        if self.opts.obs_type == 'only_mask':
            previous_actions = np.nonzero(observation[self.opts.initial_num_lines_per_side:
                                                      -self.opts.initial_num_lines_per_side])[0]
        else:
            # See comment in DQNTrainer._train_dqn_policy for observation format,
            # Index -2 recovers the mask associated to this observation
            previous_actions = np.nonzero(observation[0, -2, self.opts.initial_num_lines_per_side:
                                                      -self.opts.initial_num_lines_per_side])[0]
        if sample < eps_threshold:
            return random.choice([x for x in range(self.num_actions) if x not in previous_actions])
        with torch.no_grad():
            q_values = self(torch.from_numpy(observation).unsqueeze(0).to(self.device))['qvalue']
        return torch.argmax(q_values, dim=1).item()

    def _get_action_standard_e_greedy(self, observation, eps_threshold):
        sample = random.random()
        if sample < eps_threshold:
            return random.randrange(self.num_actions)
        with torch.no_grad():
            q_values = self(torch.from_numpy(observation).unsqueeze(0).to(self.device))['qvalue']
        return torch.argmax(q_values, dim=1).item()

    def get_action(self, observation, eps_threshold, _):
        self.model.eval()
        if self.opts.no_replacement_policy:
            return self._get_action_no_replacement(observation, eps_threshold)
        else:
            return self._get_action_standard_e_greedy(observation, eps_threshold)

    def add_experience(self, observation, action, next_observation, reward, done):
        self.memory.push(observation, action, next_observation, reward, done)

    def update_parameters(self, target_net):
        self.model.train()
        batch = self.memory.sample()
        if batch is None:
            return None
        observations = batch['observations'].to(self.device)
        next_observations = batch['next_observations'].to(self.device)
        actions = batch['actions'].to(self.device)
        rewards = batch['rewards'].to(self.device).squeeze()
        dones = batch['dones'].to(self.device)

        not_done_mask = (1 - dones).squeeze()

        # Compute Q-values and get best action according to online network
        output_cur_step = self.forward(observations)
        all_q_values_cur = output_cur_step['qvalue']
        q_values = all_q_values_cur.gather(1, actions.unsqueeze(1))

        # Compute target values using the best action found
        with torch.no_grad():
            all_q_values_next = self.forward(next_observations)['qvalue']
            target_values = torch.zeros(observations.shape[0], device=self.device)
            if not_done_mask.any().item() != 0:
                best_actions = all_q_values_next.detach().max(1)[1]
                target_values[not_done_mask] = target_net.forward(next_observations)['qvalue'] \
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

        value = None
        advantage = None
        if output_cur_step['value'] is not None:
            value = output_cur_step['value'].detach().mean().cpu().numpy()
            advantage = output_cur_step['advantage'].detach().\
                gather(1, actions.unsqueeze(1)).mean().cpu().numpy()

        return {
            'loss': loss,
            'grad_norm': grad_norm,
            'q_values_mean': q_values.detach().mean().cpu().numpy(),
            'q_values_std': q_values.detach().std().cpu().numpy(),
            'value': value,
            'advantage': advantage
        }

    def init_episode(self):
        pass


class DQNTester:

    def __init__(self, training_dir):
        # self.options = options
        self.writer = None
        self.logger = None
        self.env = None
        self.policy = None

        self.training_dir = training_dir
        self.evaluation_dir = os.path.join(training_dir, 'evaluation')

        self.folder_lock = filelock.FileLock(
            DQNTrainer.get_lock_filename(self.training_dir), timeout=-1)

        self.latest_policy_path = DQNTrainer.get_name_latest_checkpont(self.training_dir)
        self.best_test_score = -np.inf
        self.last_time_stamp = -np.inf

        self.options = None

    def init_all(self):
        # Initialize writer and logger
        self.writer = tensorboardX.SummaryWriter(os.path.join(self.evaluation_dir))
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(threadName)s - %(levelname)s: %(message)s')
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(formatter)
        self.logger.addHandler(ch)
        ch.setLevel(logging.DEBUG)
        fh = logging.FileHandler(os.path.join(self.evaluation_dir, 'evaluation.log'))
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        self.logger.addHandler(fh)

        # Read the options used for training
        while True:
            with self.folder_lock:
                options_filename = DQNTrainer.get_options_filename(self.training_dir)
                if os.path.isfile(options_filename):
                    self.logger.info(f'Options file found at {options_filename}.')
                    with open(options_filename, 'rb') as f:
                        self.options = pickle.load(f)
                    break
                else:
                    self.logger.info(f'No options file found at {options_filename}. '
                                     f'I will wait for five minutes before trying again.')
                    time.sleep(300)
        # This change is needed so that util.test_policy writes results to correct directory
        self.options.checkpoints_dir = self.evaluation_dir
        os.makedirs(self.evaluation_dir, exist_ok=True)

        # Initialize environment
        self.env = rl_env.ReconstructionEnv(self.options)
        self.options.mask_embedding_dim = self.env.metadata['mask_embed_dim']
        self.options.image_width = self.env.image_width
        self.logger.info(f'Created environment with {self.env.action_space.n} actions')

        # This is here so that it appears in SLURM stdout logs
        self.logger.info(f'Checkpoint dir for this job is {self.evaluation_dir}')
        self.logger.info(f'Evaluation will be done for model saved at {self.training_dir}')

        # Initialize policy
        self.policy = DDQN(self.env.action_space.n, rl_env.device, None, self.options).to(
            rl_env.device)

        # Load info about best checkpoint tested and timestamp
        self.load_tester_checkpoint_if_present()

    def __call__(self):
        self.init_all()
        training_done = False
        while not training_done:
            training_done = self.check_if_train_done()
            self.logger.info(f'Is training done? {training_done}.')
            checkpoint_episode, timestamp = self.load_latest_policy()

            if timestamp is None or timestamp <= self.last_time_stamp:
                # No new policy checkpoint to evaluate
                self.logger.info('No new policy to evaluate. '
                                 'I will wait for 10 minutes before trying again.')
                time.sleep(600)
                continue

            self.logger.info(f'Found a new checkpoint with timestamp {timestamp}, '
                             f'will start evaluation now.')
            test_score, _ = util.rl.utils.test_policy(self.env, self.policy, self.writer,
                                                      self.logger, checkpoint_episode, self.options)
            self.logger.info(f'The test score for the model was {test_score}.')
            self.last_time_stamp = timestamp
            if test_score > self.best_test_score:
                self.save_tester_checkpoint()
                policy_path = os.path.join(self.evaluation_dir, 'policy_best.pt')
                self.save_policy(policy_path, checkpoint_episode)
                self.best_test_score = test_score
                self.logger.info(
                    f'Saved DQN model with score {self.best_test_score} to {policy_path}, '
                    f'corresponding to episode {checkpoint_episode}.')

    def check_if_train_done(self):
        with self.folder_lock:
            return os.path.isfile(DQNTrainer.get_done_filename(self.training_dir))

    def checkpoint(self):
        self.logger.info('Received preemption signal.')
        self.save_tester_checkpoint()
        return submitit.helpers.DelayedSubmission(DQNTester(self.options))

    def save_tester_checkpoint(self):
        path = os.path.join(self.evaluation_dir, 'tester_checkpoint.pickle')
        with open(path, 'wb') as f:
            pickle.dump({
                'best_test_score': self.best_test_score,
                'last_time_stamp': self.last_time_stamp
            }, f)

    def load_tester_checkpoint_if_present(self):
        path = os.path.join(self.evaluation_dir, 'tester_checkpoint.pickle')
        if os.path.isfile(path):
            with open(path, 'rb') as f:
                checkpoint = pickle.load(f)
                self.best_test_score = checkpoint['best_test_score']
                self.last_time_stamp = checkpoint['last_time_stamp']
                self.logger.info(f'Found checkpoint from previous evaluation run. '
                                 f'Best Score set to {self.best_test_score}. '
                                 f'Last Time Stamp set to {self.last_time_stamp}')

    # noinspection PyProtectedMember
    def load_latest_policy(self):
        """ Loads the latest checkpoint and returns the training episode at which it was saved. """
        with self.folder_lock:
            if not os.path.isfile(self.latest_policy_path):
                return None, None
            timestamp = os.path.getmtime(self.latest_policy_path)
            checkpoint = torch.load(self.latest_policy_path, map_location=rl_env.device)
        self.policy.load_state_dict(checkpoint['dqn_weights'])
        return checkpoint['episode'], timestamp

    def save_policy(self, path, episode):
        torch.save({
            'dqn_weights': self.policy.state_dict(),
            'episode': episode,
        }, path)


class DQNTrainer:

    def __init__(self, options, env=None, writer=None, logger=None, load_replay_mem=True):
        self.options = options
        self.env = env
        self.steps = 0
        self.episode = 0
        self.best_test_score = -np.inf

        self.load_replay_mem = options.dqn_resume and load_replay_mem

        if self.env is not None:
            self.env = env
            self.writer = writer
            self.logger = logger

            self.logger.info('Creating DDQN model.')

            # If replay will be loaded set capacity to zero (defer allocation to `__call__()`)
            mem_capacity = 0 if self.load_replay_mem or options.dqn_only_test \
                else self._max_replay_buffer_size()
            self.logger.info(f'Creating replay buffer with capacity {mem_capacity}.')
            self.replay_memory = util.rl.replay_buffer.ReplayMemory(
                mem_capacity,
                self.env.observation_space.shape,
                options.rl_batch_size,
                options.dqn_burn_in,
                use_normalization=options.dqn_normalize)
            self.logger.info('Created replay buffer.')

            if torch.cuda.is_available() and torch.cuda.device_count() > 1:
                device = torch.device(f'cuda:{torch.cuda.device_count() - 1}')
            else:
                device = rl_env.device
            self.policy = DDQN(self.env.action_space.n, device, self.replay_memory,
                               options).to(device)
            self.target_net = DDQN(self.env.action_space.n, device, None, options).to(device)
            self.target_net.eval()
            self.logger.info(f'Created neural networks with {self.env.action_space.n} outputs.')

            self.window_size = min(self.options.num_train_images, 5000)
            self.reward_images_in_window = np.zeros(self.window_size)
            self.current_score_auc_window = np.zeros(self.window_size)

            if self.options.dqn_alternate_opt_per_epoch:
                self.env.set_epoch_finished_callback(self.update_reconstructor_and_buffer,
                                                     self.options.frequency_train_reconstructor)

            self.folder_lock = filelock.FileLock(
                DQNTrainer.get_lock_filename(self.options.checkpoints_dir), timeout=-1)

            with self.folder_lock:
                with open(DQNTrainer.get_options_filename(self.options.checkpoints_dir), 'wb') as f:
                    pickle.dump(self.options, f)

    @staticmethod
    def get_done_filename(path):
        return os.path.join(path, 'DONE')

    @staticmethod
    def get_lock_filename(path):
        return os.path.join(path, '.LOCK')

    @staticmethod
    def get_options_filename(path):
        return os.path.join(path, 'options.pickle')

    @staticmethod
    def get_name_latest_checkpont(path):
        return os.path.join(path, 'policy_checkpoint.pth')

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

        # Logging information about the options used
        self.logger.info('Creating RL acquisition run with the following options:')
        for key, value in vars(self.options).items():
            if key == 'device':
                value = value.type
            elif key == 'gpu_ids':
                value = 'cuda : ' + str(value) if torch.cuda.is_available() else 'cpu'
            self.logger.info(f"    {key:>25}: {'None' if value is None else value:<30}")

        # Initialize environment
        self.env = rl_env.ReconstructionEnv(self.options)
        self.options.mask_embedding_dim = self.env.metadata['mask_embed_dim']
        self.options.image_width = self.env.image_width
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
        if torch.cuda.is_available() and torch.cuda.device_count() > 1:
            device = torch.device(f'cuda:{torch.cuda.device_count() - 1}')
        else:
            device = rl_env.device

        self.policy = DDQN(self.env.action_space.n, device, self.replay_memory,
                           self.options).to(device)
        self.target_net = DDQN(self.env.action_space.n, device, None, self.options).to(device)

        self.window_size = min(self.options.num_train_images, 5000)
        self.reward_images_in_window = np.zeros(self.window_size)
        self.current_score_auc_window = np.zeros(self.window_size)

        if self.options.dqn_alternate_opt_per_epoch:
            self.env.set_epoch_finished_callback(self.update_reconstructor_and_buffer,
                                                 self.options.frequency_train_reconstructor)

        self.folder_lock = filelock.FileLock(
            DQNTrainer.get_lock_filename(self.options.checkpoints_dir), timeout=-1)

        with self.folder_lock:
            with open(DQNTrainer.get_options_filename(self.options.checkpoints_dir), 'wb') as f:
                pickle.dump(self.options, f)

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
            policy_path = DQNTrainer.get_name_latest_checkpont(self.options.checkpoints_dir)
            self.logger.info(f'Checking for DQN policy at {policy_path}.')
            if os.path.isfile(policy_path):
                self.load(policy_path)
                self.logger.info(f'Loaded DQN policy found at {policy_path}. Steps was set to '
                                 f'{self.steps}. Episodes set to {self.episode}.')
                self.env._image_idx_train = self.episode % self.env.num_train_images
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
        steps_epsilon = self.steps
        while self.steps < self.options.num_train_steps:
            self.logger.info('Episode {}'.format(self.episode + 1))

            # Evaluate the current policy
            if self.options.dqn_test_episode_freq is not None and (
                    self.episode % self.options.dqn_test_episode_freq == 0):
                test_score, _ = util.rl.utils.test_policy(self.env, self.policy, self.writer,
                                                          self.logger, self.episode, self.options)
                if test_score > self.best_test_score:
                    policy_path = os.path.join(self.options.checkpoints_dir, 'policy_best.pt')
                    self.save(policy_path)
                    self.best_test_score = test_score
                    self.logger.info(
                        f'Saved DQN model with score {self.best_test_score} to {policy_path}.')

            # Evaluate the current policy on training set
            if self.options.dqn_eval_train_set_episode_freq is not None and (
                    self.episode % self.options.dqn_eval_train_set_episode_freq == 0) and (
                        self.options.num_train_images <= 1000):
                util.rl.utils.test_policy(
                    self.env,
                    self.policy,
                    self.writer,
                    self.logger,
                    self.episode,
                    self.options,
                    test_on_train=True)

            # Run an episode and update model
            obs, _ = self.env.reset(
                start_with_initial_mask=self.options.train_with_fixed_initial_mask)
            done = False
            total_reward = 0
            cnt_repeated_actions = 0
            auc_score = self.env.compute_score(
                True, use_current_score=True)[0][self.options.reward_metric]
            while not done:
                epsilon = get_epsilon(steps_epsilon, self.options)
                action = self.policy.get_action(obs, epsilon, None)
                if self.options.obs_type == 'only_mask':
                    is_repeated_action = bool(obs[action + self.options.initial_num_lines_per_side])
                else:
                    # Format of observation is [bs, img_height + 2, img_width], where first
                    # img_height rows are reconstruction, next line is the mask, and final line is
                    # the mask embedding
                    is_repeated_action = bool(
                        obs[0, -2, action + self.options.initial_num_lines_per_side])

                assert not (self.options.no_replacement_policy and is_repeated_action)

                next_obs, reward, done, _ = self.env.step(action)
                self.steps += 1
                self.policy.add_experience(obs, action, next_obs, reward, done)

                update_results = None
                if self.steps >= self.options.num_steps_with_fixed_dqn_params:
                    if self.steps == self.options.num_steps_with_fixed_dqn_params:
                        self.logger.info(f'Started updating DQN weights at step {self.steps}')
                    update_results = self.policy.update_parameters(self.target_net)

                    if self.steps % self.options.target_net_update_freq == 0:
                        self.logger.info('Updating target network.')
                        self.target_net.load_state_dict(self.policy.state_dict())
                    steps_epsilon += 1

                # Adding per-step tensorboard logs
                if self.steps % 250 == 0:
                    self.writer.add_scalar('epsilon', epsilon, self.steps)
                    if update_results is not None:
                        self.writer.add_scalar('loss', update_results['loss'], self.steps)
                        self.writer.add_scalar('grad_norm', update_results['grad_norm'], self.steps)
                        self.writer.add_scalar('mean_q_value', update_results['q_values_mean'],
                                               self.steps)
                        self.writer.add_scalar('std_q_value', update_results['q_values_std'],
                                               self.steps)
                        if update_results['value'] is not None:
                            self.writer.add_scalar('value', update_results['value'], self.steps)
                            self.writer.add_scalar('advantage', update_results['advantage'],
                                                   self.steps)

                total_reward += reward
                auc_score += self.env.compute_score(
                    True, use_current_score=True)[0][self.options.reward_metric]
                obs = next_obs

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
                self.checkpoint(submit_job=False, save_memory=False)

        self.checkpoint(submit_job=False)

        with self.folder_lock:
            with open(DQNTrainer.get_done_filename(self.options.checkpoints_dir), 'w') as f:
                f.write(str(self.best_test_score))
        return self.best_test_score

    def update_reconstructor_and_buffer(self):
        self.env.retrain_reconstructor(self.logger, self.writer)
        self.replay_memory.count_seen = 0
        self.replay_memory.position = 0
        self.checkpoint(save_memory=False, submit_job=False)

    def __call__(self):
        if self.env is None:
            self._init_all()
        self.load_checkpoint_if_needed()
        if self.options.dqn_only_test:
            return None

        return self._train_dqn_policy()

    def checkpoint(self, submit_job=True, save_memory=True):
        self.logger.info('Received preemption signal.')
        policy_path = DQNTrainer.get_name_latest_checkpont(self.options.checkpoints_dir)
        self.save(policy_path)
        self.logger.info(f'Saved DQN checkpoint to {policy_path}')
        if save_memory:
            self.logger.info('Now saving replay memory.')
            memory_path = self.replay_memory.save(self.options.checkpoints_dir, 'replay_buffer.pt')
            self.logger.info(f'Saved replay buffer to {memory_path}.')
        if submit_job:
            trainer = DQNTrainer(self.options, load_replay_mem=True)
            return submitit.helpers.DelayedSubmission(trainer)

    def save(self, path):
        with self.folder_lock:
            torch.save({
                'dqn_weights': self.policy.state_dict(),
                'target_weights': self.target_net.state_dict(),
                'episode': self.episode,
                'steps': self.steps,
                'best_test_score': self.best_test_score,
                'reward_images_in_window': self.reward_images_in_window,
                'current_score_auc_window': self.current_score_auc_window
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
