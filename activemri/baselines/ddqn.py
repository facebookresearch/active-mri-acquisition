"""
activemri.baselines.ddqn.py
=======================================
Baseline implementation of Double DQN, as described in
*Van Hasselt, Hado, Arthur Guez, and David Silver. "Deep reinforcement learning with
double q-learning." arXiv preprint arXiv:1509.06461 (2015)*.
"""
import logging
import math
import os
import pickle
import random
import sys
import time
import types
from typing import Any, Dict, List, Optional, Tuple

import filelock
import numpy as np
import tensorboardX
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import activemri.envs.envs as mri_envs
import activemri.experimental.cvpr19_models.models.evaluator as cvpr19_evaluator

from . import Policy, RandomPolicy, evaluation, replay_buffer


def _encode_obs_dict(obs: Dict[str, Any]) -> torch.Tensor:
    reconstruction = obs["reconstruction"].permute(0, 3, 1, 2)
    mask_embedding = obs["extra_outputs"]["mask_embedding"]
    mask = obs["mask"]

    batch_size, num_channels, img_height, img_width = reconstruction.shape
    transformed_obs = torch.zeros(
        batch_size, num_channels, img_height + 2, img_width
    ).float()
    transformed_obs[..., :img_height, :] = reconstruction
    # The second to last row is the mask
    transformed_obs[..., img_height, :] = mask.unsqueeze(1)
    # The last row is the mask embedding (padded with 0s if necessary)
    if mask_embedding:
        mask_embed_dim = len(mask_embedding[0])
        transformed_obs[..., img_height + 1, :mask_embed_dim] = mask_embedding[
            :, :, 0, 0
        ]
    else:
        transformed_obs[:, :, img_height + 1, 0] = np.nan
    return transformed_obs


def _decode_obs_tensor(
    obs_tensor: torch.Tensor, mask_embed_dim: int
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    reconstruction = obs_tensor[..., :-2, :]
    bs = obs_tensor.shape[0]
    if torch.isnan(obs_tensor[0, 0, -1, 0]).item() == 1:
        assert mask_embed_dim == 0
        mask_embedding = None
    else:
        mask_embedding = obs_tensor[:, 0, -1, :mask_embed_dim].view(bs, -1, 1, 1)
        mask_embedding = mask_embedding.repeat(
            1, 1, reconstruction.shape[2], reconstruction.shape[3]
        )

    mask = obs_tensor[:, 0, -2, :]
    mask = mask.contiguous().view(bs, 1, 1, -1)

    return reconstruction, mask, mask_embedding


def _get_epsilon(steps_done, opts):
    return opts.epsilon_end + (opts.epsilon_start - opts.epsilon_end) * math.exp(
        -1.0 * steps_done / opts.epsilon_decay
    )


# noinspection PyAbstractClass
class Flatten(nn.Module):
    # noinspection PyMethodMayBeStatic
    def forward(self, x):
        return x.view(x.size(0), -1)


# TODO remove legacy offset if after test it looks like it's no longer necessary
class SimpleMLP(nn.Module):
    def __init__(
        self,
        budget: int,
        image_width: int,
        num_hidden_layers: int = 2,
        hidden_size: int = 32,
        ignore_mask: bool = True,
        legacy_offset: Optional[int] = None,
        ignored_cols: Optional[List[int]] = None,
    ):
        super(SimpleMLP, self).__init__()
        if ignored_cols is None:
            ignored_cols = []
        self.ignore_mask = ignore_mask
        self.num_inputs = budget if self.ignore_mask else image_width
        num_actions = image_width
        if legacy_offset:
            num_actions -= 2 * legacy_offset
        self.legacy_offset = legacy_offset
        self.ignored_cols = ignored_cols
        self.linear1 = nn.Sequential(nn.Linear(self.num_inputs, hidden_size), nn.ReLU())
        hidden_layers = []
        for i in range(num_hidden_layers):
            hidden_layers.append(
                nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.ReLU())
            )
        self.hidden = nn.Sequential(*hidden_layers)
        self.output = nn.Linear(hidden_size, num_actions)
        self.model = nn.Sequential(self.linear1, self.hidden, self.output)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, mask, _ = _decode_obs_tensor(x, 0)
        previous_actions = mask.squeeze()
        if self.ignored_cols:
            previous_actions[..., self.ignored_cols] = 0
        if self.ignore_mask:
            input_tensor = torch.zeros(x.shape[0], self.num_inputs).to(x.device)
            time_steps = previous_actions.sum(1).unsqueeze(1)
            # We allow the model to receive observations that are over budget during test
            # Code below randomizes the input to the model for these observations
            index_over_budget = (time_steps >= self.num_inputs).squeeze()
            time_steps = time_steps.clamp(0, self.num_inputs - 1)
            input_tensor.scatter_(1, time_steps.long(), 1)
            input_tensor[index_over_budget] = torch.randn_like(
                input_tensor[index_over_budget]
            )
        else:
            input_tensor = mask
        value = self.model(input_tensor)
        if self.legacy_offset:
            previous_actions = previous_actions[
                ..., self.legacy_offset : -self.legacy_offset
            ]
        return value - 1e10 * previous_actions


class EvaluatorBasedValueNetwork(nn.Module):
    """ Value network based on Zhang et al., CVPR'19 evaluator architecture. """

    def __init__(
        self, image_width: int, mask_embed_dim: int, legacy_offset: Optional[int] = None
    ):
        super().__init__()
        num_actions = image_width
        if legacy_offset:
            num_actions -= 2 * legacy_offset
        self.legacy_offset = legacy_offset
        self.evaluator = cvpr19_evaluator.EvaluatorNetwork(
            number_of_filters=128,
            number_of_conv_layers=4,
            use_sigmoid=False,
            width=image_width,
            mask_embed_dim=mask_embed_dim,
            num_output_channels=num_actions,
        )
        self.mask_embed_dim = mask_embed_dim

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Predicts action values.

        Args:
            obs(torch.Tensor): The observation tensor.

        Returns:
            torch.Tensor: Q-values for all actions at the given observation.

        Note:
            Values corresponding to active k-space columns in the observation are manually
            set to ``1e-10``.
        """
        reconstruction, mask, mask_embedding = _decode_obs_tensor(
            obs, self.evaluator.mask_embed_dim
        )
        qvalue = self.evaluator(reconstruction, mask_embedding)
        if self.legacy_offset:
            mask = mask[..., self.legacy_offset : -self.legacy_offset]
        return qvalue - 1e10 * mask.squeeze()


def _get_model(options):
    if options.dqn_model_type == "simple_mlp":
        return SimpleMLP(
            options.budget,
            options.image_width,
            legacy_offset=getattr(options, "legacy_offset", None),
            ignored_cols=getattr(options, "ignored_cols", None),
        )
    if options.dqn_model_type == "evaluator":
        return EvaluatorBasedValueNetwork(
            options.image_width,
            options.mask_embedding_dim,
            legacy_offset=getattr(options, "legacy_offset", None),
        )
    raise ValueError("Unknown model specified for DQN.")


class DDQN(nn.Module, Policy):
    """Implementation of Double DQN value network.

    The configuration is given by the ``opts`` argument, which must contain the following
    fields:

        - mask_embedding_dim(int): See
          :class:`cvpr19_models.models.evaluator.EvaluatorNetwork`.
        - gamma(float): Discount factor for target updates.
        - dqn_model_type(str): Describes the architecture of the neural net. Options
                are "simple_mlp" and "evaluator", to use :class:`SimpleMLP` and
                :class:`EvaluatorBasedValueNetwork`, respectively.
        - budget(int): The environment's budget.
        - image_width(int): The width of the input images.

    Args:
        device(``torch.device``): Device to use.
        memory(optional(``replay_buffer.ReplayMemory``)): Replay buffer to sample transitions
            from. Can be ``None``, for example, if this is a target network.
        opts(``types.SimpleNamespace``): Options for the algorithm as explained above.
    """

    def __init__(
        self,
        device: torch.device,
        memory: Optional[replay_buffer.ReplayMemory],
        opts: types.SimpleNamespace,
    ):
        super().__init__()
        self.model = _get_model(opts)
        self.memory = memory
        self.optimizer = optim.Adam(self.parameters(), lr=opts.dqn_learning_rate)
        self.opts = opts
        self.device = device
        self.random_sampler = RandomPolicy()
        self.to(device)

    def add_experience(
        self,
        observation: np.array,
        action: int,
        next_observation: np.array,
        reward: float,
        done: bool,
    ):
        self.memory.push(observation, action, next_observation, reward, done)

    def update_parameters(self, target_net: nn.Module) -> Optional[Dict[str, Any]]:
        self.model.train()
        batch = self.memory.sample()
        if batch is None:
            return None
        observations = batch["observations"].to(self.device)
        next_observations = batch["next_observations"].to(self.device)
        actions = batch["actions"].to(self.device)
        rewards = batch["rewards"].to(self.device).squeeze()
        dones = batch["dones"].to(self.device)

        not_done_mask = dones.logical_not().squeeze()

        # Compute Q-values and get best action according to online network
        output_cur_step = self.forward(observations)
        all_q_values_cur = output_cur_step
        q_values = all_q_values_cur.gather(1, actions.unsqueeze(1))

        # Compute target values using the best action found
        if self.opts.gamma == 0.0:
            target_values = rewards
        else:
            with torch.no_grad():
                all_q_values_next = self.forward(next_observations)
                target_values = torch.zeros(observations.shape[0], device=self.device)
                del observations
                if not_done_mask.any().item() != 0:
                    best_actions = all_q_values_next.detach().max(1)[1]
                    target_values[not_done_mask] = (
                        target_net.forward(next_observations)
                        .gather(1, best_actions.unsqueeze(1))[not_done_mask]
                        .squeeze()
                        .detach()
                    )

                target_values = self.opts.gamma * target_values + rewards

        # loss = F.mse_loss(q_values, target_values.unsqueeze(1))
        loss = F.smooth_l1_loss(q_values, target_values.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()

        # Compute total gradient norm (for logging purposes) and then clip gradients
        # TODO initialize with some tensor here (mypy error)
        grad_norm: torch.Tensor = 0  # type: ignore
        for p in list(filter(lambda p: p.grad is not None, self.parameters())):
            grad_norm += p.grad.data.norm(2).item() ** 2
        grad_norm = grad_norm ** 0.5
        torch.nn.utils.clip_grad_value_(self.parameters(), 1)

        self.optimizer.step()

        torch.cuda.empty_cache()

        return {
            "loss": loss,
            "grad_norm": grad_norm,
            "q_values_mean": q_values.detach().mean().cpu().numpy(),
            "q_values_std": q_values.detach().std().cpu().numpy(),
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predicts action values.

        Args:
            x(torch.Tensor): The observation tensor.

        Returns:
            Dictionary(torch.Tensor): The predicted Q-values.

        Note:
            Values corresponding to active k-space columns in the observation are manually
            set to ``1e-10``.
        """
        return self.model(x)

    def get_action(  # type: ignore
        self, obs: Dict[str, Any], eps_threshold: float = 0.0
    ) -> List[int]:
        """Returns an action sampled from an epsilon-greedy policy.

        With probability epsilon sample a random k-space column (ignoring active columns),
        otherwise return the column with the highest estimated Q-value for the observation.

        Args:
            obs(torch.Tensor): The observation for which an action is required.
            eps_threshold(float): The probability of sampling a random action instead of using
                a greedy action.
        """
        sample = random.random()
        if sample < eps_threshold:
            return self.random_sampler.get_action(obs)
        with torch.no_grad():
            self.model.eval()
            obs_tensor = _encode_obs_dict(obs)
            q_values = self(obs_tensor.to(self.device))
        actions = torch.argmax(q_values, dim=1) + getattr(self.opts, "legacy_offset", 0)
        return actions.tolist()


def _get_folder_lock(path):
    return filelock.FileLock(path, timeout=-1)


class DDQNTester:
    def __init__(
        self, env: mri_envs.ActiveMRIEnv, training_dir: str, device: torch.device
    ):
        self.env = env
        self.device = device

        self.training_dir = training_dir
        self.evaluation_dir = os.path.join(training_dir, "evaluation")
        os.makedirs(self.evaluation_dir, exist_ok=True)

        self.folder_lock_path = DDQNTrainer.get_lock_filename(training_dir)

        self.latest_policy_path = DDQNTrainer.get_name_latest_checkpoint(
            self.training_dir
        )
        self.best_test_score = -np.inf
        self.last_time_stamp = -np.inf

        self.options = None

        # Initialize writer and logger
        self.writer = tensorboardX.SummaryWriter(os.path.join(self.evaluation_dir))
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.INFO)
        formatter = logging.Formatter(
            "%(asctime)s - %(threadName)s - %(levelname)s: %(message)s"
        )
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(formatter)
        self.logger.addHandler(ch)
        ch.setLevel(logging.DEBUG)
        fh = logging.FileHandler(os.path.join(self.evaluation_dir, "evaluation.log"))
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        self.logger.addHandler(fh)

        # Read the options used for training
        options_file_found = False
        while not options_file_found:
            options_filename = DDQNTrainer.get_options_filename(self.training_dir)
            with _get_folder_lock(self.folder_lock_path):
                if os.path.isfile(options_filename):
                    self.logger.info(f"Options file found at {options_filename}.")
                    with open(options_filename, "rb") as f:
                        self.options = pickle.load(f)
                    options_file_found = True
            if not options_file_found:
                self.logger.info(f"No options file found at {options_filename}.")
                self.logger.info("I will wait for five minutes before trying again.")
                time.sleep(300)
        # This change is needed so that util.test_policy writes results to correct directory
        self.options.checkpoints_dir = self.evaluation_dir
        os.makedirs(self.evaluation_dir, exist_ok=True)

        # Initialize environment
        self.options.image_width = self.env.action_space.n
        self.logger.info(f"Created environment with {self.env.action_space.n} actions")

        self.logger.info(f"Checkpoint dir for this job is {self.evaluation_dir}")
        self.logger.info(
            f"Evaluation will be done for model saved at {self.training_dir}"
        )

        # Initialize policy
        self.policy = DDQN(device, None, self.options)

        # Load info about best checkpoint tested and timestamp
        self.load_tester_checkpoint_if_present()

    def __call__(self):
        training_done = False
        while not training_done:
            training_done = self.check_if_train_done()
            self.logger.info(f"Is training done? {training_done}.")
            checkpoint_episode, timestamp = self.load_latest_policy()

            if timestamp is None or timestamp <= self.last_time_stamp:
                # No new policy checkpoint to evaluate
                self.logger.info(
                    "No new policy to evaluate. "
                    "I will wait for 10 minutes before trying again."
                )
                time.sleep(600)
                continue

            self.logger.info(
                f"Found a new checkpoint with timestamp {timestamp}, "
                f"I will start evaluation now."
            )
            test_scores, _ = evaluation.evaluate(
                self.env,
                self.policy,
                self.options.num_test_episodes,
                self.options.seed,
                "val",
                verbose=True,
            )
            auc_score = test_scores[self.options.reward_metric].sum(axis=1).mean()
            if "mse" in self.options.reward_metric:
                auc_score *= -1
            self.logger.info(f"The test score for the model was {auc_score}.")
            self.last_time_stamp = timestamp
            if auc_score > self.best_test_score:
                self.save_tester_checkpoint()
                policy_path = os.path.join(self.evaluation_dir, "policy_best.pt")
                self.save_policy(policy_path, checkpoint_episode)
                self.best_test_score = auc_score
                self.logger.info(
                    f"Saved DQN model with score {self.best_test_score} to {policy_path}, "
                    f"corresponding to episode {checkpoint_episode}."
                )

    def check_if_train_done(self):
        with _get_folder_lock(self.folder_lock_path):
            return os.path.isfile(DDQNTrainer.get_done_filename(self.training_dir))

    def checkpoint(self):
        self.save_tester_checkpoint()

    def save_tester_checkpoint(self):
        path = os.path.join(self.evaluation_dir, "tester_checkpoint.pickle")
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "best_test_score": self.best_test_score,
                    "last_time_stamp": self.last_time_stamp,
                },
                f,
            )

    def load_tester_checkpoint_if_present(self):
        path = os.path.join(self.evaluation_dir, "tester_checkpoint.pickle")
        if os.path.isfile(path):
            with open(path, "rb") as f:
                checkpoint = pickle.load(f)
            self.best_test_score = checkpoint["best_test_score"]
            self.last_time_stamp = checkpoint["last_time_stamp"]
            self.logger.info(
                f"Found checkpoint from previous evaluation run. "
                f"Best Score set to {self.best_test_score}. "
                f"Last Time Stamp set to {self.last_time_stamp}"
            )

    # noinspection PyProtectedMember
    def load_latest_policy(self):
        with _get_folder_lock(self.folder_lock_path):
            if not os.path.isfile(self.latest_policy_path):
                return None, None
            timestamp = os.path.getmtime(self.latest_policy_path)
            checkpoint = torch.load(self.latest_policy_path, map_location=self.device)
        self.policy.load_state_dict(checkpoint["dqn_weights"])
        return checkpoint["episode"], timestamp

    def save_policy(self, path, episode):
        torch.save({"dqn_weights": self.policy.state_dict(), "episode": episode}, path)


class DDQNTrainer:
    """DDQN Trainer for active MRI acquisition.

    Configuration for the trainer is provided by argument ``options``. Must contain the
    following fields:

        - "checkpoints_dir"(str): The directory where the model will be saved to (or
          loaded from).
        - dqn_batch_size(int): The batch size to use for updates.
        - dqn_burn_in(int): How many steps to do before starting updating parameters.
        - dqn_normalize(bool): ``True`` if running mean/st. deviation should be maintained
          for observations.
        - dqn_only_test(bool): ``True`` if the model will not be trained, thus only will
          attempt to read from checkpoint and load only weights of the network (ignoring
          training related information).
        - dqn_test_episode_freq(optional(int)): How frequently (in number of env steps)
          to perform test episodes.
        - freq_dqn_checkpoint_save(int): How often (in episodes) to save the model.
        - num_train_steps(int): How many environment steps to train for.
        - replay_buffer_size(int): The capacity of the replay buffer.
        - resume(bool): If true, will try to load weights from the checkpoints dir.
        - num_test_episodes(int): How many test episodes to periodically evaluate for.
        - seed(int): Sets the seed for the environment when running evaluation episodes.
        - reward_metric(str): Which of the ``env.scores_keys()`` is used as reward. Mainly
          used for logging purposes.
        - target_net_update_freq(int): How often (in env's steps) to update the target
          network.

    Args:
        options(``types.SimpleNamespace``): Options for the trainer.
        env(``activemri.envs.ActiveMRIEnv``): Env for which the policy is trained.
        device(``torch.device``): Device to use.
    """

    def __init__(
        self,
        options: types.SimpleNamespace,
        env: mri_envs.ActiveMRIEnv,
        device: torch.device,
    ):
        self.options = options
        self.env = env
        self.options.image_width = self.env.kspace_width
        self.steps = 0
        self.episode = 0
        self.best_test_score = -np.inf
        self.device = device
        self.replay_memory = None

        self.window_size = 1000
        self.reward_images_in_window = np.zeros(self.window_size)
        self.current_score_auc_window = np.zeros(self.window_size)

        # ------- Init loggers ------
        self.writer = tensorboardX.SummaryWriter(
            os.path.join(self.options.checkpoints_dir)
        )
        self.logger = logging.getLogger()
        logging_level = logging.DEBUG if self.options.debug else logging.INFO
        self.logger.setLevel(logging_level)
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging_level)
        formatter = logging.Formatter(
            "%(asctime)s - %(threadName)s - %(levelname)s: %(message)s"
        )
        ch.setFormatter(formatter)
        self.logger.addHandler(ch)
        fh = logging.FileHandler(
            os.path.join(self.options.checkpoints_dir, "train.log")
        )
        fh.setLevel(logging_level)
        fh.setFormatter(formatter)
        self.logger.addHandler(fh)

        self.logger.info("Creating DDQN model.")

        self.logger.info(
            f"Creating replay buffer with capacity {options.mem_capacity}."
        )

        # ------- Create replay buffer and networks ------
        # See _encode_obs_dict() for tensor format
        self.obs_shape = (2, self.env.kspace_height + 2, self.env.kspace_width)
        self.replay_memory = replay_buffer.ReplayMemory(
            options.mem_capacity,
            self.obs_shape,
            self.options.dqn_batch_size,
            self.options.dqn_burn_in,
            use_normalization=self.options.dqn_normalize,
        )
        self.logger.info("Created replay buffer.")
        self.policy = DDQN(device, self.replay_memory, self.options)
        self.target_net = DDQN(device, None, self.options)
        self.target_net.eval()
        self.logger.info(
            f"Created neural networks with {self.env.action_space.n} outputs."
        )

        # ------- Files used to communicate with DDQNTester ------
        self.folder_lock_path = DDQNTrainer.get_lock_filename(
            self.options.checkpoints_dir
        )
        with _get_folder_lock(self.folder_lock_path):
            # Write options so that tester can read them
            with open(
                DDQNTrainer.get_options_filename(self.options.checkpoints_dir), "wb"
            ) as f:
                pickle.dump(self.options, f)
            # Remove previous done file since training will start over
            done_file = DDQNTrainer.get_done_filename(self.options.checkpoints_dir)
            if os.path.isfile(done_file):
                os.remove(done_file)

    @staticmethod
    def get_done_filename(path):
        return os.path.join(path, "DONE")

    @staticmethod
    def get_name_latest_checkpoint(path):
        return os.path.join(path, "policy_checkpoint.pth")

    @staticmethod
    def get_options_filename(path):
        return os.path.join(path, "options.pickle")

    @staticmethod
    def get_lock_filename(path):
        return os.path.join(path, ".LOCK")

    def _max_replay_buffer_size(self):
        return min(self.options.num_train_steps, self.options.replay_buffer_size)

    def load_checkpoint_if_needed(self):
        if self.options.dqn_only_test or self.options.resume:
            policy_path = os.path.join(self.options.dqn_weights_path)
            if os.path.isfile(policy_path):
                self.load(policy_path)
                self.logger.info(f"Loaded DQN policy found at {policy_path}.")
            else:
                self.logger.warning(f"No DQN policy found at {policy_path}.")
                if self.options.dqn_only_test:
                    raise FileNotFoundError

    def _train_dqn_policy(self):
        """ Trains the DQN policy. """
        self.logger.info(
            f"Starting training at step {self.steps}/{self.options.num_train_steps}. "
            f"Best score so far is {self.best_test_score}."
        )

        steps_epsilon = self.steps
        while self.steps < self.options.num_train_steps:
            self.logger.info("Episode {}".format(self.episode + 1))

            # Evaluate the current policy
            if self.options.dqn_test_episode_freq and (
                self.episode % self.options.dqn_test_episode_freq == 0
            ):
                test_scores, _ = evaluation.evaluate(
                    self.env,
                    self.policy,
                    self.options.num_test_episodes,
                    self.options.seed,
                    "val",
                )
                self.env.set_training()
                auc_score = test_scores[self.options.reward_metric].sum(axis=1).mean()
                if "mse" in self.options.reward_metric:
                    auc_score *= -1
                if auc_score > self.best_test_score:
                    policy_path = os.path.join(
                        self.options.checkpoints_dir, "policy_best.pt"
                    )
                    self.save(policy_path)
                    self.best_test_score = auc_score
                    self.logger.info(
                        f"Saved DQN model with score {self.best_test_score} to {policy_path}."
                    )

            # Run an episode and update model
            obs, meta = self.env.reset()
            msg = ", ".join(
                [
                    f"({meta['fname'][i]}, {meta['slice_id'][i]})"
                    for i in range(len(meta["slice_id"]))
                ]
            )
            self.logger.info(f"Episode started with images {msg}.")
            all_done = False
            total_reward = 0
            auc_score = 0
            while not all_done:
                epsilon = _get_epsilon(steps_epsilon, self.options)
                action = self.policy.get_action(obs, eps_threshold=epsilon)
                next_obs, reward, done, meta = self.env.step(action)
                auc_score += meta["current_score"][self.options.reward_metric]
                all_done = all(done)
                self.steps += 1
                obs_tensor = _encode_obs_dict(obs)
                next_obs_tensor = _encode_obs_dict(next_obs)
                batch_size = len(obs)
                for i in range(batch_size):
                    self.policy.add_experience(
                        obs_tensor[i], action[i], next_obs_tensor[i], reward[i], done[i]
                    )

                update_results = self.policy.update_parameters(self.target_net)
                torch.cuda.empty_cache()
                if self.steps % self.options.target_net_update_freq == 0:
                    self.logger.info("Updating target network.")
                    self.target_net.load_state_dict(self.policy.state_dict())
                steps_epsilon += 1

                # Adding per-step tensorboard logs
                if self.steps % 250 == 0:
                    self.logger.debug("Writing to tensorboard.")
                    self.writer.add_scalar("epsilon", epsilon, self.steps)
                    if update_results is not None:
                        self.writer.add_scalar(
                            "loss", update_results["loss"], self.steps
                        )
                        self.writer.add_scalar(
                            "grad_norm", update_results["grad_norm"], self.steps
                        )
                        self.writer.add_scalar(
                            "mean_q_value", update_results["q_values_mean"], self.steps
                        )
                        self.writer.add_scalar(
                            "std_q_value", update_results["q_values_std"], self.steps
                        )

                total_reward += reward
                obs = next_obs

            # Adding per-episode tensorboard logs
            total_reward = total_reward.mean().item()
            auc_score = auc_score.mean().item()
            self.reward_images_in_window[self.episode % self.window_size] = total_reward
            self.current_score_auc_window[self.episode % self.window_size] = auc_score
            self.writer.add_scalar("episode_reward", total_reward, self.episode)
            self.writer.add_scalar(
                "average_reward_images_in_window",
                np.sum(self.reward_images_in_window)
                / min(self.episode + 1, self.window_size),
                self.episode,
            )
            self.writer.add_scalar(
                "average_auc_score_in_window",
                np.sum(self.current_score_auc_window)
                / min(self.episode + 1, self.window_size),
                self.episode,
            )

            self.episode += 1

            if self.episode % self.options.freq_dqn_checkpoint_save == 0:
                self.checkpoint(save_memory=False)

        self.checkpoint()

        # Writing DONE file with best test score
        with _get_folder_lock(self.folder_lock_path):
            with open(
                DDQNTrainer.get_done_filename(self.options.checkpoints_dir), "w"
            ) as f:
                f.write(str(self.best_test_score))

        return self.best_test_score

    def __call__(self):
        self.load_checkpoint_if_needed()
        return self._train_dqn_policy()

    def checkpoint(self, save_memory=True):
        policy_path = DDQNTrainer.get_name_latest_checkpoint(
            self.options.checkpoints_dir
        )
        self.save(policy_path)
        self.logger.info(f"Saved DQN checkpoint to {policy_path}")
        if save_memory:
            self.logger.info("Now saving replay memory.")
            memory_path = self.replay_memory.save(
                self.options.checkpoints_dir, "replay_buffer.pt"
            )
            self.logger.info(f"Saved replay buffer to {memory_path}.")

    def save(self, path):
        with _get_folder_lock(self.folder_lock_path):
            torch.save(
                {
                    "dqn_weights": self.policy.state_dict(),
                    "target_weights": self.target_net.state_dict(),
                    "episode": self.episode,
                    "steps": self.steps,
                    "best_test_score": self.best_test_score,
                    "reward_images_in_window": self.reward_images_in_window,
                    "current_score_auc_window": self.current_score_auc_window,
                },
                path,
            )

    def load(self, path):
        checkpoint = torch.load(path)
        self.policy.load_state_dict(checkpoint["dqn_weights"])
        self.episode = checkpoint["episode"] + 1
        if not self.options.dqn_only_test:
            self.target_net.load_state_dict(checkpoint["target_weights"])
            self.steps = checkpoint["steps"]
            self.best_test_score = checkpoint["best_test_score"]
            self.reward_images_in_window = checkpoint["reward_images_in_window"]
            self.current_score_auc_window = checkpoint["current_score_auc_window"]
