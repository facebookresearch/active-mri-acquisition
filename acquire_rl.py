import logging
import math
import numpy as np
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

from models import create_model
from models.fft_utils import RFFT, IFFT, FFT
from options.rl_options import RLOptions
from tensorboardX import SummaryWriter
from util import util
from util.rl.dqn import DDQN, get_epsilon
from util.rl.replay_buffer import ExperienceBuffer

from data import CreateFtTLoader
from gym.spaces import Box, Discrete


rfft = RFFT()
ifft = IFFT()
fft = FFT()


CONJUGATE_SYMMETRIC = True
IMAGE_WIDTH = 128

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def infinite_iterator(iterator):
    while True:
        yield from iterator


# TODO fix evaluator code because I think self.seperate_mask should now be [0, i, 0, 0, i] = 1
class KSpaceMap(nn.Module):
    def __init__(self, img_width=128):
        super(KSpaceMap, self).__init__()

        self.RFFT = RFFT()
        self.IFFT = IFFT()
        self.img_width = img_width
        self.register_buffer('separated_masks', torch.FloatTensor(1, img_width, 1, 1, img_width))
        self.separated_masks.fill_(0)
        for i in range(img_width):
            self.separated_masks[0, i, 0, 0, i] = 1

    def forward(self, input, mask):
        batch_size, _, img_height, img_width = input.shape
        assert img_width == self.img_width
        k_space = self.RFFT(input[:, :1, :, :])  # Take only real part

        # This code creates w channels, where the i-th channel is a copy of k_space
        # with everything but the i-th column masked out
        k_space = k_space.unsqueeze(1).repeat(1, img_width, 1, 1, 1)  # [batch_size , w, 2, h, w]
        masked_kspace = self.separated_masks * k_space
        masked_kspace = masked_kspace.view(batch_size * img_width, 2, img_height, img_width)

        # The imaginary part is discarded
        return self.IFFT(masked_kspace)[:, 0, :, :].view(batch_size, img_width, img_height, img_width)


# noinspection PyAttributeOutsideInit
class ReconstrunctionEnv:
    def __init__(self, model, data_loader, initial_mask, opts):
        self.opts = opts

        self.observation_space = Box(low=-50000, high=50000, shape=(4, 128, 128))
        factor = 2 if CONJUGATE_SYMMETRIC else 1
        # num_actions = (IMAGE_WIDTH - factor * NUM_LINES_INITIAL) // 2
        num_actions = opts.budget + 2
        self.action_space = Discrete(num_actions)

        self._model = model
        self._data_loader = infinite_iterator(data_loader)
        self._ground_truth = None
        self._initial_mask = initial_mask.to(model.device)
        self.kspace_map = KSpaceMap(img_width=IMAGE_WIDTH).to(device)    # Used to compute spectral maps
        self.reset()

    @staticmethod
    def compute_masked_rfft(ground_truth, mask):
        state = rfft(ground_truth) * mask
        return state

    @staticmethod
    def compute_new_mask(old_mask, action):
        new_mask = old_mask.clone().squeeze()
        new_mask[action] = 1
        if CONJUGATE_SYMMETRIC:
            new_mask[IMAGE_WIDTH - action] = 1
        return new_mask.view(1, 1, 1, -1)

    @staticmethod
    def compute_score(reconstruction, ground_truth, kind='mse'):
        reconstruction = reconstruction[:, :1, :, :]
        ground_truth = ground_truth[:, :1, :, :]
        if kind == 'mse':
            score = F.mse_loss(reconstruction, ground_truth)
        elif kind == 'ssim':
            score = util.ssim_metric(reconstruction, ground_truth)
        else:
            raise ValueError
        return score

    def _compute_observation_and_score(self):
        with torch.no_grad():
            masked_rffts = ReconstrunctionEnv.compute_masked_rfft(self._ground_truth, self._current_mask)
            reconstructions, _, mask_embed = self._model.netG(ifft(masked_rffts), self._current_mask)
            spectral_maps = self.kspace_map(reconstructions[-1], self._current_mask)
            observation = torch.cat([spectral_maps, mask_embed], dim=1).shape
            score = ReconstrunctionEnv.compute_score(reconstructions[-1], self._ground_truth)
        return observation, score

    def reset(self):
        if self._ground_truth is None:
            _, self._ground_truth = next(self._data_loader)
            self._ground_truth = self._ground_truth.to(self._model.device)
        self._current_mask = self._initial_mask
        self._scans_left = self.opts.budget
        observation, self._current_score = self._compute_observation_and_score()
        return observation.squeeze().cpu().numpy().astype(np.float32)

    def step(self, action):
        assert self._scans_left > 0
        line_to_scan = self.opts.initial_num_lines + action
        has_already_been_scanned = bool(self._current_mask[0, 0, 0, line_to_scan].item())
        self._current_mask = ReconstrunctionEnv.compute_new_mask(self._current_mask, line_to_scan)
        observation, new_score = self._compute_observation_and_score()

        reward = -1.0 if has_already_been_scanned else (self._current_score - new_score).item() / 0.01
        self._current_score = new_score

        self._scans_left -= 1
        done = (self._scans_left == 0)

        return observation, reward, done, {}


def generate_initial_mask(num_lines):
        mask = torch.zeros(1, 1, 1, IMAGE_WIDTH)
        for i in range(num_lines):
            mask[0, 0, 0, i] = 1
            if CONJUGATE_SYMMETRIC:
                mask[0, 0, 0, -(i+1)] = 1
        return mask


def test_policy(env, policy, writer, num_episodes, step):
    average_total_reward = 0
    for episode in range(num_episodes):
        obs = env.reset()
        done = False
        total_reward = 0
        actions = []
        while not done:
            action = policy.get_action(obs, 0.)
            actions.append(action)
            next_obs, reward, done, _ = env.step(action)
            if done:
                next_obs = None
            total_reward += reward
            obs = next_obs
        average_total_reward += total_reward
        if episode == 0:
            logging.debug(actions)
    writer.add_scalar('eval/average_reward', average_total_reward / num_episodes, step)


def train_policy(env, policy, target_net, writer, opts):
    steps = 0
    for episode in range(opts.num_episodes):
        logging.info('Episode {}'.format(episode))
        obs = env.reset()
        done = False
        total_reward = 0
        while not done:
            epsilon = get_epsilon(steps, opts)
            action = policy.get_action(obs, epsilon)
            next_obs, reward, done, _ = env.step(action)
            steps += 1
            if done:
                next_obs = None
            policy.add_experience(obs, action, next_obs, reward)
            loss, grad_norm = policy.update_parameters(target_net, opts.rl_batch_size)
            if steps % opts.target_net_update_freq == 0:
                logging.info('Updating target network.')
                target_net.load_state_dict(policy.state_dict())
            writer.add_scalar('epsilon', epsilon, steps)
            if loss is not None:
                writer.add_scalar('loss', loss, steps)
                writer.add_scalar('grad_norm', grad_norm, steps)
            total_reward += reward
            obs = next_obs
        writer.add_scalar('episode_reward', total_reward, episode)
        if (episode + 1) % opts.agent_test_episode_freq == 0:
            test_policy(env, policy, writer, 1, episode)


def main(opts):
    test_data_loader = CreateFtTLoader(opts, is_test=True)
    model = create_model(opts)
    model.setup(opts)
    model.eval()

    writer = SummaryWriter('/checkpoint/lep/active_acq/dqn')

    env = ReconstrunctionEnv(
        model, test_data_loader, generate_initial_mask(opts.initial_num_lines), opts)

    logging.info('Created environment with {} actions'.format(env.action_space.n))

    policy = DDQN(env.action_space.n, device, ExperienceBuffer(1000000, (4, 128, 128))).to(device)
    target_net = DDQN(env.action_space.n, device, None).to(device)

    train_policy(env, policy, target_net, writer, opts)


if __name__ == '__main__':
    # Reading options
    opts = RLOptions().parse()
    opts.batchSize = 1
    opts.results_dir = opts.checkpoints_dir

    # Initializing logger
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(threadName)s - %(levelname)s: %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    main(opts)
