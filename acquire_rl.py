import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from models import create_model
from models.fft_utils import RFFT, IFFT, FFT
from options.test_options import TestOptions
from util import util

from data import CreateFtTLoader
from gym.spaces import Box, Discrete


rfft = RFFT()
ifft = IFFT()
fft = FFT()

CONJUGATE_SYMMETRIC = True
IMAGE_WIDTH = 128
NUM_LINES_INITIAL = 5
BUDGET = 20

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def infinite_iterator(iterator):
    while True:
        yield from iterator


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class DDQN(nn.Module):
    def __init__(self, num_actions):
        super(DDQN, self).__init__()
        self.conv_image = nn.Sequential(
            nn.Conv2d(2, 128, kernel_size=3, stride=2, padding=0), nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=0), nn.ReLU(),
            nn.Conv2d(256, 128, kernel_size=3, stride=2, padding=0), nn.ReLU(),
            nn.Conv2d(128, 16, kernel_size=3, stride=1, padding=0), nn.ReLU(), Flatten()
            # nn.Conv2d(2, 32, kernel_size=8, stride=4, padding=0), nn.ReLU(),
            # nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0), nn.ReLU(),
            # nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=0), nn.ReLU(),
            # nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=0), nn.ReLU(), Flatten()
        )

        self.conv_fft = nn.Sequential(
            nn.Conv2d(2, 128, kernel_size=3, stride=2, padding=0), nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=0), nn.ReLU(),
            nn.Conv2d(256, 128, kernel_size=3, stride=2, padding=0), nn.ReLU(),
            nn.Conv2d(128, 16, kernel_size=3, stride=1, padding=0), nn.ReLU(), Flatten()
            # nn.Conv2d(2, 32, kernel_size=8, stride=4, padding=0), nn.ReLU(),
            # nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0), nn.ReLU(),
            # nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=0), nn.ReLU(),
            # nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=0), nn.ReLU(), Flatten()
        )

        self.fc = nn.Sequential(
            nn.Linear(2 * 13 * 13 * 16, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, num_actions)
        )

        self.optimizer = optim.Adam(self.parameters(), lr=3e-4)

    def forward(self, x):
        reconstructions = x[:, :2, :, :]
        masked_rffts = x[:, 2:, :, :]

        image_encoding = self.conv_image(reconstructions)
        rfft_encoding = self.conv_image(masked_rffts)

        return self.fc(torch.cat((image_encoding, rfft_encoding), dim=1))

    def optimize(self, batch):
        observations = torch.tensor(batch['observations']).to(device)
        actions = torch.tensor(batch['actions']).to(device)
        self.zero_grad()
        predictions = self(observations)
        loss = F.cross_entropy(predictions, actions)
        loss.backward()
        self.optimizer.step()

        return loss.item()


# noinspection PyAttributeOutsideInit
class ReconstrunctionEnv:
    def __init__(self, model, data_loader, initial_mask, budget):
        self.observation_space = Box(low=-50000, high=50000, shape=(4, 128, 128))
        factor = 2 if CONJUGATE_SYMMETRIC else 1
        self.action_space = Discrete(IMAGE_WIDTH - factor * NUM_LINES_INITIAL)

        self._model = model
        self._data_loader = infinite_iterator(data_loader)
        self._ground_truth = None
        self._initial_mask = initial_mask.to(model.device)
        self.budget = budget
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
    def compute_score(reconstruction, ground_truth, type='mse'):
        reconstruction = reconstruction[:, :1, :, :]
        ground_truth = ground_truth[:, :1, :, :]
        if type == 'mse':
            score = F.mse_loss(reconstruction, ground_truth)
        elif type == 'ssim':
            score = util.ssim_metric(reconstruction, ground_truth)
        else:
            raise ValueError
        return score

    @staticmethod
    def _make_state(reconstructions, masked_rffts):
        return np.concatenate((reconstructions[-1].squeeze().cpu().numpy(),
                               masked_rffts.squeeze().cpu().numpy()))

    def _compute_masked_rfft_reconstruction_and_score(self):
        with torch.no_grad():
            masked_rffts = ReconstrunctionEnv.compute_masked_rfft(self._ground_truth, self._current_mask)
            reconstructions, _, _ = self._model.netG(ifft(masked_rffts), self._current_mask)
            score = ReconstrunctionEnv.compute_score(reconstructions[-1], self._ground_truth)
        return masked_rffts, reconstructions, score

    def reset(self):
        _, self._ground_truth = next(self._data_loader)
        self._ground_truth = self._ground_truth.to(self._model.device)
        self._current_mask = self._initial_mask
        self._scans_left = self.budget
        masked_rffts, reconstructions, self._current_score = self._compute_masked_rfft_reconstruction_and_score()
        return ReconstrunctionEnv._make_state(reconstructions, masked_rffts)

    def step(self, action):
        assert self._scans_left > 0
        line_to_scan = NUM_LINES_INITIAL + action
        self._current_mask = ReconstrunctionEnv.compute_new_mask(self._current_mask, line_to_scan)
        masked_rffts, reconstructions, new_score = self._compute_masked_rfft_reconstruction_and_score()

        reward = self._current_score - new_score
        self._current_score = new_score

        self._scans_left -= 1
        done = (self._scans_left == 0)

        return ReconstrunctionEnv._make_state(reconstructions, masked_rffts), reward.item(), done, {}


def generate_initial_mask(num_lines):
        mask = torch.zeros(1, 1, 1, IMAGE_WIDTH)
        for i in range(num_lines):
            mask[0, 0, 0, i] = 1
            if CONJUGATE_SYMMETRIC:
                mask[0, 0, 0, -(i+1)] = 1
        return mask


if __name__ == '__main__':
    opts = TestOptions().parse()
    assert opts.no_dropout

    opts.batchSize = 1
    opts.results_dir = opts.checkpoints_dir
    test_data_loader = CreateFtTLoader(opts, is_test=True)
    model = create_model(opts)
    model.setup(opts)
    model.eval()

    env = ReconstrunctionEnv(model, test_data_loader, generate_initial_mask(NUM_LINES_INITIAL), BUDGET)

    policy = DDQN(env.action_space.n)
    policy = policy.to(device)

    min_o = np.inf
    max_o = -np.inf
    batch = {'observations': [], 'actions': []}
    for i in range(1000):
        env.reset()
        total_reward = 0
        for j in range(10, 50):
            obs, reward, done, _ = env.step(j)
            # pred = policy(torch.tensor(obs).unsqueeze(0))
            batch['observations'].append(obs)
            batch['actions'].append(j)
            # min_o = min(min_o, np.min(obs))
            # max_o = max(max_o, np.max(obs))
            total_reward += reward
            if done:
                break
        print(total_reward)

    for i in range(50):
        loss = policy.optimize(batch)
        print(i, loss)

    policy.eval()
    for obs in batch['observations']:
        print(torch.argmax(policy(torch.tensor(obs).unsqueeze(0).to(device))))

    print(min_o, max_o)
