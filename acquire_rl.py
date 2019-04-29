import math
import numpy as np
import torch
import torch.nn.functional as F

from models import create_model
from models.fft_utils import RFFT, IFFT, FFT
from options.test_options import TestOptions
from tensorboardX import SummaryWriter
from util import util
from util.rl.dqn import DDQN
from util.rl.replay_buffer import ExperienceBuffer

from data import CreateFtTLoader
from gym.spaces import Box, Discrete


rfft = RFFT()
ifft = IFFT()
fft = FFT()

# TODO make these command line args
CONJUGATE_SYMMETRIC = True
IMAGE_WIDTH = 128
NUM_LINES_INITIAL = 5
BUDGET = 10
EPS_START = 0.9
EPS_END = 0.01
EPS_DECAY = 10000
NUM_EPISODES = 10000
WINDOW = 100
LOG_Q_VALUE_STEP = 10
BATCH_SIZE = 16
TEST_FREQ = 2

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_epsilon(steps_done):
    return EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)


def infinite_iterator(iterator):
    while True:
        yield from iterator


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
        if self._ground_truth is None:
            _, self._ground_truth = next(self._data_loader)
            self._ground_truth = self._ground_truth.to(self._model.device)
        self._current_mask = self._initial_mask
        self._scans_left = self.budget
        masked_rffts, reconstructions, self._current_score = self._compute_masked_rfft_reconstruction_and_score()
        return ReconstrunctionEnv._make_state(reconstructions, masked_rffts)

    def step(self, action):
        assert self._scans_left > 0
        line_to_scan = NUM_LINES_INITIAL + action
        if int(self._current_mask[0, 0, 0, line_to_scan].item()) == 1:
            return None, -1.0, True, {}
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
            print(actions)
    writer.add_scalar('eval/average_reward', average_total_reward / num_episodes, step)


def train_policy(env, policy, target_net, writer):
    steps = 0
    for episode in range(NUM_EPISODES):
        print('Episode {}'.format(episode))
        target_net.load_state_dict(policy.state_dict())
        obs = env.reset()
        done = False
        total_reward = 0
        while not done:
            epsilon = get_epsilon(steps)
            action = policy.get_action(obs, epsilon)
            next_obs, reward, done, _ = env.step(action)
            steps += 1
            if done:
                next_obs = None
            policy.add_experience(obs, action, next_obs, reward)
            loss, grad_norm = policy.update_parameters(target_net, BATCH_SIZE)
            writer.add_scalar('epsilon', epsilon, steps)
            if loss is not None:
                writer.add_scalar('loss', loss, steps)
                writer.add_scalar('grad_norm', grad_norm, steps)
            total_reward += reward
            obs = next_obs
        writer.add_scalar('episode_reward', total_reward, episode)
        if (episode + 1) % TEST_FREQ == 0:
            test_policy(env, policy, writer, 1, episode)


def main(opts):
    test_data_loader = CreateFtTLoader(opts, is_test=True)
    model = create_model(opts)
    model.setup(opts)
    model.eval()

    writer = SummaryWriter('/checkpoint/lep/active_acq/dqn')

    env = ReconstrunctionEnv(model, test_data_loader, generate_initial_mask(NUM_LINES_INITIAL), BUDGET)

    policy = DDQN(env.action_space.n, device, ExperienceBuffer(1000000, (4, 128, 128))).to(device)
    target_net = DDQN(env.action_space.n, device, None).to(device)

    train_policy(env, policy, target_net, writer)


if __name__ == '__main__':
    opts = TestOptions().parse()
    assert opts.no_dropout

    opts.batchSize = 1
    opts.results_dir = opts.checkpoints_dir

    main(opts)
