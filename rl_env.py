import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from data import CreateFtTLoader
from models import create_model
from models.fft_utils import RFFT, IFFT, FFT
from util import util

from gym.spaces import Box, Discrete


rfft = RFFT()
ifft = IFFT()
fft = FFT()


CONJUGATE_SYMMETRIC = True
IMAGE_WIDTH = 128
NUM_LINES_INITIAL = 10

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class KSpaceMap(nn.Module):
    """Auxiliary module used to compute spectral maps of a zero-filled reconstruction.
        See https://arxiv.org/pdf/1902.03051.pdf for details.
    """
    def __init__(self, img_width=128):
        super(KSpaceMap, self).__init__()

        self.img_width = img_width
        self.register_buffer('separated_masks', torch.FloatTensor(1, img_width, 1, 1, img_width))
        self.separated_masks.fill_(0)
        for i in range(img_width):
            self.separated_masks[0, i, 0, 0, i] = 1

    def forward(self, input, mask):
        batch_size, _, img_height, img_width = input.shape
        assert img_width == self.img_width
        k_space = rfft(input[:, :1, :, :])  # Take only real part

        # This code creates w channels, where the i-th channel is a copy of k_space
        # with everything but the i-th column masked out
        k_space = k_space.unsqueeze(1).repeat(1, img_width, 1, 1, 1)  # [batch_size , w, 2, h, w]
        masked_kspace = self.separated_masks * k_space
        masked_kspace = masked_kspace.view(batch_size * img_width, 2, img_height, img_width)

        # The imaginary part is discarded
        return ifft(masked_kspace)[:, 0, :, :].view(batch_size, img_width, img_height, img_width)


# noinspection PyAttributeOutsideInit
class ReconstrunctionEnv:
    """ RL environment representing the active acquisition process with reconstruction model. """
    def __init__(self, initial_mask, opts):
        self.opts = opts
        train_loader, valid_loader = CreateFtTLoader(opts, is_test=False)
        test_loader = CreateFtTLoader(opts, is_test=True)
        self._dataset_train = train_loader.dataset
        self._dataset_test = test_loader.dataset
        model = create_model(opts)
        model.setup(opts)
        model.eval()

        obs_shape = None
        if opts.rl_model_type == 'spectral_maps':
            obs_shape = (134, 128, 128)
        if opts.rl_model_type == 'two_streams':
            obs_shape = (4, 128, 128)
        self.observation_space = Box(low=-50000, high=50000, shape=obs_shape)
        factor = 2 if CONJUGATE_SYMMETRIC else 1
        num_actions = (IMAGE_WIDTH - factor * NUM_LINES_INITIAL) // 2
        # num_actions = opts.budget + 10
        self.action_space = Discrete(num_actions)

        self._model = model
        self._ground_truth = None
        self._initial_mask = initial_mask.to(model.device)
        self.k_space_map = KSpaceMap(img_width=IMAGE_WIDTH).to(device)    # Used to compute spectral maps

        # These two store a shuffling of the datasets
        self._test_order = np.load('data/rl_test_order.npy')
        self._train_order = np.load('data/rl_train_order.npy')
        self.image_idx_test = 0
        self.image_idx_train = 0
        self.is_testing = False
        self.reset()

    def set_testing(self):
        self.is_testing = True
        self.image_idx_test = 0

    def set_training(self):
        self.is_testing = False
        self.image_idx_train = 0

    @staticmethod
    def compute_masked_rfft(ground_truth, mask):
        state = rfft(ground_truth) * mask
        return state

    @staticmethod
    def _compute_score(reconstruction, ground_truth, kind='mse'):
        reconstruction = reconstruction[:, :1, :, :]
        ground_truth = ground_truth[:, :1, :, :]
        if kind == 'mse':
            score = F.mse_loss(reconstruction, ground_truth)
        elif kind == 'ssim':
            score = util.ssim_metric(reconstruction, ground_truth)
        else:
            raise ValueError
        return score

    def compute_new_mask(self, old_mask, action):
        """ Computes a new mask by adding the action to the given old mask.

            Note that action is relative to the set of valid k-space lines that can be scanned. That is, action = 0
            represents the lowest index of k-space lines that are not part of the initial mask.
        """
        line_to_scan = self.opts.initial_num_lines + action
        new_mask = old_mask.clone().squeeze()
        had_arlready_been_scanned = bool(new_mask[line_to_scan])
        new_mask[line_to_scan] = 1
        if CONJUGATE_SYMMETRIC:
            new_mask[IMAGE_WIDTH - line_to_scan - 1] = 1
        return new_mask.view(1, 1, 1, -1), had_arlready_been_scanned

    def compute_score(self, use_reconstruction=True, kind='mse', ground_truth=None, mask_to_use=None):
        """ Computes the score (MSE or SSIM) of the current state with respect to the current ground truth.

            This method takes the current ground truth, masks it with the current mask and creates
            a zero-filled reconstruction from the masked image; this zero-filled reconstruction can be passed
            through the reconstruction network. The score evaluates the difference between the final reconstruction
            and the current ground truth.

            It is possible to pass alternate ground truth and mask.

            @:param use_reconstruction: specifies if the reconstruction network will be used or not.
            @:param ground_truth: specifies if the score has to be computed with respect to an alternate "ground truth".
            @:param mask_to_use: specifies if the score has to be computed with an alternate mask.
        """
        with torch.no_grad():
            if ground_truth is None:
                ground_truth = self._ground_truth
            if mask_to_use is None:
                mask_to_use = self._current_mask
            masked_rffts = ReconstrunctionEnv.compute_masked_rfft(ground_truth, mask_to_use)
            if use_reconstruction:
                reconstructions, _, mask_embed = self._model.netG(ifft(masked_rffts), mask_to_use)
                image = reconstructions[-1]
            else:
                image = ifft(masked_rffts)
        return [ReconstrunctionEnv._compute_score(img.unsqueeze(0), ground_truth, kind) for img in image]

    def _compute_observation_and_score_spectral_maps(self):
        with torch.no_grad():
            masked_rffts = ReconstrunctionEnv.compute_masked_rfft(self._ground_truth, self._current_mask)
            reconstructions, _, mask_embed = self._model.netG(ifft(masked_rffts), self._current_mask)
            spectral_maps = self.k_space_map(reconstructions[-1], self._current_mask)
            observation = torch.cat([spectral_maps, mask_embed], dim=1)
            score = ReconstrunctionEnv._compute_score(reconstructions[-1], self._ground_truth)
        return observation.squeeze().cpu().numpy().astype(np.float32), score

    def _compute_observation_and_score_two_streams(self):
        with torch.no_grad():
            masked_rffts = ReconstrunctionEnv.compute_masked_rfft(self._ground_truth, self._current_mask)
            reconstructions, _, _ = self._model.netG(ifft(masked_rffts), self._current_mask)
            observation = torch.cat([reconstructions[-1], masked_rffts], dim=1)
            score = ReconstrunctionEnv._compute_score(reconstructions[-1], self._ground_truth)
        return observation.squeeze().cpu().numpy().astype(np.float32), score

    def _compute_observation_and_score(self):
        if self.opts.rl_model_type == 'spectral_maps':
            return self._compute_observation_and_score_spectral_maps()
        if self.opts.rl_model_type == 'two_streams':
            return self._compute_observation_and_score_two_streams()

    def reset(self):
        """ Loads a new image from the dataset and starts a new episode with this image.

            If [[self.opts.sequential_images]] is True, it loops over images in the dataset in order. Otherwise,
            it selects a random image from the first [[self.num_{train/test}_images]] in the dataset. The dataset
            is ordered according to [[self._{train/test}_order]].

        """
        if self.opts.sequential_images:
            if self.is_testing:
                if self.image_idx_test == min(self.opts.num_test_images, len(self._dataset_test)):
                    return None     # Returns None to signal that testing is done
                _, self._ground_truth = self._dataset_test.__getitem__(self._test_order[self.image_idx_test])
                logging.debug('Testing episode started with image {}'.format(self._test_order[self.image_idx_test]))
                self.image_idx_test += 1
            else:
                _, self._ground_truth = self._dataset_train.__getitem__(self._train_order[self.image_idx_train])
                logging.debug('Training episode started with image {}'.format(self._train_order[self.image_idx_train]))
                self.image_idx_train = (self.image_idx_train + 1) % self.opts.num_train_images
        else:
            dataset_to_check = self._dataset_test if self.is_testing else self._dataset_train
            max_num_images = self.opts.num_test_images if self.is_testing else self.opts.num_train_images
            dataset_len = min(len(dataset_to_check), max_num_images)
            index_chosen_image = np.random.choice(dataset_len)
            logging.debug('{} episode started with randomly chosen image {}/{}'.format(
                'Testing' if self.is_testing else 'Training', index_chosen_image, dataset_len))
            _, self._ground_truth = self._dataset_train.__getitem__(index_chosen_image)
        self._ground_truth = self._ground_truth.to(self._model.device).unsqueeze(0)
        self._current_mask = self._initial_mask
        self._scans_left = min(self.opts.budget, self.action_space.n)
        observation, self._current_score = self._compute_observation_and_score()
        return observation

    def step(self, action):
        """ Adds a new line (specified by the action) to the current mask and computes the resulting observation and
            reward (drop in MSE after reconstructing with respect to the current ground truth).
        """
        assert self._scans_left > 0
        self._current_mask, has_already_been_scanned = self.compute_new_mask(self._current_mask, action)
        observation, new_score = self._compute_observation_and_score()

        reward = -1.0 if has_already_been_scanned else (self._current_score - new_score).item() / 0.01
        self._current_score = new_score

        self._scans_left -= 1
        done = (self._scans_left == 0)

        return observation, reward, done, {}

    def get_evaluator_action(self):
        """ Returns the action recommended by the evaluator network of [[self._model]]. """
        with torch.no_grad():
            self._model.set_input_exp(self._ground_truth, self._current_mask)
            self._model.forward()
            evaluator_input = self._model.create_D_input(self._model.fake_B)
            kspace_scores = self._model.netD(evaluator_input, self._current_mask)
            kspace_scores.masked_fill_(self._current_mask.squeeze().byte(), 100000)
            return torch.argmin(kspace_scores).item() - NUM_LINES_INITIAL


def generate_initial_mask(num_lines):
        mask = torch.zeros(1, 1, 1, IMAGE_WIDTH)
        for i in range(num_lines):
            mask[0, 0, 0, i] = 1
            if CONJUGATE_SYMMETRIC:
                mask[0, 0, 0, -(i+1)] = 1
        return mask
