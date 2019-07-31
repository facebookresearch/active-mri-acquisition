import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from data import create_data_loaders
from models.evaluator import EvaluatorNetwork
from models.fft_utils import RFFT, IFFT, FFT, preprocess_inputs, clamp, load_checkpoint
from models.reconstruction import ReconstructorNetwork
from util import util
from typing import Dict, Tuple, Union

from gym.spaces import Box, Discrete

rfft = RFFT()
ifft = IFFT()
fft = FFT()
fft_functions = {'rfft': rfft, 'ifft': ifft, 'fft': fft}

CONJUGATE_SYMMETRIC = True
IMAGE_WIDTH = 128
NUM_LINES_INITIAL = 10

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# TODO Organize imports and finish adding type info


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
        k_space = rfft(input[:, :1, ...])  # Take only real part

        # This code creates w channels, where the i-th channel is a copy of k_space
        # with everything but the i-th column masked out
        k_space = k_space.unsqueeze(1).repeat(1, img_width, 1, 1, 1)  # [batch_size , w, 2, h, w]
        masked_kspace = self.separated_masks * k_space
        masked_kspace = masked_kspace.view(batch_size * img_width, 2, img_height, img_width)

        # The imaginary part is discarded
        return ifft(masked_kspace)[:, 0, ...].view(batch_size, img_width, img_height, img_width)


# noinspection PyAttributeOutsideInit
class ReconstructionEnv:
    """ RL environment representing the active acquisition process with reconstruction model. """

    def __init__(self, initial_mask, options):
        """Creates a new environment.

            @param `initial_mask`: The initial mask to use at the start of each episode.
            @param `options`: Should specify the following options:
                -`checkpoints_dir`: directory where models are stored.
                -`sequential_images`: If true, each episode presents the next image in the dataset,
                    otherwise a random image is presented.
                -`budget`: how many actions to choose (the horizon of the episode).
                -`rl_obs_type`: one of {'spectral_maps', 'two_streams', 'concatenate_mask'}
                -`initial_num_lines`: how many k-space lines to start with.
                -'num_train_images`: the number of images to use for training. If it's None,
                    then all images will be used.
                -`num_test_images`: the number of images to use for test. If it's None, then
                    all images will be used.
        """
        # TODO remove initial_mask argument (control this generation inside the class)

        self.options = options
        self.options.device = device
        train_loader, valid_loader = create_data_loaders(options, is_test=False)
        test_loader = create_data_loaders(options, is_test=True)
        self._dataset_train = train_loader.dataset
        self._dataset_test = test_loader.dataset
        self.num_train_images = self.options.num_train_images
        self.num_test_images = self.options.num_test_images
        if self.num_train_images is None or len(self._dataset_train) < self.num_train_images:
            self.num_train_images = len(self._dataset_train)
        if self.num_test_images is None or len(self._dataset_test) < self.num_test_images:
            self.num_test_images = len(self._dataset_test)

        checkpoint = load_checkpoint(options.checkpoints_dir, 'regular_checkpoint.pth')
        self._reconstructor = ReconstructorNetwork(
            number_of_cascade_blocks=checkpoint['options'].number_of_cascade_blocks,
            n_downsampling=checkpoint['options'].n_downsampling,
            number_of_filters=checkpoint['options'].number_of_reconstructor_filters,
            number_of_layers_residual_bottleneck=checkpoint['options']
            .number_of_layers_residual_bottleneck,
            mask_embed_dim=checkpoint['options'].mask_embed_dim,
            dropout_probability=checkpoint['options'].dropout_probability,
            img_width=128,  # TODO : CHANGE!
            use_deconv=checkpoint['options'].use_deconv)
        self._reconstructor.load_state_dict(
            {key.replace('module.', ''): val
             for key, val in checkpoint['reconstructor'].items()})
        self._reconstructor.to(device)

        self._evaluator = EvaluatorNetwork(
            number_of_filters=checkpoint['options'].number_of_evaluator_filters,
            number_of_conv_layers=checkpoint['options'].number_of_evaluator_convolution_layers,
            use_sigmoid=False,
            width=checkpoint['options'].image_width,
            mask_embed_dim=checkpoint['options'].mask_embed_dim)
        self._evaluator.load_state_dict(
            {key.replace('module.', ''): val
             for key, val in checkpoint['evaluator'].items()})
        self._evaluator.to(device)

        obs_shape = None
        if options.rl_obs_type == 'spectral_maps':
            obs_shape = (134, 128, 128)
        if options.rl_obs_type == 'two_streams':
            obs_shape = (4, 128, 128)
        if options.rl_obs_type == 'concatenate_mask':
            obs_shape = (2, 129, 128)
        self.observation_space = Box(low=-50000, high=50000, shape=obs_shape)

        factor = 2 if CONJUGATE_SYMMETRIC else 1
        num_actions = (IMAGE_WIDTH - factor * NUM_LINES_INITIAL) // 2
        self.action_space = Discrete(num_actions)

        self._ground_truth = None
        self._initial_mask = initial_mask.to(device)
        self.k_space_map = KSpaceMap(img_width=IMAGE_WIDTH).to(device)

        # These two store a shuffling of the datasets
        # self._test_order = np.load('data/rl_test_order.npy')
        # self._train_order = np.load('data/rl_train_order.npy')
        self._train_order = np.random.permutation(len(self._dataset_train))
        self._test_order = np.random.permutation(len(self._dataset_test))
        self.image_idx_test = 0
        self.image_idx_train = 0
        self.is_testing = False

    def set_testing(self, reset_index=True):
        self.is_testing = True
        if reset_index:
            self.image_idx_test = 0

    def set_training(self, reset_index=False):
        self.is_testing = False
        if reset_index:
            self.image_idx_train = 0

    @staticmethod
    def compute_masked_rfft(ground_truth: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        state = rfft(ground_truth) * mask
        return state

    @staticmethod
    def _compute_score(reconstruction: torch.Tensor, ground_truth: torch.Tensor,
                       kind: str = 'mse') -> torch.Tensor:
        reconstruction = reconstruction[:, :1, ...]
        ground_truth = ground_truth[:, :1, ...]
        if kind == 'mse':
            score = F.mse_loss(reconstruction, ground_truth)
        elif kind == 'ssim':
            score = util.ssim_metric(reconstruction, ground_truth)
        elif kind == 'psnr':
            score = 20 * torch.log10(ground_truth.max() - ground_truth.min()) - \
                    10 * torch.log10(F.mse_loss(reconstruction, ground_truth))
        else:
            raise ValueError
        return score

    def compute_new_mask(self, old_mask: torch.Tensor, action: int) -> Tuple[torch.Tensor, bool]:
        """ Computes a new mask by adding the action to the given old mask.

            Note that action is relative to the set of valid k-space lines that can be scanned.
            That is, action = 0 represents the lowest index of k-space lines that are not part of
            the initial mask.
        """
        line_to_scan = self.options.initial_num_lines + action
        new_mask = old_mask.clone().squeeze()
        had_already_been_scanned = bool(new_mask[line_to_scan])
        new_mask[line_to_scan] = 1
        if CONJUGATE_SYMMETRIC:
            new_mask[IMAGE_WIDTH - line_to_scan - 1] = 1
        return new_mask.view(1, 1, 1, -1), had_already_been_scanned

    def compute_score(self,
                      use_reconstruction=True,
                      kind='mse',
                      ground_truth=None,
                      mask_to_use=None):
        """ Computes the score (MSE or SSIM) of current state with respect to current ground truth.

            This method takes the current ground truth, masks it with the current mask and creates
            a zero-filled reconstruction from the masked image. Additionally, this zero-filled
            reconstruction can be passed through the reconstruction network, `self._reconstructor`.
            The score evaluates the difference between the final reconstruction and the current
            ground truth.

            It is possible to pass alternate ground truth and mask to compute the score with
            respect to, instead of `self._ground_truth` and `self._current_mask`.

            @:param use_reconstruction: specifies if the reconstruction network will be used or not.
            @:param ground_truth: specifies if the score has to be computed with respect to an
                alternate "ground truth".
            @:param mask_to_use: specifies if the score has to be computed with an alternate mask.
        """
        with torch.no_grad():
            if ground_truth is None:
                ground_truth = self._ground_truth
            if mask_to_use is None:
                mask_to_use = self._current_mask
            image, _, _ = preprocess_inputs(
                ground_truth, mask_to_use, fft_functions, self.options, clamp_target=False)
            if use_reconstruction:
                image, _, _ = self._reconstructor(
                    image, mask_to_use)  # pass through reconstruction network
        return [
            ReconstructionEnv._compute_score(img.unsqueeze(0), ground_truth, kind) for img in image
        ]

    def _compute_observation_and_score(self) -> Tuple[torch.Tensor, float]:
        with torch.no_grad():
            zero_filled_reconstruction, _, _, masked_rffts = preprocess_inputs(
                self._ground_truth,
                self._current_mask,
                fft_functions,
                self.options,
                return_masked_k_space=True)
            reconstruction, _, mask_embed = self._reconstructor(zero_filled_reconstruction,
                                                                self._current_mask)
            score = ReconstructionEnv._compute_score(reconstruction, self._ground_truth)

            if self.options.rl_obs_type == 'spectral_maps':
                spectral_maps = self.k_space_map(reconstruction, self._current_mask)
                observation = torch.cat([spectral_maps, mask_embed], dim=1)
            elif self.options.rl_obs_type == 'two_streams':
                observation = torch.cat([reconstruction, masked_rffts], dim=1)
            elif self.options.rl_obs_type == 'concatenate_mask':
                observation = torch.cat(
                    [reconstruction,
                     self._current_mask.repeat(1, reconstruction.shape[1], 1, 1)],
                    dim=2)
            else:
                raise ValueError
        return observation.squeeze().cpu().numpy().astype(np.float32), score

    def reset(self) -> Tuple[Union[np.ndarray, None], Dict]:
        """ Loads a new image from the dataset and starts a new episode with this image.

            If `self.options.sequential_images` is True, it loops over images in the dataset in
            order. Otherwise, it selects a random image from the first
            `self.num_{train/test}_images` in the dataset. In the sequential case,
            the dataset is ordered according to `self._{train/test}_order`.
        """
        info = {}
        if self.options.sequential_images:
            if self.is_testing:
                if self.image_idx_test == min(self.num_test_images, len(self._dataset_test)):
                    return None, info  # Returns None to signal that testing is done
                info['split'] = 'test'
                info['image_idx'] = f'{self._test_order[self.image_idx_test]}'
                _, self._ground_truth = self._dataset_test.__getitem__(
                    self._test_order[self.image_idx_test])
                logging.debug(
                    f'Testing episode started with image {self._test_order[self.image_idx_test]}')
                self.image_idx_test += 1
            else:
                info['split'] = 'train'
                info['image_idx'] = f'{self._train_order[self.image_idx_train]}'
                _, self._ground_truth = self._dataset_train.__getitem__(
                    self._train_order[self.image_idx_train])
                logging.debug(
                    f'Train episode started with image {self._train_order[self.image_idx_train]}')
                self.image_idx_train = (self.image_idx_train + 1) % self.num_train_images
        else:
            dataset_to_check = self._dataset_test if self.is_testing else self._dataset_train
            info['split'] = 'test' if self.is_testing else 'train'
            if self.is_testing:
                max_num_images = self.num_test_images
            else:
                max_num_images = self.num_train_images
            dataset_len = min(len(dataset_to_check), max_num_images)
            index_chosen_image = np.random.choice(dataset_len)
            info['image_idx'] = f'{index_chosen_image}'
            logging.debug('{} episode started with randomly chosen image {}/{}'.format(
                'Testing' if self.is_testing else 'Training', index_chosen_image, dataset_len))
            _, self._ground_truth = self._dataset_train.__getitem__(index_chosen_image)
        self._ground_truth = self._ground_truth.to(device).unsqueeze(0)
        self._current_mask = self._initial_mask
        self._scans_left = min(self.options.budget, self.action_space.n)
        observation, self._current_score = self._compute_observation_and_score()
        return observation, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """ Adds a new line (specified by the action) to the current mask and computes the
            resulting observation and reward (drop in MSE after reconstructing with respect to the
            current ground truth).
        """
        assert self._scans_left > 0
        self._current_mask, has_already_been_scanned = self.compute_new_mask(
            self._current_mask, action)
        observation, new_score = self._compute_observation_and_score()

        reward = -1.0 if has_already_been_scanned else (
            self._current_score - new_score).item() / 0.01
        self._current_score = new_score

        self._scans_left -= 1
        done = (self._scans_left == 0)

        return observation, reward, done, {}

    def get_evaluator_action(self) -> int:
        """ Returns the action recommended by the evaluator network of `self._evaluator`. """
        with torch.no_grad():
            image, _, _ = preprocess_inputs(self._ground_truth, self._current_mask, fft_functions,
                                            self.options)
            reconstruction, _, mask_embedding = self._reconstructor(image, self._current_mask)
            k_space_scores = self._evaluator(clamp(reconstruction[:, :1, ...]), mask_embedding)
            k_space_scores.masked_fill_(self._current_mask.squeeze().byte(), 100000)
            return torch.argmin(k_space_scores).item() - NUM_LINES_INITIAL


def generate_initial_mask(num_lines):
    mask = torch.zeros(1, 1, 1, IMAGE_WIDTH)
    for i in range(num_lines):
        mask[0, 0, 0, i] = 1
        if CONJUGATE_SYMMETRIC:
            mask[0, 0, 0, -(i + 1)] = 1
    return mask
