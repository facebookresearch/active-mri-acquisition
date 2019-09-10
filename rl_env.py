import logging
import os

import gym.spaces
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union

from data import create_data_loaders
from models.evaluator import EvaluatorNetwork
from models.fft_utils import RFFT, IFFT, FFT, preprocess_inputs, to_magnitude, center_crop
from models.reconstruction import ReconstructorNetwork
from util import util

rfft = RFFT()
ifft = IFFT()
fft = FFT()
fft_functions = {'rfft': rfft, 'ifft': ifft, 'fft': fft}

CONJUGATE_SYMMETRIC = True

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# TODO Organize imports and finish adding type info


def load_checkpoint(checkpoints_dir, name='best_checkpoint.pth'):
    checkpoint_path = os.path.join(checkpoints_dir, name)
    if os.path.isfile(checkpoint_path):
        logging.info(f'Found checkpoint at {checkpoint_path}.')
        return torch.load(checkpoint_path)
    logging.info(f'No checkpoint found at {checkpoint_path}.')
    return None


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
                -`reconstructor_dir`: directory where reconstructor is stored.
                -`evaluator_dir`: directory where evaluator is stored.
                -`sequential_images`: If true, each episode presents the next image in the dataset,
                    otherwise a random image is presented.
                -`budget`: how many actions to choose (the horizon of the episode).
                -`obs_type`: one of {'fourier_space', 'image_space'}
                -`initial_num_lines`: how many k-space lines to start with.
                -'num_train_images`: the number of images to use for training. If it's None,
                    then all images will be used.
                -`num_test_images`: the number of images to use for test. If it's None, then
                    all images will be used.
                -`test_set`: the name of the test set to use (train, val, or test).
        """
        # TODO remove initial_mask argument (control this generation inside the class)

        self.options = options
        self.options.device = device
        self.image_height = 640 if options.dataroot == 'KNEE_RAW' else 128
        self.image_width = 368 if options.dataroot == 'KNEE_RAW' else 128
        train_loader, valid_loader = create_data_loaders(options, is_test=False)
        test_loader = create_data_loaders(options, is_test=True)

        self._dataset_train = train_loader.dataset
        if options.test_set == 'train':
            self._dataset_test = train_loader.dataset
        elif options.test_set == 'valid':
            self._dataset_test = valid_loader.dataset
        else:
            self._dataset_test = test_loader.dataset

        self.num_train_images = min(self.options.num_train_images, len(self._dataset_train))
        self.num_test_images = min(self.options.num_test_images, len(self._dataset_test))

        # For the training data the
        rng = np.random.RandomState(options.seed)
        rng_train = np.random.RandomState() if options.rl_env_train_no_seed else rng
        self._train_order = rng_train.permutation(len(self._dataset_train))
        self._test_order = rng.permutation(len(self._dataset_test))
        self._image_idx_test = 0
        self._image_idx_train = 0
        self.is_testing = False

        reconstructor_checkpoint = load_checkpoint(options.reconstructor_dir, 'best_checkpoint.pth')
        self._reconstructor = ReconstructorNetwork(
            number_of_cascade_blocks=reconstructor_checkpoint['options'].number_of_cascade_blocks,
            n_downsampling=reconstructor_checkpoint['options'].n_downsampling,
            number_of_filters=reconstructor_checkpoint['options'].number_of_reconstructor_filters,
            number_of_layers_residual_bottleneck=reconstructor_checkpoint['options']
            .number_of_layers_residual_bottleneck,
            mask_embed_dim=reconstructor_checkpoint['options'].mask_embed_dim,
            dropout_probability=reconstructor_checkpoint['options'].dropout_probability,
            img_width=self.image_width,
            use_deconv=reconstructor_checkpoint['options'].use_deconv)
        self._reconstructor.load_state_dict({
            key.replace('module.', ''): val
            for key, val in reconstructor_checkpoint['reconstructor'].items()
        })
        self._reconstructor.to(device)
        logging.info('Loaded reconstructor from checkpoint.')

        self._evaluator = None
        evaluator_checkpoint = load_checkpoint(options.evaluator_dir, 'best_checkpoint.pth')
        if evaluator_checkpoint is not None and evaluator_checkpoint['evaluator'] is not None:
            self._evaluator = EvaluatorNetwork(
                number_of_filters=evaluator_checkpoint['options'].number_of_evaluator_filters,
                number_of_conv_layers=evaluator_checkpoint['options']
                .number_of_evaluator_convolution_layers,
                use_sigmoid=False,
                width=evaluator_checkpoint['options'].image_width,
                mask_embed_dim=evaluator_checkpoint['options'].mask_embed_dim)
            logging.info(f'Loaded evaluator from checkpoint.')
            self._evaluator.load_state_dict({
                key.replace('module.', ''): val
                for key, val in evaluator_checkpoint['evaluator'].items()
            })
            self._evaluator.to(device)

        # The extra row represents the current mask
        obs_shape = (2, self.image_height + 1, self.image_width)
        self.observation_space = gym.spaces.Box(low=-50000, high=50000, shape=obs_shape)

        factor = 2 if CONJUGATE_SYMMETRIC else 1
        num_actions = (self.image_width - factor * options.initial_num_lines) // 2
        self.action_space = gym.spaces.Discrete(num_actions)

        self._ground_truth = None
        self._initial_mask = initial_mask.to(device)
        self.k_space_map = KSpaceMap(img_width=self.image_width).to(device)

    def set_testing(self, reset_index=True):
        self.is_testing = True
        if reset_index:
            self._image_idx_test = 0

    def set_training(self, reset_index=False):
        self.is_testing = False
        if reset_index:
            self._image_idx_train = 0

    @staticmethod
    def _compute_score(self,
                       reconstruction: torch.Tensor,
                       ground_truth: torch.Tensor,
                       kind: str = 'mse') -> torch.Tensor:

        # Compute magnitude (for metrics)
        also_clamp_and_scale = self.options.dataroot != 'KNEE_RAW'
        reconstruction = to_magnitude(reconstruction, also_clamp_and_scale=also_clamp_and_scale)
        ground_truth = to_magnitude(ground_truth, also_clamp_and_scale=also_clamp_and_scale)
        if self.options.dataroot == 'KNEE_RAW':  # crop data
            reconstruction = center_crop(reconstruction, [320, 320])
            ground_truth = center_crop(ground_truth, [320, 320])

        # reconstruction = reconstruction[:, :1, ...]
        # ground_truth = ground_truth[:, :1, ...]
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
            new_mask[self.image_width - line_to_scan - 1] = 1
        return new_mask.view(1, 1, 1, -1), had_already_been_scanned

    def compute_score(
            self,
            use_reconstruction: bool = True,
            kind: str = 'mse',
            ground_truth: Optional[torch.Tensor] = None,
            mask_to_use: Optional[torch.Tensor] = None,
            kspace: Optional[torch.Tensor] = None,
    ) -> List[torch.Tensor]:
        """ Computes the score (MSE or SSIM) of current state with respect to current ground truth.

            This method takes the current ground truth, masks it with the current mask and creates
            a zero-filled reconstruction from the masked image. Additionally, this zero-filled
            reconstruction can be passed through the reconstruction network, `self._reconstructor`.
            The score evaluates the difference between the final reconstruction and the current
            ground truth.

            It is possible to pass alternate ground truth and mask to compute the score with
            respect to, instead of `self._ground_truth` and `self._current_mask`.

            @:param `use_reconstruction`: specifies if the reconstruction network will be used.
            @:param `kind`: specifies what the score function is ('mse', 'ssim', 'psnr')
            @:param `ground_truth`: specifies if the score has to be computed with respect to an
                alternate "ground truth".
            @:param `mask_to_use`: specifies if the score has to be computed with an alternate mask.
        """
        with torch.no_grad():
            if ground_truth is None:
                ground_truth = self._ground_truth
            if mask_to_use is None:
                mask_to_use = self._current_mask

            if self.options.dataroot != 'KNEE_RAW':
                image, _, _ = preprocess_inputs(
                    (mask_to_use, ground_truth), fft_functions, self.options, clamp_target=False)
            else:
                if kspace is None:
                    kspace = self._k_space
                image, _, _ = preprocess_inputs(
                    (mask_to_use, ground_truth, kspace),
                    fft_functions,
                    self.options,
                    clamp_target=False)
            if use_reconstruction:
                image, _, _ = self._reconstructor(
                    image, mask_to_use)  # pass through reconstruction network
        return [
            ReconstructionEnv._compute_score(self, img.unsqueeze(0), ground_truth, kind)
            for img in image
        ]

    def _compute_observation_and_score(self) -> Tuple[torch.Tensor, float]:
        with torch.no_grad():
            if self.options.dataroot != 'KNEE_RAW':
                zero_filled_reconstruction, _, _, masked_rffts = preprocess_inputs(
                    (self._current_mask, self._ground_truth),
                    fft_functions,
                    self.options,
                    return_masked_k_space=True)
            else:
                zero_filled_reconstruction, _, _, masked_rffts = preprocess_inputs(
                    (self._current_mask, self._ground_truth, self._k_space),
                    fft_functions,
                    self.options,
                    return_masked_k_space=True)
            reconstruction, _, mask_embed = self._reconstructor(zero_filled_reconstruction,
                                                                self._current_mask)
            score = ReconstructionEnv._compute_score(self, reconstruction, self._ground_truth)

            # TODO add if for fourier/image
            observation = torch.cat(
                [reconstruction,
                 self._current_mask.repeat(1, reconstruction.shape[1], 1, 1)],
                dim=2)
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
                if self._image_idx_test == min(self.num_test_images, len(self._dataset_test)):
                    return None, info  # Returns None to signal that testing is done
                info['split'] = 'test'
                info['image_idx'] = self._test_order[self._image_idx_test]
                tmp = self._dataset_test.__getitem__(info['image_idx'])
                self._ground_truth = tmp[1]
                if self.options.dataroot == 'KNEE_RAW':
                    # store k-space data too
                    self._k_space = tmp[2].unsqueeze(0)
                    self._ground_truth = self._ground_truth.permute(2, 0, 1)
                logging.debug(
                    f'Testing episode started with image {self._test_order[self._image_idx_test]}')
                self._image_idx_test += 1
            else:
                info['split'] = 'train'
                info['image_idx'] = self._train_order[self._image_idx_train]
                tmp = self._dataset_train.__getitem__(info['image_idx'])
                self._ground_truth = tmp[1]
                if self.options.dataroot == 'KNEE_RAW':
                    # store k-space data too
                    self._k_space = tmp[2].unsqueeze(0)
                    self._ground_truth = self._ground_truth.permute(2, 0, 1)
                logging.debug(
                    f'Train episode started with image {self._train_order[self._image_idx_train]}')
                self._image_idx_train = (self._image_idx_train + 1) % self.num_train_images
        else:
            dataset_to_check = self._dataset_test if self.is_testing else self._dataset_train
            info['split'] = 'test' if self.is_testing else 'train'
            if self.is_testing:
                max_num_images = self.num_test_images
            else:
                max_num_images = self.num_train_images
            dataset_len = min(len(dataset_to_check), max_num_images)
            index_chosen_image = np.random.choice(dataset_len)
            info['image_idx'] = index_chosen_image
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

        reward_ = -new_score if self.options.use_score_as_reward \
            else self._current_score - new_score
        factor = 1 if self.options.use_score_as_reward else 100
        reward = -1.0 if has_already_been_scanned else reward_.item() * factor
        self._current_score = new_score

        self._scans_left -= 1
        done = (self._scans_left == 0)

        return observation, reward, done, {}

    def get_evaluator_action(self) -> int:
        """ Returns the action recommended by the evaluator network of `self._evaluator`. """
        with torch.no_grad():
            if self.options.dataroot != 'KNEE_RAW':
                image, _, _ = preprocess_inputs((self._current_mask, self._ground_truth),
                                                fft_functions, self.options)
            else:
                image, _, _ = preprocess_inputs(
                    (self._current_mask, self._ground_truth, self._k_space), fft_functions,
                    self.options)

            reconstruction, _, mask_embedding = self._reconstructor(image, self._current_mask)
            k_space_scores = self._evaluator(reconstruction, mask_embedding)
            k_space_scores.masked_fill_(self._current_mask.squeeze().byte(), 100000)
            return torch.argmin(k_space_scores).item() - self.options.initial_num_lines


def generate_initial_mask(num_lines, options):
    mask = torch.zeros(1, 1, 1, 368 if options.dataroot == 'KNEE_RAW' else 128)
    for i in range(num_lines):
        mask[0, 0, 0, i] = 1
        if CONJUGATE_SYMMETRIC:
            mask[0, 0, 0, -(i + 1)] = 1
    return mask
