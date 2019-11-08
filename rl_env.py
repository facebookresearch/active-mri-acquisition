import argparse
import logging
import os
from enum import Enum

import gym.spaces
import numpy as np
import torch
import torch.nn.functional as F
from typing import Callable, Dict, List, Optional, Tuple, Union

import data
import models.evaluator
import models.fft_utils
import models.reconstruction
import util.util
import util.rl.reconstructor_rl_trainer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_checkpoint(checkpoints_dir: str, name: str = 'best_checkpoint.pth') -> Optional[Dict]:
    checkpoint_path = os.path.join(checkpoints_dir, name)
    if os.path.isfile(checkpoint_path):
        logging.info(f'Found checkpoint at {checkpoint_path}.')
        return torch.load(checkpoint_path)
    logging.info(f'No checkpoint found at {checkpoint_path}.')
    return None


class ReconstructionEnv:
    """ RL environment representing the active acquisition process with reconstruction model. """

    class DataMode(Enum):
        TRAIN = 1
        TEST = 2
        TEST_ON_TRAIN = 3

    def __init__(self, options: argparse.Namespace):
        """Creates a new environment.

            @param `options`: Should specify the following options:
                -`reconstructor_dir`: directory where reconstructor is stored.
                -`evaluator_dir`: directory where evaluator is stored.
                -`sequential_images`: If true, each episode presents the next image in the dataset,
                    otherwise a random image is presented.
                -`budget`: how many actions to choose (the horizon of the episode).
                -`obs_type`: one of {'fourier_space', 'image_space'}
                -`initial_num_lines_per_side`: how many k-space lines to start with.
                -'num_train_images`: the number of images to use for training. If it's None,
                    then all images will be used.
                -`num_test_images`: the number of images to use for test. If it's None, then
                    all images will be used.
                -`test_set`: the name of the test set to use (train, val, or test).
        """
        self.options = options
        self.options.device = device
        self.image_height = 640 if options.dataroot == 'KNEE_RAW' else 128
        self.image_width = 368 if options.dataroot == 'KNEE_RAW' else 128
        self.conjugate_symmetry = (options.dataroot != 'KNEE_RAW')
        train_loader, valid_loader = data.create_data_loaders(options, is_test=False)
        test_loader = data.create_data_loaders(options, is_test=True)

        self._dataset_train = train_loader.dataset
        self.split_names = {
            ReconstructionEnv.DataMode.TRAIN: 'train',
            ReconstructionEnv.DataMode.TEST: 'test',
            ReconstructionEnv.DataMode.TEST_ON_TRAIN: 'test_on_train'
        }
        if options.test_set == 'train':
            self._dataset_test = train_loader.dataset
        elif options.test_set == 'val':
            self._dataset_test = valid_loader.dataset
        elif options.test_set == 'test':
            self._dataset_test = test_loader.dataset
        else:
            raise ValueError('Valid options are train, val, test')

        self.num_train_images = min(self.options.num_train_images, len(self._dataset_train))
        self.num_test_images = min(self.options.num_test_images, len(self._dataset_test))
        self.latest_train_images = []

        # For the training data the
        rng = np.random.RandomState(options.seed)
        rng_train = np.random.RandomState() if options.rl_env_train_no_seed else rng
        self._train_order = rng_train.permutation(len(self._dataset_train))
        self._test_order = rng.permutation(len(self._dataset_test))
        self._image_idx_test = 0
        self._image_idx_train = -1
        self.data_model = ReconstructionEnv.DataMode.TRAIN

        reconstructor_checkpoint = load_checkpoint(options.reconstructor_dir, 'best_checkpoint.pth')
        self._reconstructor = models.reconstruction.ReconstructorNetwork(
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
            # TODO: this is true only in case of single gpu:
            key.replace('module.', ''): val
            for key, val in reconstructor_checkpoint['reconstructor'].items()
        })
        self._reconstructor.eval()
        self._reconstructor.to(device)
        logging.info('Loaded reconstructor from checkpoint.')

        self._evaluator = None
        evaluator_checkpoint = None
        if options.evaluator_dir is not None:
            evaluator_checkpoint = load_checkpoint(options.evaluator_dir, 'best_checkpoint.pth')
        if evaluator_checkpoint is not None and evaluator_checkpoint['evaluator'] is not None:
            self._evaluator = models.evaluator.EvaluatorNetwork(
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
            self._evaluator.eval()
            self._evaluator.to(device)

        self.observation_space = None  # The observation is a dict unless `obs_to_numpy` is used
        if self.options.obs_to_numpy:
            # The extra rows represents the current mask and the mask embedding
            obs_shape = (2, self.image_height + 2, self.image_width)
            self.observation_space = gym.spaces.Box(low=-50000, high=50000, shape=obs_shape)

        self.metadata = {'mask_embed_dim': reconstructor_checkpoint['options'].mask_embed_dim}

        factor = 2 if self.conjugate_symmetry else 1
        num_actions = (self.image_width - 2 * options.initial_num_lines_per_side) // factor
        self.action_space = gym.spaces.Discrete(num_actions)

        self._initial_mask = self._generate_initial_mask().to(device)
        self._ground_truth = None
        self._k_space = None
        self._current_mask = None
        self._current_score = None
        self._scans_left = None

        # Variables used when alternate optimization is active
        self.mask_dict = {}
        self.epoch_count_callback = None
        self.epoch_frequency_callback = None
        self.reconstructor_trainer = util.rl.reconstructor_rl_trainer.ReconstructorRLTrainer(
            self._reconstructor, self._dataset_train, self.options)

    def _generate_initial_mask(self):
        mask = torch.zeros(1, 1, 1, self.image_width)
        for i in range(self.options.initial_num_lines_per_side):
            mask[0, 0, 0, i] = 1
            mask[0, 0, 0, -(i + 1)] = 1
        return mask

    def _get_current_reconstruction_and_mask_embedding(self) -> Tuple[torch.Tensor, torch.Tensor]:
        zero_filled_image, _, _ = models.fft_utils.preprocess_inputs(
            (self._current_mask, self._ground_truth, self._k_space), self.options.dataroot, device)

        if self.options.use_reconstructions:
            reconstruction, _, mask_embed = self._reconstructor(zero_filled_image,
                                                                self._current_mask)
        else:
            reconstruction = zero_filled_image
            mask_embed = None

        return reconstruction, mask_embed

    def set_testing(self, use_training_set=False, reset_index=True):
        self.data_model = ReconstructionEnv.DataMode.TEST_ON_TRAIN if use_training_set \
            else ReconstructionEnv.DataMode.TEST
        if reset_index:
            self._image_idx_test = 0

    def set_training(self, reset_index=False):
        self.data_model = ReconstructionEnv.DataMode.TRAIN
        if reset_index:
            self._image_idx_train = -1

    @staticmethod
    def _compute_score(reconstruction: torch.Tensor, ground_truth: torch.Tensor,
                       is_raw: bool) -> Dict[str, torch.Tensor]:
        # Compute magnitude (for metrics)
        reconstruction = models.fft_utils.to_magnitude(reconstruction)
        ground_truth = models.fft_utils.to_magnitude(ground_truth)
        if is_raw:  # crop data
            reconstruction = models.fft_utils.center_crop(reconstruction, [320, 320])
            ground_truth = models.fft_utils.center_crop(ground_truth, [320, 320])

        mse = F.mse_loss(reconstruction, ground_truth)
        ssim = util.util.ssim_metric(reconstruction, ground_truth)
        psnr = util.util.psnr_metric(reconstruction, ground_truth)

        score = {'mse': mse, 'ssim': ssim, 'psnr': psnr}

        return score

    def compute_new_mask(self, old_mask: torch.Tensor, action: int) -> Tuple[torch.Tensor, bool]:
        """ Computes a new mask by adding the action to the given old mask.

            Note that action is relative to the set of valid k-space lines that can be scanned.
            That is, action = 0 represents the lowest index of k-space lines that are not part of
            the initial mask.
        """
        line_to_scan = self.options.initial_num_lines_per_side + action
        new_mask = old_mask.clone().squeeze()
        had_already_been_scanned = bool(new_mask[line_to_scan])
        new_mask[line_to_scan] = 1
        if self.conjugate_symmetry:
            new_mask[-(line_to_scan + 1)] = 1
        return new_mask.view(1, 1, 1, -1), had_already_been_scanned

    def compute_score(self,
                      use_reconstruction: bool = True,
                      ground_truth: Optional[torch.Tensor] = None,
                      mask_to_use: Optional[torch.Tensor] = None,
                      k_space: Optional[torch.Tensor] = None,
                      use_current_score: bool = False) -> List[Dict[str, torch.Tensor]]:
        """ Computes the score (MSE or SSIM) of current state with respect to current ground truth.

            This method takes the current ground truth, masks it with the current mask and creates
            a zero-filled reconstruction from the masked image. Additionally, this zero-filled
            reconstruction can be passed through the reconstruction network, `self._reconstructor`.
            The score evaluates the difference between the final reconstruction and the current
            ground truth.

            It is possible to pass alternate ground truth and mask to compute the score with
            respect to, instead of `self._ground_truth` and `self._current_mask`.

            @:param `use_reconstruction`: specifies if the reconstruction network will be used.
            @:param `ground_truth`: specifies if the score has to be computed with respect to an
                alternate "ground truth".
            @:param `mask_to_use`: specifies if the score has to be computed with an alternate mask.
            @:param `k_space`: specifies if the score has to be computed with an alternate k-space.
            @:param `use_current_score`: If true, the method returns the saved current score.
        """
        if use_current_score and use_reconstruction:
            return [self._current_score]
        with torch.no_grad():
            if ground_truth is None:
                ground_truth = self._ground_truth
            if mask_to_use is None:
                mask_to_use = self._current_mask
            if k_space is None:
                k_space = self._k_space
            image, _, _ = models.fft_utils.preprocess_inputs((mask_to_use, ground_truth, k_space),
                                                             self.options.dataroot, device)
            if use_reconstruction:  # pass through reconstruction network
                image, _, _ = self._reconstructor(image, mask_to_use)
        return [
            ReconstructionEnv._compute_score(
                img.unsqueeze(0), ground_truth, self.options.dataroot == 'KNEE_RAW')
            for img in image
        ]

    def _compute_observation_and_score(self) -> Tuple[Union[Dict, np.ndarray], Dict]:
        with torch.no_grad():
            reconstruction, mask_embedding = self._get_current_reconstruction_and_mask_embedding()
            score = ReconstructionEnv._compute_score(reconstruction, self._ground_truth,
                                                     self.options.dataroot == 'KNEE_RAW')

            if self.options.obs_type == 'fourier_space':
                reconstruction = models.fft_utils.fft(reconstruction)

            observation = {
                'reconstruction': reconstruction,
                'mask': self._current_mask,
                'mask_embedding': mask_embedding
            }

            if self.options.obs_to_numpy:
                observation = np.zeros(self.observation_space.shape).astype(np.float32)
                observation[:2, :self.image_height, :] = reconstruction[0].cpu().numpy()
                # The second to last row is the mask
                observation[:, self.image_height, :] = self._current_mask.cpu().numpy()
                # The last row is the mask embedding (padded with 0s if necessary)
                observation[:, self.image_height + 1, :self.metadata['mask_embed_dim']] = \
                    mask_embedding[0, :, 0, 0].cpu().numpy()

        return observation, score

    def reset(self) -> Tuple[Union[Dict, None], Dict]:
        """ Loads a new image from the dataset and starts a new episode with this image.

            If `self.options.sequential_images` is True, it loops over images in the dataset in
            order. Otherwise, it selects a random image from the first
            `self.num_{train/test}_images` in the dataset. In the sequential case,
            the dataset is ordered according to `self._{train/test}_order`.
        """
        info = {}
        if self.options.sequential_images:
            if self.data_model == ReconstructionEnv.DataMode.TEST:
                if self._image_idx_test == min(self.num_test_images, len(self._dataset_test)):
                    return None, info  # Returns None to signal that testing is done
                set_order = self._test_order
                dataset = self._dataset_test
                set_idx = self._image_idx_test
                self._image_idx_test += 1
            else:
                set_order = self._train_order
                dataset = self._dataset_train
                if self.data_model == ReconstructionEnv.DataMode.TEST_ON_TRAIN:
                    if self._image_idx_test == min(self.num_train_images, len(self._dataset_train)):
                        return None, info  # Returns None to signal that testing is done
                    set_idx = self._image_idx_test
                    self._image_idx_test += 1
                else:
                    if self.epoch_count_callback is not None:
                        if (self._image_idx_train + 1) % self.epoch_frequency_callback == 0:
                            self.epoch_count_callback()

                    self._image_idx_train = (self._image_idx_train + 1) % self.num_train_images

                    if self._image_idx_train == 0:
                        self._reset_saved_masks_dict()

                    set_idx = self._image_idx_train

            info['split'] = self.split_names[self.data_model]
            info['image_idx'] = set_order[set_idx]
            tmp = dataset.__getitem__(info['image_idx'])
            self._ground_truth = tmp[1]
            if self.options.dataroot == 'KNEE_RAW':  # store k-space data too
                self._k_space = tmp[2].unsqueeze(0)
                self._ground_truth = self._ground_truth.permute(2, 0, 1)
            logging.debug(f"{info['split'].capitalize()} episode started "
                          f"with image {set_order[set_idx]}")
        else:
            using_test_set = self.data_model == ReconstructionEnv.DataMode.TEST
            dataset_to_check = self._dataset_test if using_test_set else self._dataset_train
            info['split'] = 'test' if using_test_set else 'train'
            if using_test_set:
                max_num_images = self.num_test_images
            else:
                max_num_images = self.num_train_images
            dataset_len = min(len(dataset_to_check), max_num_images)
            index_chosen_image = np.random.choice(dataset_len)
            info['image_idx'] = index_chosen_image
            logging.debug('{} episode started with randomly chosen image {}/{}'.format(
                'Testing' if using_test_set else 'Training', index_chosen_image, dataset_len))
            _, self._ground_truth = self._dataset_train.__getitem__(index_chosen_image)
        self._ground_truth = self._ground_truth.to(device).unsqueeze(0)
        self._current_mask = self._initial_mask
        self._scans_left = min(self.options.budget, self.action_space.n)
        observation, score = self._compute_observation_and_score()
        self._current_score = score
        return observation, info

    def step(self, action: int) -> Tuple[Dict[str, torch.Tensor], float, bool, Dict]:
        """ Adds a new line (specified by the action) to the current mask and computes the
            resulting observation and reward (drop in MSE after reconstructing with respect to the
            current ground truth).
        """
        assert self._scans_left > 0
        self._current_mask, has_already_been_scanned = self.compute_new_mask(
            self._current_mask, action)
        if self.data_model == ReconstructionEnv.DataMode.TRAIN:
            self.mask_dict[self._train_order[self._image_idx_train]][-self._scans_left] = \
                self._current_mask.squeeze().cpu().numpy()
        observation, new_score = self._compute_observation_and_score()

        metric = self.options.reward_metric
        reward_ = new_score[metric] if self.options.use_score_as_reward \
            else new_score[metric] - self._current_score[metric]
        factor = 1 if self.options.use_score_as_reward else 100
        if self.options.reward_metric == 'mse':
            factor *= -1  # We try to minimize MSE, but DQN maximizes
        reward = -1.0 if has_already_been_scanned else reward_.item() * factor
        self._current_score = new_score

        self._scans_left -= 1
        done = (self._scans_left == 0)

        return observation, reward, done, {}

    def get_evaluator_action(self, obs: Dict[str, torch.Tensor]) -> int:
        """ Returns the action recommended by the evaluator network of `self._evaluator`. """
        with torch.no_grad():
            assert not self.options.obs_type == 'fourier_space' and not self.options.obs_to_numpy
            k_space_scores = self._evaluator(obs['reconstruction'].to(device),
                                             obs['mask_embedding'].to(device))
            k_space_scores.masked_fill_(obs['mask'].to(device).squeeze().byte(), 100000)
            return torch.argmin(k_space_scores).item() - self.options.initial_num_lines_per_side

    def retrain_reconstructor(self, logger, writer):
        logger.info(
            f'Training reconstructor for {self.options.num_epochs_train_reconstructor} epochs')
        self.reconstructor_trainer(self.mask_dict, writer)
        logger.info('Done training reconstructor')

        self._reset_saved_masks_dict()  # Reset mask dictionary for next epoch

    def _reset_saved_masks_dict(self):
        self.mask_dict.clear()
        for image_index in self._train_order[:self.options.num_train_images]:
            self.mask_dict[image_index] = np.zeros(
                (self.options.budget, self.image_width), dtype=np.float32)

    def set_epoch_finished_callback(self, callback: Callable, frequency: int):
        # Every frequency number of epochs, the callback function will be called
        self.epoch_count_callback = callback
        self.epoch_frequency_callback = frequency
