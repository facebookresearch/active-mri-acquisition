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
import util.rl.utils

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


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

        self.options.rnl_params = f'{2 * self.options.initial_num_lines_per_side},' \
            f'{2 * (self.options.initial_num_lines_per_side + 1)},1,5'

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

        self.rng = np.random.RandomState(options.seed)
        rng_train = np.random.RandomState() if options.rl_env_train_no_seed else self.rng
        self._train_order = rng_train.permutation(len(self._dataset_train))
        self._test_order = self.rng.permutation(len(self._dataset_test))
        self._image_idx_test = 0
        self._image_idx_train = 0
        self.data_mode = ReconstructionEnv.DataMode.TRAIN

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
        logging.info('Loaded reconstructor and original options from checkpoint.')
        logging.info('Checking if new weights are available from alternate optimization steps.')
        reconstructor_alt_opt_checkpoint = load_checkpoint(self.options.checkpoints_dir,
                                                           'best_alt_opt_reconstructor.pt')
        self._start_epoch_for_alt_opt = 0
        if reconstructor_alt_opt_checkpoint is not None:
            logging.info('Found a more recent reconstructor from alternate optimization.')
            self._reconstructor.load_state_dict((reconstructor_alt_opt_checkpoint['state_dict']))
            self._start_epoch_for_alt_opt = reconstructor_alt_opt_checkpoint['epoch']
            logging.info(
                f'Start epoch for alternate optimization set to {self._start_epoch_for_alt_opt}.')

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
                height=640 if options.dataroot == 'KNEE_RAW' else None,
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
            if self.options.obs_type == 'only_mask':
                obs_shape = (self.image_width,)
            else:
                # The extra rows represents the current mask and the mask embedding
                obs_shape = (2, self.image_height + 2, self.image_width)
            self.observation_space = gym.spaces.Box(low=-50000, high=50000, shape=obs_shape)

        self.metadata = {'mask_embed_dim': reconstructor_checkpoint['options'].mask_embed_dim}

        # Setting up valid actions
        factor = 2 if self.conjugate_symmetry else 1
        num_actions = (self.image_width - 2 * options.initial_num_lines_per_side) // factor
        self.valid_actions = list(range(num_actions))
        if self.options.dataroot == 'KNEE_RAW':
            # use invalid_actions when using k_space knee data that are zero padded
            # at some frequencies
            # TODO: do we want to change the fixed numbers below or not?
            invalid_actions = list(
                range(166 - self.options.initial_num_lines_per_side,
                      202 - self.options.initial_num_lines_per_side, 1))
            self.valid_actions = np.setdiff1d(self.valid_actions, invalid_actions)
        self.action_space = gym.spaces.Discrete(len(self.valid_actions))

        self._initial_mask = self._generate_initial_lowf_mask().to(device)
        self._ground_truth = None
        self._k_space = None
        self._current_mask = None
        self._current_score = None
        self._initial_score_episode = None
        self._scans_left = None
        self._reference_mean_for_reward = None
        self._reference_std_for_reward = None
        self._last_action = None

        # Variables used when alternate optimization is active
        self.mask_dict = {}
        self.epoch_count_callback = None
        self.epoch_frequency_callback = None
        self.reconstructor_trainer = util.rl.reconstructor_rl_trainer.ReconstructorRLTrainer(
            self._reconstructor, self._dataset_train, self.options,
            self.update_reconstructor_from_alt_opt)

        # Pre-compute reward normalization if necessary
        if options.normalize_rewards_on_val:
            logging.info('Running random policy to get reference point for reward.')
            random_policy = util.rl.simple_baselines.RandomPolicy(range(self.action_space.n))
            self.set_testing()
            _, statistics = util.rl.utils.test_policy(
                self, random_policy, None, None, 0, self.options, leave_no_trace=True)
            logging.info('Done computing reference.')
            self.set_reference_point_for_rewards(statistics)

    def _generate_initial_lowf_mask(self):
        mask = torch.zeros(1, 1, 1, self.image_width)
        for i in range(self.options.initial_num_lines_per_side):
            mask[0, 0, 0, i] = 1
            mask[0, 0, 0, -(i + 1)] = 1
        return mask

    def _get_current_reconstruction_and_mask_embedding(self) \
            -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        zero_filled_image, _, mask_to_use = models.fft_utils.preprocess_inputs(
            (self._current_mask, self._ground_truth, self._k_space), self.options.dataroot, device)

        if self.options.use_reconstructions:
            reconstruction, _, mask_embed = self._reconstructor(zero_filled_image, mask_to_use)
        else:
            reconstruction = zero_filled_image
            mask_embed = None

        return reconstruction, mask_embed, mask_to_use

    def set_testing(self, use_training_set=False, reset_index=True):
        self.data_mode = ReconstructionEnv.DataMode.TEST_ON_TRAIN if use_training_set \
            else ReconstructionEnv.DataMode.TEST
        if reset_index:
            self._image_idx_test = 0

    def set_training(self, reset_index=False):
        self.data_mode = ReconstructionEnv.DataMode.TRAIN
        if reset_index:
            self._image_idx_train = 0

    @staticmethod
    def _compute_score(reconstruction: torch.Tensor, ground_truth: torch.Tensor,
                       is_raw: bool) -> Dict[str, torch.Tensor]:
        # Compute magnitude (for metrics)
        reconstruction = models.fft_utils.to_magnitude(reconstruction)
        ground_truth = models.fft_utils.to_magnitude(ground_truth)
        if is_raw:  # crop data
            reconstruction = models.fft_utils.center_crop(reconstruction, [320, 320])
            ground_truth = models.fft_utils.center_crop(ground_truth, [320, 320])

        mse = F.mse_loss(reconstruction, ground_truth).cpu()
        ssim = util.util.ssim_metric(reconstruction, ground_truth)
        psnr = util.util.psnr_metric(reconstruction, ground_truth)

        score = {'mse': mse, 'ssim': ssim, 'psnr': psnr}

        return score

    def compute_new_mask(self, old_mask: torch.Tensor, action: int,
                         reverse=False) -> Tuple[torch.Tensor, bool]:
        """ Computes a new mask by adding the action to the given old mask.

            Note that action is relative to the set of valid k-space lines that can be scanned.
            That is, action = 0 represents the lowest index of k-space lines that are not part of
            the initial mask.

            If `reverse` is True, then the action is removed rather than added.
        """
        line_to_scan = self.options.initial_num_lines_per_side + action
        new_mask = old_mask.clone().squeeze()
        had_already_been_scanned = bool(new_mask[line_to_scan])
        bool_value = 0 if reverse else 1
        new_mask[line_to_scan] = bool_value
        if self.conjugate_symmetry:
            new_mask[-(line_to_scan + 1)] = bool_value
        return new_mask.view(1, 1, 1, -1), had_already_been_scanned

    def compute_score(self,
                      use_reconstruction: bool = True,
                      ground_truth: Optional[torch.Tensor] = None,
                      mask_to_use: Optional[torch.Tensor] = None,
                      k_space: Optional[torch.Tensor] = None,
                      use_current_score: bool = False,
                      use_zz_score: bool = False) -> List[Dict[str, torch.Tensor]]:
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
        if use_zz_score and self._last_action is not None:
            return self._compute_zz_score()
        if use_current_score and use_reconstruction:
            return [self._current_score]
        with torch.no_grad():
            if ground_truth is None:
                ground_truth = self._ground_truth
            if mask_to_use is None:
                mask_to_use = self._current_mask
            if k_space is None:
                k_space = self._k_space
            image, _, mask_to_use = models.fft_utils.preprocess_inputs(
                (mask_to_use, ground_truth, k_space), self.options.dataroot, device)
            if use_reconstruction:  # pass through reconstruction network
                image, _, _ = self._reconstructor(image, mask_to_use)
        return [
            ReconstructionEnv._compute_score(
                img.unsqueeze(0), ground_truth, self.options.dataroot == 'KNEE_RAW')
            for img in image
        ]

    def _compute_zz_score(self) -> List[Dict[str, torch.Tensor]]:
        """ Evaluates the score they way it was done for CVPR'19. """
        with torch.no_grad():
            # This method uses the reconstruction at the point just before the action was taken,
            # which is computed below
            mask_before_action, _ = self.compute_new_mask(
                self._current_mask, self._last_action, reverse=True)
            zero_filled_image, _, mask_to_use = models.fft_utils.preprocess_inputs(
                (mask_before_action, self._ground_truth, self._k_space), self.options.dataroot,
                device)
            reconstruction_before_action, _, _ = self._reconstructor(zero_filled_image, mask_to_use)
            ft_reconstruction = models.fft_utils.fft(reconstruction_before_action)
            ft_gt = models.fft_utils.fft(self._ground_truth)

            # Now we need the effect of adding the gt scan, so we use the mask as it was before
            # removing the action (i.e., `self.current_mask`)
            reconstr_plus_gt_col = models.fft_utils.ifft(
                torch.where(self._current_mask.byte(), ft_gt, ft_reconstruction))
            return [
                ReconstructionEnv._compute_score(
                    img.unsqueeze(0), self._ground_truth, self.options.dataroot == 'KNEE_RAW')
                for img in reconstr_plus_gt_col
            ]

    def _compute_observation_and_score(self) -> Tuple[Union[Dict, np.ndarray], Dict]:
        with torch.no_grad():
            reconstruction, mask_embedding, mask = \
                self._get_current_reconstruction_and_mask_embedding()
            score = ReconstructionEnv._compute_score(reconstruction, self._ground_truth,
                                                     self.options.dataroot == 'KNEE_RAW')

            if self.options.obs_type == 'only_mask':
                observation = {'mask': self._current_mask}
                if self.options.obs_to_numpy:
                    observation = self._current_mask.squeeze().cpu().numpy()
                return observation, score

            if self.options.obs_type == 'fourier_space':
                reconstruction = models.fft_utils.fft(reconstruction)

            observation = {
                'reconstruction': reconstruction,
                'mask': mask,
                'mask_embedding': mask_embedding
            }

            if self.options.obs_to_numpy:
                observation = np.zeros(self.observation_space.shape).astype(np.float32)
                observation[:2, :self.image_height, :] = reconstruction[0].cpu().numpy()
                # The second to last row is the mask
                observation[:, self.image_height, :] = mask.cpu().numpy()
                # The last row is the mask embedding (padded with 0s if necessary)
                if self.metadata['mask_embed_dim'] == 0:
                    observation[:, self.image_height + 1, 0] = np.nan
                else:
                    observation[:, self.image_height + 1, :self.metadata['mask_embed_dim']] = \
                        mask_embedding[0, :, 0, 0].cpu().numpy()

        return observation, score

    def reset(self, start_with_initial_mask=False) -> Tuple[Union[Dict, None], Dict]:
        """ Loads a new image from the dataset and starts a new episode with this image.

            Loops over images in the dataset in order. The dataset is ordered according to
            `self._{train/test}_order`.
        """
        info = {}
        self._last_action = None
        if self.data_mode == ReconstructionEnv.DataMode.TEST:
            if self._image_idx_test == min(self.num_test_images, len(self._dataset_test)):
                return None, info  # Returns None to signal that testing is done
            set_order = self._test_order
            dataset = self._dataset_test
            set_idx = self._image_idx_test
            self._image_idx_test += 1
        else:
            set_order = self._train_order
            dataset = self._dataset_train
            if self.data_mode == ReconstructionEnv.DataMode.TEST_ON_TRAIN:
                if self._image_idx_test == min(self.num_train_images, len(self._dataset_train)):
                    return None, info  # Returns None to signal that testing is done
                set_idx = self._image_idx_test
                self._image_idx_test += 1
            else:
                if self.epoch_count_callback is not None:
                    if self._image_idx_train != 0 and \
                            self._image_idx_train % self.epoch_frequency_callback == 0:
                        self.epoch_count_callback()

                if self._image_idx_train == 0:
                    self._reset_saved_masks_dict()

                set_idx = self._image_idx_train
                self._image_idx_train = (self._image_idx_train + 1) % self.num_train_images

        info['split'] = self.split_names[self.data_mode]
        info['image_idx'] = set_order[set_idx]
        mask_image_raw = dataset.__getitem__(info['image_idx'])

        # Separate image data into ground truth, mask and k-space (the last one for RAW only)
        self._ground_truth = mask_image_raw[1]
        if self.options.dataroot == 'KNEE_RAW':  # store k-space data too
            self._k_space = mask_image_raw[2].unsqueeze(0)
            self._ground_truth = self._ground_truth.permute(2, 0, 1)
        logging.debug(f"{info['split'].capitalize()} episode started "
                      f"with image {set_order[set_idx]}")

        self._ground_truth = self._ground_truth.to(device).unsqueeze(0)
        self._current_mask = self._initial_mask if start_with_initial_mask \
            else mask_image_raw[0].to(device).unsqueeze(0)
        if self._current_mask.byte().all() == 1:
            # No valid actions in this mask, replace with initial mask to have a valid mask
            self._current_mask = self._initial_mask
        self._scans_left = min(self.options.budget, self.action_space.n)
        observation, score = self._compute_observation_and_score()
        self._current_score = score
        self._initial_score_episode = {
            key: value.item()
            for key, value in self._current_score.items()
        }
        return observation, info

    def step(self, action: int) -> Tuple[Dict[str, torch.Tensor], float, bool, Dict]:
        """ Adds a new line (specified by the action) to the current mask and computes the
            resulting observation and reward (drop in MSE after reconstructing with respect to the
            current ground truth).
        """
        assert self._scans_left > 0
        self._last_action = action
        self._current_mask, has_already_been_scanned = self.compute_new_mask(
            self._current_mask, action)
        if self.data_mode == ReconstructionEnv.DataMode.TRAIN:
            image_idx = self._train_order[self._image_idx_train]
            if image_idx not in self.mask_dict.keys():
                self.mask_dict[image_idx] = np.zeros(
                    (self.options.budget, self.image_width), dtype=np.float32)

            self.mask_dict[image_idx][-self._scans_left] = \
                self._current_mask.squeeze().cpu().numpy()
        observation, new_score = self._compute_observation_and_score()

        metric = self.options.reward_metric
        if self.options.use_score_as_reward:
            reward_ = new_score[metric] - self._initial_score_episode[metric]
        else:
            reward_ = new_score[metric] - self._current_score[metric]
        factor = 100
        if self.options.reward_metric == 'mse':
            factor *= -1  # We try to minimize MSE, but DQN maximizes
        reward = -1.0 if has_already_been_scanned else reward_.item() * factor

        # Apply normalization if present
        if self.data_mode == ReconstructionEnv.DataMode.TRAIN and \
                self.options.normalize_rewards_on_val:
            reward /= np.abs(self._reference_mean_for_reward[self.options.budget - 1])
            reward /= (self.options.budget - 1)
        self._current_score = new_score

        self._scans_left -= 1
        done = (self._scans_left == 0) or (self._current_mask.byte().all() == 1)

        return observation, reward, done, {}

    def get_evaluator_action(self, obs: Dict[str, torch.Tensor]) -> int:
        """ Returns the action recommended by the evaluator network of `self._evaluator`. """
        with torch.no_grad():
            assert not self.options.obs_type == 'fourier_space' and not self.options.obs_to_numpy
            mask_embedding = None if obs['mask_embedding'] is None \
                else obs['mask_embedding'].to(device)
            k_space_scores = self._evaluator(obs['reconstruction'].to(device), mask_embedding,
                                             obs['mask'] if self.options.add_mask_eval else None)
            k_space_scores.masked_fill_(obs['mask'].to(device).squeeze().byte(), 100000)
            # if self.options.dataroot == 'KNEE_RAW':
            #     tmp = torch.zeros(obs['mask'].shape)
            #     tmp[0, 0, 0, 166:202] = 1
            #     k_space_scores.masked_fill_(tmp.to(device).squeeze().byte(), 100000)
            return torch.argmin(k_space_scores).item() - self.options.initial_num_lines_per_side

    def set_reference_point_for_rewards(self, statistics: Dict[str, Dict]):
        logging.info('Reference point will be set for reward.')
        num_actions = min(self.options.budget, self.action_space.n)
        self._reference_mean_for_reward = np.ndarray(num_actions)
        self._reference_std_for_reward = np.ndarray(num_actions)
        for t in range(num_actions - 1, -1, -1):
            self._reference_mean_for_reward[t] = statistics['rewards'][num_actions - t]['mean']
            self._reference_std_for_reward[t] = np.sqrt(
                statistics['rewards'][num_actions - t]['m2'] /
                (statistics['rewards'][num_actions - t]['count'] - 1))
        logging.info(f'The following reference will be used:')
        logging.info(f'    mean: {self._reference_mean_for_reward}')
        logging.info(f'std: {self._reference_std_for_reward}')

    def retrain_reconstructor(self, logger, writer):
        logger.info(
            f'Training reconstructor for {self.options.num_epochs_train_reconstructor} epochs')
        del self._current_mask
        del self._ground_truth
        self._initial_mask = self._initial_mask.to('cpu')
        self._reconstructor.to(torch.device('cpu'))
        torch.cuda.empty_cache()
        # `reconstructor_trainer` will perform updates on its own `DataParallel` copy of the
        # reconstructor, and update the env's reconstructor every time a best validation score
        # is achieved (by calling `ReconstructionEnv.update_reconstructor_from_alt_opt()`)
        epochs_performed = self.reconstructor_trainer(self._start_epoch_for_alt_opt, self.mask_dict,
                                                      logger, writer)
        self._start_epoch_for_alt_opt += epochs_performed
        self._reconstructor.to(device)
        self._initial_mask = self._initial_mask.to(device)

        logger.info('Done training reconstructor')
        self._reset_saved_masks_dict()  # Reset mask dictionary for next epoch

    def update_reconstructor_from_alt_opt(
            self, trained_reconstructor: models.reconstruction.ReconstructorNetwork, epoch: int):
        self._reconstructor.load_state_dict({
            key.replace('module.', ''): val
            for key, val in trained_reconstructor.state_dict().items()
        })
        torch.save({
            'state_dict': self._reconstructor.state_dict(),
            'epoch': epoch
        }, os.path.join(self.options.checkpoints_dir, 'best_alt_opt_reconstructor.pt'))

    def _reset_saved_masks_dict(self):
        self.mask_dict.clear()

    def set_epoch_finished_callback(self, callback: Callable, frequency: int):
        # Every frequency number of epochs, the callback function will be called
        self.epoch_count_callback = callback
        self.epoch_frequency_callback = frequency
