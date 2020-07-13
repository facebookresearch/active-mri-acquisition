import logging
import os
import types

import gym
import gym.spaces
import numpy as np
import torch
from typing import Dict, List, Optional, Tuple, Union

import data
import models.evaluator
import models.fft_utils
import models.reconstruction
import util.util
import util.rl.utils

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

RAW_IMG_HEIGHT = 640
RAW_IMG_WIDTH = 368
DICOM_IMG_HEIGHT = 128
DICOM_IMG_WIDTH = 128
START_PADDING_RAW = 166
END_PADDING_RAW = 202
RAW_CENTER_CROP_SIZE = 320


def load_checkpoint(
    checkpoints_dir: str, name: str = "best_checkpoint.pth"
) -> Optional[Dict]:
    checkpoint_path = os.path.join(checkpoints_dir, name)
    if os.path.isfile(checkpoint_path):
        logging.info(f"Found checkpoint at {checkpoint_path}.")
        return torch.load(checkpoint_path)
    logging.info(f"No checkpoint found at {checkpoint_path}.")
    return None


# TODO set the directory for fastMRI data to be configurable
class ReconstructionEnv(gym.Env):
    """ Gym-like environment representing the active MRI acquisition process.

        Args:
            options (types.SimpleNamespace): Configuration options for the environment.\n
                *Mandatory fields:*\n

                \t-``dataroot`` (str)- the type of MRI data to be used. The two valid options are
                        "KNEE" for knee data stored in DICOM format, and "KNEE_RAW" for knee data
                        stored in RAW format.\n
                \t-``reconstructor_dir`` (str) - path to directory where reconstructor is stored.\n
                \t-``evaluator_dir`` (str) - path to directory where evaluator is stored.\n

                *Optional fields (i.e., can either be missing or set to None):*\n

                \t-``budget`` (int) - the horizon for an episode. Defaults to the number of non-active
                        columns in the initial mask (or half of this, for DICOM data).\n
                \t-``obs_type`` (str) - one of {"image_space", "fourier_space", "only_mask"}. See
                        :meth:`step` for description.
                        Defaults to "image_space".\n
                \t-``test_set`` (str) - indicates the data to use for testing episodes (i.e.,
                        when ``self.data_mode`` is "test"). Options are "train", "val", "test".
                        See :meth:`set_testing`). Defaults to "val".\n
                \t-``num_train_images`` (int) - how many images to select from the training dataset.
                        Defaults to the full dataset.\n
                \t-``num_test_images`` (int) - how many images to select from the test dataset.
                        Defaults to the full dataset.\n
                \t-``initial_num_lines_per_side`` (int) - how many active k-space columns
                        on each side of the initial mask for each episode.
                        Defaults to 5 for "KNEE" data and 15 for "KNEE_RAW" data.\n

                *Optional fields - advanced usage:*\n

                \t-``rnl_params`` (str) - "Random Number of Lines" parameters."
                        :meth:`reset` by default initializes episodes using the masks that are
                        returned by our MRI data loader, which are randomly sampled according to
                        some distribution. The ``rnl_params`` option is used to control this
                        distribution. For details, see :mod:`data.masking_utils`.\n


    """

    def __init__(self, options: types.SimpleNamespace):
        self.options = options
        self.options.device = device
        self.image_height = (
            RAW_IMG_HEIGHT if options.dataroot == "KNEE_RAW" else DICOM_IMG_HEIGHT
        )
        self.image_width = (
            RAW_IMG_WIDTH if options.dataroot == "KNEE_RAW" else DICOM_IMG_WIDTH
        )
        self.conjugate_symmetry = options.dataroot != "KNEE_RAW"

        self.options.rnl_params = (
            f"{2 * self.options.initial_num_lines_per_side},"
            f"{2 * (self.options.initial_num_lines_per_side + 1)},1,5"
        )

        train_loader, valid_loader = data.create_data_loaders(options, is_test=False)
        test_loader = data.create_data_loaders(options, is_test=True)

        self._dataset_train = train_loader.dataset
        if options.test_set == "train":
            self._dataset_test = train_loader.dataset
        elif options.test_set == "val":
            self._dataset_test = valid_loader.dataset
        elif options.test_set == "test":
            self._dataset_test = test_loader.dataset
        else:
            raise ValueError("Valid options are train, val, test")

        self.num_train_images = min(
            self.options.num_train_images, len(self._dataset_train)
        )
        self.num_test_images = min(
            self.options.num_test_images, len(self._dataset_test)
        )
        self.latest_train_images = []

        self.rng = np.random.RandomState(options.seed)
        rng_train = (
            np.random.RandomState() if options.rl_env_train_no_seed else self.rng
        )
        self._train_order = rng_train.permutation(len(self._dataset_train))
        self._test_order = self.rng.permutation(len(self._dataset_test))
        if self.options.test_set_shift is not None:
            assert self.options.test_set_shift < (len(self._dataset_test) - 1)
            self._test_order = np.roll(self._test_order, -self.options.test_set_shift)
        self._image_idx_test = 0
        self._image_idx_train = 0
        self.data_mode = "train"

        reconstructor_checkpoint = load_checkpoint(
            options.reconstructor_dir, "best_checkpoint.pth"
        )
        self._reconstructor = models.reconstruction.ReconstructorNetwork(
            number_of_cascade_blocks=reconstructor_checkpoint[
                "options"
            ].number_of_cascade_blocks,
            n_downsampling=reconstructor_checkpoint["options"].n_downsampling,
            number_of_filters=reconstructor_checkpoint[
                "options"
            ].number_of_reconstructor_filters,
            number_of_layers_residual_bottleneck=reconstructor_checkpoint[
                "options"
            ].number_of_layers_residual_bottleneck,
            mask_embed_dim=reconstructor_checkpoint["options"].mask_embed_dim,
            dropout_probability=reconstructor_checkpoint["options"].dropout_probability,
            img_width=self.image_width,
            use_deconv=reconstructor_checkpoint["options"].use_deconv,
        )
        self._reconstructor.load_state_dict(
            {
                # TODO: this is true only in case of single gpu:
                key.replace("module.", ""): val
                for key, val in reconstructor_checkpoint["reconstructor"].items()
            }
        )
        self._reconstructor.eval()
        self._reconstructor.to(device)
        logging.info("Loaded reconstructor and original options from checkpoint.")

        self._evaluator = None
        evaluator_checkpoint = None
        if options.evaluator_dir is not None:
            evaluator_checkpoint = load_checkpoint(
                options.evaluator_dir, "best_checkpoint.pth"
            )
        if (
            evaluator_checkpoint is not None
            and evaluator_checkpoint["evaluator"] is not None
        ):
            self._evaluator = models.evaluator.EvaluatorNetwork(
                number_of_filters=evaluator_checkpoint[
                    "options"
                ].number_of_evaluator_filters,
                number_of_conv_layers=evaluator_checkpoint[
                    "options"
                ].number_of_evaluator_convolution_layers,
                use_sigmoid=False,
                width=evaluator_checkpoint["options"].image_width,
                height=RAW_IMG_HEIGHT if options.dataroot == "KNEE_RAW" else None,
                mask_embed_dim=evaluator_checkpoint["options"].mask_embed_dim,
            )
            logging.info(f"Loaded evaluator from checkpoint.")
            self._evaluator.load_state_dict(
                {
                    key.replace("module.", ""): val
                    for key, val in evaluator_checkpoint["evaluator"].items()
                }
            )
            self._evaluator.eval()
            self._evaluator.to(device)

        self.observation_space = (
            None  # The observation is a dict unless `obs_to_numpy` is used
        )
        if self.options.obs_to_numpy:
            if self.options.obs_type == "only_mask":
                obs_shape = (self.image_width,)
            else:
                # The extra rows represents the current mask and the mask embedding
                obs_shape = (2, self.image_height + 2, self.image_width)
            self.observation_space = gym.spaces.Box(
                low=-50000, high=50000, shape=obs_shape
            )

        self.metadata = {
            "mask_embed_dim": reconstructor_checkpoint["options"].mask_embed_dim
        }

        # Setting up valid actions
        factor = 2 if self.conjugate_symmetry else 1
        num_actions = (
            self.image_width - 2 * options.initial_num_lines_per_side
        ) // factor
        self.valid_actions = list(range(num_actions))
        if self.options.dataroot == "KNEE_RAW":
            # use invalid_actions when using k_space knee data that are zero padded
            # at some frequencies
            # TODO: do we want to change the fixed numbers below or not?
            invalid_actions = list(
                range(
                    START_PADDING_RAW - self.options.initial_num_lines_per_side,
                    END_PADDING_RAW - self.options.initial_num_lines_per_side,
                    1,
                )
            )
            self.valid_actions = np.setdiff1d(self.valid_actions, invalid_actions)
        self.action_space = gym.spaces.Discrete(len(self.valid_actions))

        self._initial_mask = self._generate_initial_lowf_mask().to(device)
        self._ground_truth = None
        self._current_reconstruction = None
        self._k_space = None
        self._current_mask = None
        self._current_score = None
        self._initial_score_episode = None
        self._scans_left = None
        self._reference_mean_for_reward = None
        self._reference_std_for_reward = None
        self._max_cols_cutoff = None

    def _generate_initial_lowf_mask(self):
        mask = torch.zeros(1, 1, 1, self.image_width)
        for i in range(self.options.initial_num_lines_per_side):
            mask[0, 0, 0, i] = 1
            mask[0, 0, 0, -(i + 1)] = 1
        return mask

    def _get_current_reconstruction_and_mask_embedding(
        self,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        prev_reconstruction = (
            self._current_reconstruction
            if self.options.keep_prev_reconstruction
            else None
        )
        reconstructor_input, _, mask_to_use = models.fft_utils.preprocess_inputs(
            (self._current_mask, self._ground_truth, self._k_space),
            self.options.dataroot,
            device,
            prev_reconstruction,
        )

        if self.options.use_reconstructions:
            reconstruction, _, mask_embed = self._reconstructor(
                reconstructor_input, mask_to_use
            )
        else:
            reconstruction = reconstructor_input
            mask_embed = None

        return reconstruction, mask_embed, mask_to_use

    def set_testing(self, use_training_set=False, reset_index=True):
        self.data_mode = "test_on_train" if use_training_set else "test"
        if reset_index:
            self._image_idx_test = 0
        self._max_cols_cutoff = self.options.test_num_cols_cutoff

    def set_training(self, reset_index=False):
        self.data_mode = "train"
        if reset_index:
            self._image_idx_train = 0
        self._max_cols_cutoff = None

    @staticmethod
    def convert_num_cols_to_acceleration(num_cols, dataroot):
        if dataroot == "KNEE":
            return DICOM_IMG_WIDTH / num_cols
        if dataroot == "KNEE_RAW":
            return (RAW_IMG_WIDTH - (END_PADDING_RAW - START_PADDING_RAW)) / num_cols
        raise ValueError("Dataset type not understood.")

    @staticmethod
    def _compute_score_given_tensors(
        reconstruction: torch.Tensor, ground_truth: torch.Tensor, is_raw: bool
    ) -> Dict[str, torch.Tensor]:
        # Compute magnitude (for metrics)
        reconstruction = models.fft_utils.to_magnitude(reconstruction)
        ground_truth = models.fft_utils.to_magnitude(ground_truth)
        if is_raw:  # crop data
            reconstruction = models.fft_utils.center_crop(
                reconstruction, [RAW_CENTER_CROP_SIZE, RAW_CENTER_CROP_SIZE]
            )
            ground_truth = models.fft_utils.center_crop(
                ground_truth, [RAW_CENTER_CROP_SIZE, RAW_CENTER_CROP_SIZE]
            )

        mse = util.util.compute_mse(reconstruction, ground_truth)
        nmse = util.util.compute_nmse(reconstruction, ground_truth)
        ssim = util.util.compute_ssims(reconstruction, ground_truth)
        psnr = util.util.compute_psnrs(reconstruction, ground_truth)

        score = {"mse": mse, "nmse": nmse, "ssim": ssim, "psnr": psnr}

        return score

    def compute_new_mask(
        self, old_mask: torch.Tensor, action: int
    ) -> Tuple[torch.Tensor, bool]:
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

    def get_num_active_columns_in_obs(self, obs):
        """ Returns the number of active columns in the given (single) observation's mask. """
        if self.options.obs_to_numpy:
            mask = (
                obs
                if self.options.obs_type == "only_mask"
                else obs[0, self.image_height, :]
            )
            num_active_cols = len(mask.nonzero()[0])
        else:
            num_active_cols = len(obs["mask"].nonzero())
        if self.options.dataroot == "KNEE_RAW":
            num_active_cols -= (
                END_PADDING_RAW - START_PADDING_RAW
            )  # remove count of padding cols
        return num_active_cols

    def compute_score(
        self,
        use_reconstruction: bool = True,
        ground_truth: Optional[torch.Tensor] = None,
        mask_to_use: Optional[torch.Tensor] = None,
        k_space: Optional[torch.Tensor] = None,
        use_current_score: bool = False,
    ) -> List[Dict[str, torch.Tensor]]:
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
            prev_reconstruction = (
                self._current_reconstruction
                if self.options.keep_prev_reconstruction
                else None
            )
            image, _, mask_to_use = models.fft_utils.preprocess_inputs(
                (mask_to_use, ground_truth, k_space),
                self.options.dataroot,
                device,
                prev_reconstruction,
            )
            if use_reconstruction:  # pass through reconstruction network
                image, _, _ = self._reconstructor(image, mask_to_use)
        return [
            ReconstructionEnv._compute_score_given_tensors(
                img.unsqueeze(0), ground_truth, self.options.dataroot == "KNEE_RAW"
            )
            for img in image
        ]

    def _compute_observation_and_score(self) -> Tuple[Union[Dict, np.ndarray], Dict]:
        with torch.no_grad():
            # Note that `mask` is a processed version of `self._current_mask` that also includes
            # the padding for RAW data
            (
                reconstruction,
                mask_embedding,
                mask,
            ) = self._get_current_reconstruction_and_mask_embedding()
            score = ReconstructionEnv._compute_score_given_tensors(
                reconstruction, self._ground_truth, self.options.dataroot == "KNEE_RAW"
            )

            if self.options.obs_type == "only_mask":
                observation = {"mask": mask}
                if self.options.obs_to_numpy:
                    observation = mask.squeeze().cpu().numpy()
                return observation, score

            if self.options.obs_type == "fourier_space":
                reconstruction = models.fft_utils.fft(reconstruction)

            self._current_reconstruction = reconstruction
            observation = {
                "reconstruction": reconstruction,
                "mask": mask,
                "mask_embedding": mask_embedding,
            }

            if self.options.obs_to_numpy:
                observation = np.zeros(self.observation_space.shape).astype(np.float32)
                observation[:2, : self.image_height, :] = (
                    reconstruction[0].cpu().numpy()
                )
                # The second to last row is the mask
                observation[:, self.image_height, :] = mask.cpu().numpy()
                # The last row is the mask embedding (padded with 0s if necessary)
                if self.metadata["mask_embed_dim"] == 0:
                    observation[:, self.image_height + 1, 0] = np.nan
                else:
                    observation[
                        :, self.image_height + 1, : self.metadata["mask_embed_dim"]
                    ] = (mask_embedding[0, :, 0, 0].cpu().numpy())

        return observation, score

    def reset(self, start_with_initial_mask=False) -> Tuple[Union[Dict, None], Dict]:
        """ Loads a new image from the dataset and starts a new episode with this image.

            Loops over images in the dataset in order. The dataset is ordered according to
            `self._{train/test}_order`.
        """
        info = {}
        if self.data_mode == "test":
            if self._image_idx_test == min(
                self.num_test_images, len(self._dataset_test)
            ):
                return None, info  # Returns None to signal that testing is done
            set_order = self._test_order
            dataset = self._dataset_test
            set_idx = self._image_idx_test
            self._image_idx_test += 1
        else:
            set_order = self._train_order
            dataset = self._dataset_train
            if self.data_mode == "test_on_train":
                if self._image_idx_test == min(
                    self.num_train_images, len(self._dataset_train)
                ):
                    return None, info  # Returns None to signal that testing is done
                set_idx = self._image_idx_test
                self._image_idx_test += 1
            else:
                set_idx = self._image_idx_train
                self._image_idx_train = (
                    self._image_idx_train + 1
                ) % self.num_train_images

        info["split"] = self.data_mode
        info["image_idx"] = set_order[set_idx]
        mask_image_raw = dataset.__getitem__(info["image_idx"])

        # Separate image data into ground truth, mask and k-space (the last one for RAW only)
        self._ground_truth = mask_image_raw[1]
        if self.options.dataroot == "KNEE_RAW":  # store k-space data too
            self._k_space = mask_image_raw[2].unsqueeze(0)
            self._ground_truth = self._ground_truth.permute(2, 0, 1)
        logging.debug(
            f"{info['split'].capitalize()} episode started "
            f"with image {set_order[set_idx]}"
        )

        self._ground_truth = self._ground_truth.to(device).unsqueeze(0)
        self._current_reconstruction = None
        self._current_mask = (
            self._initial_mask
            if start_with_initial_mask
            else mask_image_raw[0].to(device).unsqueeze(0)
        )
        if self._current_mask.byte().all() == 1:
            # No valid actions in this mask, replace with initial mask to have a valid mask
            self._current_mask = self._initial_mask
        self._scans_left = min(self.options.budget, self.action_space.n)
        observation, score = self._compute_observation_and_score()
        self._current_score = score
        self._initial_score_episode = {
            key: value.item() for key, value in self._current_score.items()
        }
        return observation, info

    def step(self, action: int) -> Tuple[Dict[str, torch.Tensor], float, bool, Dict]:
        """ Adds a new line (specified by the action) to the current mask and computes the
            resulting observation and reward (drop in MSE after reconstructing with respect to the
            current ground truth).
        """
        self._current_mask, has_already_been_scanned = self.compute_new_mask(
            self._current_mask, action
        )
        observation, new_score = self._compute_observation_and_score()

        metric = self.options.reward_metric
        if self.options.use_score_as_reward:
            reward_ = new_score[metric] - self._initial_score_episode[metric]
        else:
            reward_ = new_score[metric] - self._current_score[metric]
        factor = self.options.reward_scaling
        if self.options.reward_metric == "mse" or self.options.reward_metric == "nmse":
            factor *= -1  # We try to minimize MSE, but DQN maximizes
        reward = -1.0 if has_already_been_scanned else reward_.item() * factor
        self._current_score = new_score

        self._scans_left -= 1

        if self._max_cols_cutoff is None:
            assert self._scans_left >= 0
            done = (self._scans_left == 0) or (self._current_mask.byte().all() == 1)
        else:
            done = (
                self.get_num_active_columns_in_obs(observation) >= self._max_cols_cutoff
            )

        return observation, reward, done, {}

    def get_evaluator_action(self, obs: Dict[str, torch.Tensor]) -> int:
        """ Returns the action recommended by the evaluator network of `self._evaluator`. """
        with torch.no_grad():
            assert (
                not self.options.obs_type == "fourier_space"
                and not self.options.obs_to_numpy
            )
            mask_embedding = (
                None
                if obs["mask_embedding"] is None
                else obs["mask_embedding"].to(device)
            )
            k_space_scores = self._evaluator(
                obs["reconstruction"].to(device),
                mask_embedding,
                obs["mask"] if self.options.add_mask_eval else None,
            )
            # Just fill chosen actions with some very large number to prevent from selecting again.
            k_space_scores.masked_fill_(obs["mask"].to(device).squeeze().byte(), 100000)
            return (
                torch.argmin(k_space_scores).item()
                - self.options.initial_num_lines_per_side
            )

    def render(self, mode="human"):
        pass
