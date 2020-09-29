"""
activemri.envs.envs.py
====================================
Gym-like environment for active MRI acquisition.
"""
import functools
import json
import os
import pathlib
import warnings
from typing import (
    Any,
    Callable,
    Dict,
    Iterator,
    List,
    Mapping,
    Optional,
    Sequence,
    Sized,
    Tuple,
    Union,
)

import fastmri.data
import gym
import numpy as np
import torch
import torch.utils.data

import activemri.data.singlecoil_knee_data as scknee_data
import activemri.data.transforms
import activemri.envs.masks
import activemri.envs.util
import activemri.models

DataInitFnReturnType = Tuple[
    torch.utils.data.Dataset, torch.utils.data.Dataset, torch.utils.data.Dataset
]


# -----------------------------------------------------------------------------
#                               DATA HANDLING
# -----------------------------------------------------------------------------
class CyclicSampler(torch.utils.data.Sampler):
    def __init__(
        self,
        data_source: Sized,
        order: Optional[Sized] = None,
        loops: int = 1,
    ):
        torch.utils.data.Sampler.__init__(self, data_source)
        assert loops > 0
        assert order is None or len(order) == len(data_source)
        self.data_source = data_source
        self.order = order if order is not None else range(len(self.data_source))
        self.loops = loops

    def _iterator(self):
        for _ in range(self.loops):
            for j in self.order:
                yield j

    def __iter__(self):
        return iter(self._iterator())

    def __len__(self):
        return len(self.data_source) * self.loops


# TODO Add a fastMRI batch type (make it the return of void_transform)
def _env_collate_fn(
    batch: Tuple[Union[np.array, list], ...]
) -> Tuple[Union[np.array, list], ...]:
    ret = []
    for i in range(6):  # kspace, mask, target, attrs, fname, slice_id
        ret.append([item[i] for item in batch])
    return tuple(ret)


class DataHandler:
    def __init__(
        self,
        data_source: torch.utils.data.Dataset,
        seed: Optional[int],
        batch_size: int = 1,
        loops: int = 1,
        collate_fn: Optional[Callable] = None,
    ):
        self._iter = None  # type: Iterator[Any]
        self._collate_fn = collate_fn
        self._batch_size = batch_size
        self._loops = loops
        self._init_impl(data_source, seed, batch_size, loops, collate_fn)

    def _init_impl(
        self,
        data_source: torch.utils.data.Dataset,
        seed: Optional[int],
        batch_size: int = 1,
        loops: int = 1,
        collate_fn: Optional[Callable] = None,
    ):
        rng = np.random.RandomState(seed)
        order = rng.permutation(len(data_source))
        sampler = CyclicSampler(data_source, order, loops=loops)
        if collate_fn:
            self._data_loader = torch.utils.data.DataLoader(
                data_source,
                batch_size=batch_size,
                sampler=sampler,
                collate_fn=collate_fn,
            )
        else:
            self._data_loader = torch.utils.data.DataLoader(
                data_source, batch_size=batch_size, sampler=sampler
            )
        self._iter = iter(self._data_loader)

    def reset(self):
        self._iter = iter(self._data_loader)

    def __iter__(self):
        return self._iter

    def __next__(self):
        return next(self._iter)

    def seed(self, seed: int):
        self._init_impl(
            self._data_loader.dataset,
            seed,
            self._batch_size,
            self._loops,
            self._collate_fn,
        )


# -----------------------------------------------------------------------------
#                           BASE ACTIVE MRI ENV
# -----------------------------------------------------------------------------


class ActiveMRIEnv(gym.Env):
    """Base class for all active MRI acquisition environments.

    This class provides the core logic implementation of the k-space acquisition process.
    The class is not to be used directly, but rather one of its subclasses should be
    instantiated. Subclasses of `ActiveMRIEnv` are responsible for data initialization
    and specifying configuration options for the environment.

    Args:
        kspace_shape(tuple(int,int)): Shape of the k-space slices for the dataset.
        num_parallel_episodes(int): Determines the number images that will be processed
                                    simultaneously by :meth:`reset()` and :meth:`step()`.
                                    Defaults to 1.
        budget(optional(int)): The length of an acquisition episode. Defaults to ``None``,
                               which indicates that episode will continue until all k-space
                               columns have been acquired.
        seed(optional(int)): The seed for the environment's random number generator, which is
                             an instance of ``numpy.random.RandomState``. Defaults to ``None``.
        no_checkpoint(optional(bool)): Set to ``True`` if you want to run your reconstructor
                                       model without loading anything from a checkpoint.

    """

    _num_loops_train_data = 100000

    metadata = {"render.modes": ["human"], "video.frames_per_second": None}

    def __init__(
        self,
        kspace_shape: Tuple[int, int],
        num_parallel_episodes: int = 1,
        budget: Optional[int] = None,
        seed: Optional[int] = None,
        no_checkpoint: Optional[bool] = False,
    ):
        # Default initialization
        self._cfg: Mapping[str, Any] = None
        self._needs_checkpoint = not no_checkpoint
        self._data_location: str = None
        self._reconstructor: activemri.models.Reconstructor = None
        self._transform: Callable = None
        self._train_data_handler: DataHandler = None
        self._val_data_handler: DataHandler = None
        self._test_data_handler: DataHandler = None
        self._device = torch.device("cpu")
        self.num_parallel_episodes = num_parallel_episodes
        self.budget = budget

        self._seed = seed
        self._rng = np.random.RandomState(seed)
        self.reward_metric = "mse"

        # Init from provided configuration
        self.kspace_height, self.kspace_width = kspace_shape

        # Gym init
        # Observation is a dictionary
        self.observation_space = None
        self.action_space = gym.spaces.Discrete(self.kspace_width)

        # This is changed by `set_training()`, `set_val()`, `set_test()`
        self._current_data_handler: DataHandler = None

        # These are changed every call to `reset()`
        self._current_ground_truth: torch.Tensor = None
        self._transform_wrapper: Callable = None
        self._current_k_space: torch.Tensor = None
        self._did_reset = False
        self._steps_since_reset = 0
        # These three are changed every call to `reset()` and every call to `step()`
        self._current_reconstruction_numpy: np.ndarray = None
        self._current_score: Dict[str, np.ndarray] = None
        self._current_mask: torch.Tensor = None

    # -------------------------------------------------------------------------
    # Protected methods
    # -------------------------------------------------------------------------
    def _setup(
        self,
        cfg_filename: str,
        data_init_func: Callable[[], DataInitFnReturnType],
    ):
        self._init_from_config_file(cfg_filename)
        self._setup_data_handlers(data_init_func)

    def _setup_data_handlers(
        self,
        data_init_func: Callable[[], DataInitFnReturnType],
    ):
        train_data, val_data, test_data = data_init_func()
        self._train_data_handler = DataHandler(
            train_data,
            self._seed,
            batch_size=self.num_parallel_episodes,
            loops=self._num_loops_train_data,
            collate_fn=_env_collate_fn,
        )
        self._val_data_handler = DataHandler(
            val_data,
            self._seed + 1 if self._seed else None,
            batch_size=self.num_parallel_episodes,
            loops=1,
            collate_fn=_env_collate_fn,
        )
        self._test_data_handler = DataHandler(
            test_data,
            self._seed + 2 if self._seed else None,
            batch_size=self.num_parallel_episodes,
            loops=1,
            collate_fn=_env_collate_fn,
        )
        self._current_data_handler = self._train_data_handler

    def _init_from_config_dict(self, cfg: Mapping[str, Any]):
        self._cfg = cfg
        self._data_location = cfg["data_location"]
        if not os.path.isdir(self._data_location):
            default_cfg, defaults_fname = activemri.envs.util.get_defaults_json()
            self._data_location = default_cfg["data_location"]
            if not os.path.isdir(self._data_location):
                raise RuntimeError(
                    f"No 'data_location' key found in the given config. Please "
                    f"write dataset location in your JSON config, or in file {defaults_fname} "
                    f"(to use as a default)."
                )
        self._device = torch.device(cfg["device"])
        self.reward_metric = cfg["reward_metric"]
        if self.reward_metric not in ["mse", "ssim", "psnr", "nmse"]:
            raise ValueError("Reward metric must be one of mse, nmse, ssim, or psnr.")
        mask_func = activemri.envs.util.import_object_from_str(cfg["mask"]["function"])
        self._mask_func = functools.partial(mask_func, cfg["mask"]["args"])

        # Instantiating reconstructor
        reconstructor_cfg = cfg["reconstructor"]
        reconstructor_cls = activemri.envs.util.import_object_from_str(
            reconstructor_cfg["cls"]
        )

        checkpoint_fname = pathlib.Path(reconstructor_cfg["checkpoint_fname"])
        default_cfg, defaults_fname = activemri.envs.util.get_defaults_json()
        saved_models_dir = default_cfg["saved_models_dir"]
        checkpoint_path = pathlib.Path(saved_models_dir) / checkpoint_fname
        if self._needs_checkpoint and not checkpoint_path.is_file():
            raise RuntimeError(
                f"No checkpoint was found at {str(checkpoint_path)}. "
                f"Please make sure that both 'checkpoint_fname' (in your JSON config) "
                f"and 'saved_models_dir' (in {defaults_fname}) are configured correctly."
            )

        checkpoint = (
            torch.load(str(checkpoint_path)) if checkpoint_path.is_file() else None
        )
        options = reconstructor_cfg["options"]
        if checkpoint and "options" in checkpoint:
            msg = (
                f"Checkpoint at {checkpoint_path.name} has an 'options' key. "
                f"This will override the options defined in configuration file."
            )
            warnings.warn(msg)
            options = checkpoint["options"]
            assert isinstance(options, dict)
        self._reconstructor = reconstructor_cls(**options)
        self._reconstructor.init_from_checkpoint(checkpoint)
        self._reconstructor.eval()
        self._reconstructor.to(self._device)
        self._transform = activemri.envs.util.import_object_from_str(
            reconstructor_cfg["transform"]
        )

    def _init_from_config_file(self, config_filename: str):
        with open(config_filename, "rb") as f:
            self._init_from_config_dict(json.load(f))

    @staticmethod
    def _void_transform(
        kspace: torch.Tensor,
        mask: torch.Tensor,
        target: torch.Tensor,
        attrs: List[Dict[str, Any]],
        fname: List[str],
        slice_id: List[int],
    ) -> Tuple:
        return kspace, mask, target, attrs, fname, slice_id

    def _send_tuple_to_device(self, the_tuple: Tuple[Union[Any, torch.Tensor]]):
        the_tuple_device = []
        for i in range(len(the_tuple)):
            if isinstance(the_tuple[i], torch.Tensor):
                the_tuple_device.append(the_tuple[i].to(self._device))
            else:
                the_tuple_device.append(the_tuple[i])
        return tuple(the_tuple_device)

    @staticmethod
    def _send_dict_to_cpu_and_detach(the_dict: Dict[str, Union[Any, torch.Tensor]]):
        the_dict_cpu = {}
        for key in the_dict:
            if isinstance(the_dict[key], torch.Tensor):
                the_dict_cpu[key] = the_dict[key].detach().cpu()
            else:
                the_dict_cpu[key] = the_dict[key]
        return the_dict_cpu

    def _compute_obs_and_score(
        self, override_current_mask: Optional[torch.Tensor] = None
    ) -> Tuple[Dict[str, Any], Dict[str, np.ndarray]]:
        mask_to_use = (
            override_current_mask
            if override_current_mask is not None
            else self._current_mask
        )
        reconstructor_input = self._transform_wrapper(
            kspace=self._current_k_space,
            mask=mask_to_use,
            ground_truth=self._current_ground_truth,
        )

        reconstructor_input = self._send_tuple_to_device(reconstructor_input)
        with torch.no_grad():
            extra_outputs = self._reconstructor(*reconstructor_input)

        extra_outputs = self._send_dict_to_cpu_and_detach(extra_outputs)
        reconstruction = extra_outputs["reconstruction"]

        # this dict is only for storing the other outputs
        del extra_outputs["reconstruction"]

        # noinspection PyUnusedLocal
        reconstructor_input = None  # de-referencing GPU tensors

        score = self._compute_score_given_tensors(
            *self._process_tensors_for_score_fns(
                reconstruction, self._current_ground_truth
            )
        )

        obs = {
            "reconstruction": reconstruction,
            "extra_outputs": extra_outputs,
            "mask": self._current_mask.clone().view(self._current_mask.shape[0], -1),
        }

        return obs, score

    def _clear_cache_and_unset_did_reset(self):
        self._current_mask = None
        self._current_ground_truth = None
        self._current_reconstruction_numpy = None
        self._transform_wrapper = None
        self._current_k_space = None
        self._current_score = None
        self._steps_since_reset = 0
        self._did_reset = False

    # noinspection PyMethodMayBeStatic
    def _process_tensors_for_score_fns(
        self, reconstruction: torch.Tensor, ground_truth: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return reconstruction, ground_truth

    @staticmethod
    def _compute_score_given_tensors(
        reconstruction: torch.Tensor, ground_truth: torch.Tensor
    ) -> Dict[str, np.ndarray]:
        mse = activemri.envs.util.compute_mse(reconstruction, ground_truth)
        nmse = activemri.envs.util.compute_nmse(reconstruction, ground_truth)
        ssim = activemri.envs.util.compute_ssim(reconstruction, ground_truth)
        psnr = activemri.envs.util.compute_psnr(reconstruction, ground_truth)

        return {"mse": mse, "nmse": nmse, "ssim": ssim, "psnr": psnr}

    @staticmethod
    def _convert_to_gray(array: np.ndarray) -> np.ndarray:
        M = np.max(array)
        m = np.min(array)
        return (255 * (array - m) / (M - m)).astype(np.uint8)

    @staticmethod
    def _render_arrays(
        ground_truth: np.ndarray, reconstruction: np.ndarray, mask: np.ndarray
    ) -> List[np.ndarray]:
        batch_size, img_height, img_width = ground_truth.shape
        frames = []
        for i in range(batch_size):
            mask_i = np.tile(mask[i], (1, img_height, 1))

            pad = 32
            mask_begin = pad
            mask_end = mask_begin + mask.shape[-1]
            gt_begin = mask_end + pad
            gt_end = gt_begin + img_width
            rec_begin = gt_end + pad
            rec_end = rec_begin + img_width
            error_begin = rec_end + pad
            error_end = error_begin + img_width
            frame = 128 * np.ones((img_height, error_end + pad), dtype=np.uint8)
            frame[:, mask_begin:mask_end] = 255 * mask_i
            frame[:, gt_begin:gt_end] = ActiveMRIEnv._convert_to_gray(ground_truth[i])
            frame[:, rec_begin:rec_end] = ActiveMRIEnv._convert_to_gray(
                reconstruction[i]
            )
            rel_error = np.abs((ground_truth[i] - reconstruction[i]) / ground_truth[i])
            frame[:, error_begin:error_end] = 255 * rel_error.astype(np.uint8)

            frames.append(frame)
        return frames

    # -------------------------------------------------------------------------
    # Public methods
    # -------------------------------------------------------------------------
    def reset(self) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Starts a new acquisition episode with a batch of images.

        This methods performs the following steps:

            1. Reads a batch of images from the environment's dataset.
            2. Creates an initial acquisition mask for each image.
            3. Passes the loaded data and the initial masks to the transform function,
               producing a batch of inputs for the environment's reconstructor model.
            4. Calls the reconstructor model on this input and returns its output
               as an observation.


        The observation returned is a dictionary with the following keys:
            - *"reconstruction"(torch.Tensor):* The reconstruction produced by the
              environment's reconstructor model, using the current
              acquisition mask.
            - *"extra_outputs"(dict(str,Any)):* A dictionary with any additional
              outputs produced by the reconstructor  (e.g., uncertainty maps).
            - *"mask"(torch.Tensor):* The current acquisition mask.

        |

        Returns:
            tuple: tuple containing:
            - obs(dict(str,any): Observation dictionary.
            - metadata(dict(str,any): Metadata information containing the following keys:

                - *"fname"(list(str)):* the filenames of the image read from the dataset.
                - *"slice_id"(list(int)):* slice indices for each image within the volume.
                - *"current_score"(dict(str,float):* A dictionary with the error measures
                  for the reconstruction (e.g., "mse", "nmse", "ssim", "psnr"). The measures
                  considered can be obtained with :meth:`score_keys()`.
        """
        self._did_reset = True
        try:
            kspace, _, ground_truth, attrs, fname, slice_id = next(
                self._current_data_handler
            )
        except StopIteration:
            return {}, {}

        self._current_ground_truth = torch.from_numpy(np.stack(ground_truth))

        # Converting k-space to torch is better handled by transform,
        # since we have both complex and non-complex versions
        self._current_k_space = kspace

        self._transform_wrapper = functools.partial(
            self._transform, attrs=attrs, fname=fname, slice_id=slice_id
        )
        kspace_shapes = [tuple(k.shape) for k in kspace]
        self._current_mask = self._mask_func(kspace_shapes, self._rng, attrs=attrs)
        obs, self._current_score = self._compute_obs_and_score()
        self._current_reconstruction_numpy = obs["reconstruction"].cpu().numpy()
        self._steps_since_reset = 0

        meta = {
            "fname": fname,
            "slice_id": slice_id,
            "current_score": self._current_score,
        }
        return obs, meta

    # TODO look how to handle the num_parallel_episodes=1 special case
    def step(
        self, action: Union[int, Sequence[int]]
    ) -> Tuple[Dict[str, Any], np.ndarray, List[bool], Dict]:
        """Performs a step of active MRI acquisition.

        Given a set of indices for k-space columns to acquire, updates the current batch
        of masks with their corresponding indices, creates a new batch of reconstructions,
        and returns the corresponding observations and rewards (for the observation format
        see :meth:`reset()`). The reward is the improvement in score with
        respect to the reconstruction before adding the indices. The specific score metric
        used is determined by ``env.reward_metric``.

        The method also returns a list of booleans, indicating whether any episodes in the
        batch have already concluded.

        The last return value is a metadata dictionary. It contains a single key
        "current_score", which contains a dictionary with the error measures for the
        reconstruction (e.g., ``"mse", "nmse", "ssim", "psnr"``). The measures
        considered can be obtained with :meth:`score_keys()`.

        Args:
            action(union(int, sequence(int))): Indices for k-space columns to acquire. The
                                               length of the sequence must be equal to the
                                               current number of parallel episodes
                                               (i.e., ``obs["reconstruction"].shape[0]``).
                                               If only an ``int`` is passed, the index will
                                               be replicated for the whole batch of episodes.

        Returns:
            tuple: The transition information in the order
            ``(next_observation, reward, done, meta)``. The types and shapes are:

              - ``next_observation(dict):`` Dictionary format (see :meth:`reset()`).
              - ``reward(np.ndarray)``: length equal to current number of parallel
                episodes.
              - ``done(list(bool))``: same length as ``reward``.
              - ``meta(dict)``: see description above.

        """
        if not self._did_reset:
            raise RuntimeError(
                "Attempting to call env.step() before calling env.reset()."
            )
        if isinstance(action, int):
            action = [action for _ in range(self.num_parallel_episodes)]
        self._current_mask = activemri.envs.masks.update_masks_from_indices(
            self._current_mask, action
        )
        obs, new_score = self._compute_obs_and_score()
        self._current_reconstruction_numpy = obs["reconstruction"].cpu().numpy()

        reward = new_score[self.reward_metric] - self._current_score[self.reward_metric]
        if self.reward_metric in ["mse", "nmse"]:
            reward *= -1
        else:
            assert self.reward_metric in ["ssim", "psnr"]
        self._current_score = new_score
        self._steps_since_reset += 1

        done = activemri.envs.masks.check_masks_complete(self._current_mask)
        if self.budget and self._steps_since_reset >= self.budget:
            done = [True] * len(done)
        return obs, reward, done, {"current_score": self._current_score}

    def try_action(
        self, action: Union[int, Sequence[int]]
    ) -> Tuple[Dict[str, Any], Dict[str, np.ndarray]]:
        """Simulates the effects of actions without changing the environment's state.

        This method operates almost exactly as :meth:`step()`, with the exception that
        the environment's state is not altered. The method returns the next observation
        and the resulting reconstruction score after applying the give k-space columns to
        each image in the current batch of episodes.

        Args:
            action(union(int, sequence(int))): Indices for k-space columns to acquire. The
                                               length of the sequence must be equal to the
                                               current number of parallel episodes
                                               (i.e., ``obs["reconstruction"].shape[0]``).
                                               If only an ``int`` is passed, the index will
                                               be replicated for the whole batch of episodes.

        Returns:
            tuple: The reconstruction information in the order
            ``(next_observation, current_score)``. The types and shapes are:

              - ``next_observation(dict):`` Dictionary format (see :meth:`reset()`).
              - ``current_score(dict(str, float))``: A dictionary with the error measures
                  for the reconstruction (e.g., "mse", "nmse", "ssim", "psnr"). The measures
                  considered can be obtained with `ActiveMRIEnv.score_keys()`.

        """
        if not self._did_reset:
            raise RuntimeError(
                "Attempting to call env.try_action() before calling env.reset()."
            )
        if isinstance(action, int):
            action = [action for _ in range(self.num_parallel_episodes)]
        new_mask = activemri.envs.masks.update_masks_from_indices(
            self._current_mask, action
        )
        obs, new_score = self._compute_obs_and_score(override_current_mask=new_mask)

        return obs, new_score

    def render(self, mode="human"):
        """Renders information about the environment's current state.

        Returns:
            ``np.ndarray``: An image frame containing, from left to right: current
                            acquisition mask, current ground image, current reconstruction,
                            and current relative reconstruction error.
        """
        pass

    def seed(self, seed: Optional[int] = None):
        """Sets the seed for the internal number generator.

        This seeds affects the order of the data loader for all loop modalities (i.e.,
        training, validation, test).

        Args:
            seed(optional(int)): The seed for the environment's random number generator.
        """
        self._seed = seed
        self._rng = np.random.RandomState(seed)
        self._train_data_handler.seed(seed)
        self._val_data_handler.seed(seed)
        self._test_data_handler.seed(seed)

    def set_training(self, reset: bool = False):
        """Sets the environment to use the training data loader.

        Args:
            reset(bool): If ``True``, also resets the data loader so that it starts again
                         from the first image in the loop order.

        Warning:
            After this method is called the ``env.reset()`` needs to be called again, otherwise
            an exception will be thrown.
        """
        if reset:
            self._train_data_handler.reset()
        self._current_data_handler = self._train_data_handler
        self._clear_cache_and_unset_did_reset()

    def set_val(self, reset: bool = True):
        """Sets the environment to use the validation data loader.

        Args:
            reset(bool): If ``True``, also resets the data loader so that it starts again
                         from the first image in the loop order.

        Warning:
            After this method is called the ``env.reset()`` needs to be called again, otherwise
            an exception will be thrown.
        """
        if reset:
            self._val_data_handler.reset()
        self._current_data_handler = self._val_data_handler
        self._clear_cache_and_unset_did_reset()

    def set_test(self, reset: bool = True):
        """Sets the environment to use the test data loader.

        Args:
            reset(bool): If ``True``, also resets the data loader so that it starts again
                         from the first image in the loop order.

        Warning:
            After this method is called the ``env.reset()`` needs to be called again, otherwise
            an exception will be thrown.
        """
        if reset:
            self._test_data_handler.reset()
        self._current_data_handler = self._test_data_handler
        self._clear_cache_and_unset_did_reset()

    @staticmethod
    def score_keys() -> List[str]:
        """ Returns the list of score metric names used by this environment. """
        return ["mse", "nmse", "ssim", "psnr"]


# -----------------------------------------------------------------------------
#                             CUSTOM ENVIRONMENTS
# -----------------------------------------------------------------------------
class MICCAI2020Env(ActiveMRIEnv):
    """Implementation of environment used for *Pineda et al., MICCAI 2020*.

    This environment is provided to facilitate replication of the experiments performed
    in *Luis Pineda, Sumana Basu, Adriana Romero, Roberto Calandra, Michal Drozdzal,
    "Active MR k-space Sampling with Reinforcement Learning". MICCAI 2020.*

    The dataset is the same as that of :class:`SingleCoilKneeEnv`, except that we provide
    a custom validation/test split of the original validation data. The environment's
    configuration file is set to use the reconstruction model used in the paper
    (see :class:`activemri.models.cvpr19_reconstructor.CVPR19Reconstructor`), as well
    as the proper transform to generate inputs for this model.

    The k-space shape of this environment is set to ``(640, 368)``.

    Args:
        num_parallel_episodes(int): Determines the number images that will be processed
                                    simultaneously by :meth:`reset()` and :meth:`step()`.
                                    Defaults to 1.
        budget(optional(int)): The length of an acquisition episode. Defaults to ``None``,
                               which indicates that episode will continue until all k-space
                               columns have been acquired.
        seed(optional(int)): The seed for the environment's random number generator, which is
                             an instance of ``numpy.random.RandomState``. Defaults to ``None``.
        extreme(bool): ``True`` or ``False`` for running extreme acceleration or normal
                       acceleration scenarios described in the paper, respectively.
    """

    KSPACE_WIDTH = scknee_data.MICCAI2020Data.KSPACE_WIDTH
    START_PADDING = scknee_data.MICCAI2020Data.START_PADDING
    END_PADDING = scknee_data.MICCAI2020Data.END_PADDING
    CENTER_CROP_SIZE = scknee_data.MICCAI2020Data.CENTER_CROP_SIZE

    def __init__(
        self,
        num_parallel_episodes: int = 1,
        budget: Optional[int] = None,
        seed: Optional[int] = None,
        extreme: bool = False,
        obs_includes_padding: bool = True,
    ):
        super().__init__(
            (640, self.KSPACE_WIDTH),
            num_parallel_episodes=num_parallel_episodes,
            budget=budget,
            seed=seed,
        )
        if extreme:
            self._setup("configs/miccai-2020-extreme-acc.json", self._create_dataset)
        else:
            self._setup("configs/miccai-2020-normal-acc.json", self._create_dataset)
        self.obs_includes_padding = obs_includes_padding

    # -------------------------------------------------------------------------
    # Protected methods
    # -------------------------------------------------------------------------
    def _create_dataset(self) -> DataInitFnReturnType:
        root_path = pathlib.Path(self._data_location)
        train_path = root_path / "knee_singlecoil_train"
        val_and_test_path = root_path / "knee_singlecoil_val"

        train_data = scknee_data.MICCAI2020Data(
            train_path,
            ActiveMRIEnv._void_transform,
            num_cols=self.KSPACE_WIDTH,
        )
        val_data = scknee_data.MICCAI2020Data(
            val_and_test_path,
            ActiveMRIEnv._void_transform,
            custom_split="val",
            num_cols=self.KSPACE_WIDTH,
        )
        test_data = scknee_data.MICCAI2020Data(
            val_and_test_path,
            ActiveMRIEnv._void_transform,
            custom_split="test",
            num_cols=self.KSPACE_WIDTH,
        )
        return train_data, val_data, test_data

    def _process_tensors_for_score_fns(
        self, reconstruction: torch.Tensor, ground_truth: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Compute magnitude (for metrics)
        reconstruction = activemri.data.transforms.to_magnitude(reconstruction, dim=3)
        ground_truth = activemri.data.transforms.to_magnitude(ground_truth, dim=3)

        reconstruction = activemri.data.transforms.center_crop(
            reconstruction, (self.CENTER_CROP_SIZE, self.CENTER_CROP_SIZE)
        )
        ground_truth = activemri.data.transforms.center_crop(
            ground_truth, (self.CENTER_CROP_SIZE, self.CENTER_CROP_SIZE)
        )
        return reconstruction, ground_truth

    # -------------------------------------------------------------------------
    # Public methods
    # -------------------------------------------------------------------------
    def reset(
        self,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        obs, meta = super().reset()
        if not obs:
            return obs, meta
        if self.obs_includes_padding:
            obs["mask"][:, self.START_PADDING : self.END_PADDING] = 1
        return obs, meta

    def step(
        self, action: Union[int, Sequence[int]]
    ) -> Tuple[Dict[str, Any], np.ndarray, List[bool], Dict]:
        obs, reward, done, meta = super().step(action)
        if self.obs_includes_padding:
            obs["mask"][:, self.START_PADDING : self.END_PADDING] = 1
        return obs, reward, done, meta

    def render(self, mode="human"):
        gt = self._current_ground_truth.cpu().numpy()
        rec = self._current_reconstruction_numpy

        gt = activemri.data.transforms.center_crop(
            (gt ** 2).sum(axis=3) ** 0.5, (self.CENTER_CROP_SIZE, self.CENTER_CROP_SIZE)
        )
        rec = activemri.data.transforms.center_crop(
            (rec ** 2).sum(axis=3) ** 0.5,
            (self.CENTER_CROP_SIZE, self.CENTER_CROP_SIZE),
        )
        return ActiveMRIEnv._render_arrays(gt, rec, self._current_mask.cpu().numpy())


class FastMRIEnv(ActiveMRIEnv):
    """Base class for all fastMRI environments.

    This class can be used to instantiate active acquisition environments using fastMRI
    data. However, for convenience we provided subclasses of ``FastMRIEnv`` with
    default configuration options for each dataset:

        - :class:`SingleCoilKneeEnv`
        - :class:`MultiCoilKneeEnv`
        - :class:`SingleCoilBrainEnv`
        - :class:`MultiCoilKneeEnv`

    The shape of the k-space is set to ``(640, max(num_cols))``.

    Args:
        config_path(str): The path to the JSON configuration file.
        dataset_name(str): One of "knee_singlecoil", "multicoil" (for knee),
                           "brain_multicoil". Primarily used to locate the fastMRI
                           dataset in the user's fastMRI data root folder.
        num_parallel_episodes(int): Determines the number images that will be processed
                                    simultaneously by :meth:`reset()` and :meth:`step()`.
                                    Defaults to 1.
        budget(optional(int)): The length of an acquisition episode. Defaults to ``None``,
                               which indicates that episode will continue until all k-space
                               columns have been acquired.
        seed(optional(int)): The seed for the environment's random number generator, which is
                             an instance of ``numpy.random.RandomState``. Defaults to ``None``.
        num_cols(sequence(int)): Used to filter k-space data to only use images whose k-space
                                 width is in this tuple. Defaults to ``(368, 372)``.
    """

    def __init__(
        self,
        config_path: str,
        dataset_name: str,
        num_parallel_episodes: int = 1,
        budget: Optional[int] = None,
        seed: Optional[int] = None,
        num_cols: Sequence[int] = (368, 372),
    ):
        assert dataset_name in ["knee_singlecoil", "multicoil", "brain_multicoil"]
        challenge = "singlecoil" if dataset_name == "knee_singlecoil" else "multicoil"
        super().__init__(
            (640, np.max(num_cols)),
            num_parallel_episodes=num_parallel_episodes,
            budget=budget,
            seed=seed,
        )
        self.num_cols = num_cols
        self.dataset_name = dataset_name
        self.challenge = challenge
        self._setup(config_path, self._create_dataset)

    def _create_dataset(self) -> DataInitFnReturnType:
        root_path = pathlib.Path(self._data_location)
        train_path = root_path / f"{self.dataset_name}_train"
        val_path = root_path / f"{self.dataset_name}_val"

        train_data = fastmri.data.SliceDataset(
            train_path,
            ActiveMRIEnv._void_transform,
            challenge=self.challenge,
            num_cols=self.num_cols,
            dataset_cache_file=pathlib.Path(
                f"__datacache__/train_{self.dataset_name}_cache.pkl"
            ),
        )
        val_data = fastmri.data.SliceDataset(
            val_path,
            ActiveMRIEnv._void_transform,
            challenge=self.challenge,
            num_cols=self.num_cols,
            dataset_cache_file=pathlib.Path(
                f"__datacache__/val_{self.dataset_name}_cache.pkl"
            ),
        )
        test_data = fastmri.data.SliceDataset(
            val_path,
            ActiveMRIEnv._void_transform,
            challenge=self.challenge,
            num_cols=self.num_cols,
            dataset_cache_file=pathlib.Path(
                f"__datacache__/val_{self.dataset_name}_cache.pkl"
            ),
        )
        return train_data, val_data, test_data

    def render(self, mode="human"):
        return ActiveMRIEnv._render_arrays(
            self._current_ground_truth.cpu().numpy(),
            self._current_reconstruction_numpy,
            self._current_mask.cpu().numpy(),
        )


class SingleCoilKneeEnv(FastMRIEnv):
    """Convenience class to access single-coil knee data.

    Loads the configuration from ``configs/single-coil-knee.json``.

    Args:
        num_parallel_episodes(int): Determines the number images that will be processed
                                    simultaneously by :meth:`reset()` and :meth:`step()`.
                                    Defaults to 1.
        budget(optional(int)): The length of an acquisition episode. Defaults to ``None``,
                               which indicates that episode will continue until all k-space
                               columns have been acquired.
        seed(optional(int)): The seed for the environment's random number generator, which is
                             an instance of ``numpy.random.RandomState``. Defaults to ``None``.
        num_cols(sequence(int)): Used to filter k-space data to only use images whose k-space
                                 width is in this tuple. Defaults to ``(368, 372)``.
    """

    def __init__(
        self,
        num_parallel_episodes: int = 1,
        budget: Optional[int] = None,
        seed: Optional[int] = None,
        num_cols: Sequence[int] = (368, 372),
    ):
        super().__init__(
            "configs/single-coil-knee.json",
            "knee_singlecoil",
            num_parallel_episodes=num_parallel_episodes,
            budget=budget,
            seed=seed,
            num_cols=num_cols,
        )


class MultiCoilKneeEnv(FastMRIEnv):
    """Convenience class to access multi-coil knee data.

    Loads the configuration from ``configs/multi-coil-knee.json``.

    Args:
        num_parallel_episodes(int): Determines the number images that will be processed
                                    simultaneously by :meth:`reset()` and :meth:`step()`.
                                    Defaults to 1.
        budget(optional(int)): The length of an acquisition episode. Defaults to ``None``,
                               which indicates that episode will continue until all k-space
                               columns have been acquired.
        seed(optional(int)): The seed for the environment's random number generator, which is
                             an instance of ``numpy.random.RandomState``. Defaults to ``None``.
        num_cols(sequence(int)): Used to filter k-space data to only use images whose k-space
                                 width is in this tuple. Defaults to ``(368, 372)``.
    """

    def __init__(
        self,
        num_parallel_episodes: int = 1,
        budget: Optional[int] = None,
        seed: Optional[int] = None,
        num_cols: Sequence[int] = (368, 372),
    ):

        super().__init__(
            "configs/multi-coil-knee.json",
            "multicoil",
            num_parallel_episodes=num_parallel_episodes,
            budget=budget,
            seed=seed,
            num_cols=num_cols,
        )
