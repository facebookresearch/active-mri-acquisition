import abc
import json
import pathlib
import warnings

from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Sized,
    Tuple,
    Union,
)

import gym
import numpy as np
import torch
import torch.utils.data

import activemri.data.singlecoil_knee_data as scknee_data
import activemri.envs.util
import activemri.envs.mask_functions
import activemri.models.singlecoil_knee_transforms as sc_transforms


# -----------------------------------------------------------------------------
#                               DATA HANDLING
# -----------------------------------------------------------------------------
class CyclicSampler(torch.utils.data.Sampler):
    def __init__(
        self, data_source: Sized, order: Optional[Sized] = None, loops: int = 1,
    ):
        torch.utils.data.Sampler.__init__(self, data_source)
        assert loops > 0
        assert order is None or len(order) == len(data_source)
        self.data_source = data_source
        self.order = order if order is not None else range(len(self.data_source))
        self.loops = loops

    def __iterator(self):
        for _ in range(self.loops):
            for j in self.order:
                yield j

    def __iter__(self):
        return iter(self.__iterator())

    def __len__(self):
        return len(self.data_source) * self.loops


def _env_collate_fn(batch):
    ret = [
        torch.stack([item[0] for item in batch]),  # kspace
        torch.stack([item[1] for item in batch]),  # mask
        torch.stack([item[2] for item in batch]),  # target
    ]
    for i in range(3, 6):  # attrs, fname, slice_id
        arg_i = [item[i] for item in batch]
        ret.append(arg_i)

    return tuple(ret)


class DataHandler:
    def __init__(
        self,
        data_source: Sized,
        seed: Optional[int],
        batch_size: int = 1,
        loops: int = 1,
        collate_fn: Optional[Callable] = None,
    ):
        rng = np.random.RandomState(seed)
        order = rng.permutation(len(data_source))
        sampler = CyclicSampler(data_source, order, loops=loops)
        if collate_fn:
            self.__data_loader = torch.utils.data.DataLoader(
                data_source,
                batch_size=batch_size,
                sampler=sampler,
                collate_fn=collate_fn,
            )
        else:
            self.__data_loader = torch.utils.data.DataLoader(
                data_source, batch_size=batch_size, sampler=sampler
            )
        self.__iter = iter(self.__data_loader)

    def reset(self):
        self.__iter = iter(self.__data_loader)

    def __iter__(self):
        return self.__iter

    def __next__(self):
        return next(self.__iter)


# -----------------------------------------------------------------------------
#                           BASE ACTIVE MRI ENV
# -----------------------------------------------------------------------------
# TODO Add option to resize default img size (need to pass this option to reconstructor)
# TODO Add option to control batch size (default 2)
# TODO See if there is a clean way to have access to current image indices from the env
# TODO add reward scaling option
class ActiveMRIEnv(gym.Env):
    def __init__(self, img_width: int, img_height: int):
        # Default initialization
        self._data_location = None
        self._reconstructor = None
        self._transform = None
        self._train_data_handler = None
        self._val_data_handler = None
        self._test_data_handler = None
        self._device = torch.device("cpu")
        self._batch_size = 2

        self.horizon = None
        self.seed = None
        self.rng = np.random.RandomState()
        self.reward_metric = "mse"

        # Init from provided configuration
        self._img_width = img_width
        self._img_height = img_height

        # Gym init
        self.observation_space = None  # Observation will be a dictionary
        self.action_space = gym.spaces.Discrete(img_width)

        self._current_mask = None
        self._current_ground_truth = None
        self._transform_wrapper = None
        self._current_k_space = None
        self._current_score = None

    # -------------------------------------------------------------------------
    # Protected methods
    # -------------------------------------------------------------------------
    def _setup(self, cfg_filename: str, data_init_func: Callable):
        self._init_from_config_file(cfg_filename)
        data_init_func()

    def _init_from_config_dict(self, cfg: Mapping[str, Any]):
        self._data_location = cfg["data_location"]
        self._device = torch.device(cfg["device"])
        self.reward_metric = cfg["reward_metric"]
        if self.reward_metric not in ["mse", "ssim", "psnr", "nmse"]:
            raise ValueError("Reward metric must be one of mse, nmse, ssim, or psnr.")
        mask_func = activemri.envs.util.import_object_from_str(cfg["mask"]["function"])
        self._mask_func = lambda size, rng: mask_func(cfg["mask"]["args"], size, rng)

        # Instantiating reconstructor
        reconstructor_cfg = cfg["reconstructor"]
        reconstructor_cls = activemri.envs.util.import_object_from_str(
            reconstructor_cfg["cls"]
        )
        checkpoint_path = pathlib.Path(reconstructor_cfg["checkpoint_path"])
        checkpoint = torch.load(checkpoint_path) if checkpoint_path.is_file() else None
        options = reconstructor_cfg["options"]
        if checkpoint and "options" in checkpoint:
            msg = (
                f"Checkpoint at {checkpoint_path.name} has an 'options' field. "
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

    def __send_tuple_to_device(self, the_tuple: Tuple[Union[Any, torch.Tensor]]):
        the_tuple_device = []
        for i in range(len(the_tuple)):
            if isinstance(the_tuple[i], torch.Tensor):
                the_tuple_device.append(the_tuple[i].to(self._device))
            else:
                the_tuple_device.append(the_tuple[i])
        return tuple(the_tuple_device)

    @staticmethod
    def __send_dict_to_cpu_and_detach(the_dict: Dict[str, Union[Any, torch.Tensor]]):
        the_dict_cpu = {}
        for key in the_dict:
            if isinstance(the_dict[key], torch.Tensor):
                the_dict_cpu[key] = the_dict[key].detach().cpu()
        return the_dict_cpu

    def __compute_obs_and_score(self) -> Tuple[Dict[str, Any], Dict[str, np.ndarray]]:
        reconstructor_input = self._transform_wrapper(
            self._current_k_space, self._current_mask, self._current_ground_truth
        )

        reconstructor_input = self.__send_tuple_to_device(reconstructor_input)
        extra_outputs = self._reconstructor(*reconstructor_input)

        extra_outputs = self.__send_dict_to_cpu_and_detach(extra_outputs)
        reconstruction = extra_outputs["reconstruction"]
        del extra_outputs["reconstruction"]

        # noinspection PyUnusedLocal
        reconstructor_input = None  # de-referencing GPU tensors

        score = self._compute_score_given_tensors(
            reconstruction, self._current_ground_truth
        )

        obs = {
            "reconstruction": reconstruction,
            "extra_outputs": extra_outputs,
            "mask": self._current_mask,
        }

        return obs, score

    @abc.abstractmethod
    def _compute_score_given_tensors(
        self, reconstruction: torch.Tensor, ground_truth: torch.Tensor
    ) -> Dict[str, np.ndarray]:
        pass

    # -------------------------------------------------------------------------
    # Public methods
    # -------------------------------------------------------------------------
    def reset(self,) -> Dict[str, np.ndarray]:

        kspace, _, ground_truth, attrs, fname, slice_id = next(self._train_data_handler)
        self._current_ground_truth = ground_truth
        self._current_k_space = kspace
        self._transform_wrapper = lambda ks, mask, gt: self._transform(
            ks, mask, gt, attrs, fname, slice_id
        )
        self._current_mask = self._mask_func(self._batch_size, self.rng)
        obs, self._current_score = self.__compute_obs_and_score()

        return obs

    def step(
        self, action: int, batched_actions: Optional[Iterable[int]] = None
    ) -> Tuple[Dict[str, Any], np.ndarray, List[bool], Dict]:
        indices = (
            [action for _ in range(self._batch_size)]
            if batched_actions is None
            else batched_actions
        )
        self._current_mask = activemri.envs.mask_functions.update_masks_from_indices(
            self._current_mask, indices
        )
        obs, new_score = self.__compute_obs_and_score()
        reward = new_score[self.reward_metric] - self._current_score[self.reward_metric]
        if self.reward_metric in ["mse", "nmse"]:
            reward *= -1
        self._current_score = new_score

        done = activemri.envs.mask_functions.check_masks_complete(self._current_mask)
        return obs, reward, done, {}

    def render(self, mode="human"):
        pass

    def seed(self, seed: Optional[int] = None):
        self.seed = seed
        self.rng = np.random.RandomState(seed)

    def reset_val(self,) -> Dict[str, np.ndarray]:
        pass

    def step_val(
        self, action: int
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], float, bool]:
        pass

    def reset_test(self,) -> Dict[str, np.ndarray]:
        pass

    def step_test(
        self, action: int
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], float, bool]:
        pass

    def restart_train(self):
        self._train_data_handler.reset()

    def restart_val(self):
        self._val_data_handler.reset()

    def restart_test(self):
        self._test_data_handler.reset()

    def valid_actions(self) -> List[int]:
        pass


# -----------------------------------------------------------------------------
#                             CUSTOM ENVIRONMENTS
# -----------------------------------------------------------------------------
class SingleCoilKneeRAWEnv(ActiveMRIEnv):
    IMAGE_WIDTH = scknee_data.RawSliceData.IMAGE_WIDTH
    IMAGE_HEIGHT = scknee_data.RawSliceData.IMAGE_HEIGHT
    START_PADDING = scknee_data.RawSliceData.START_PADDING
    END_PADDING = scknee_data.RawSliceData.END_PADDING
    CENTER_CROP_SIZE = scknee_data.RawSliceData.CENTER_CROP_SIZE

    def __init__(self):
        ActiveMRIEnv.__init__(self, self.IMAGE_WIDTH, self.IMAGE_HEIGHT)
        self._setup("configs/miccai_raw.json", self.__setup_data_handlers)

    # -------------------------------------------------------------------------
    # Private methods
    # -------------------------------------------------------------------------
    def __setup_data_handlers(self):
        root_path = pathlib.Path(self._data_location)
        train_path = root_path.joinpath("knee_singlecoil_train")
        val_and_test_path = root_path.joinpath("knee_singlecoil_val")

        # Setting up training data loader
        train_data = scknee_data.RawSliceData(
            train_path, activemri.envs.transform, num_cols=self.IMAGE_WIDTH,
        )
        self._train_data_handler = DataHandler(
            train_data,
            self.seed,
            batch_size=self._batch_size,
            loops=1000,
            collate_fn=_env_collate_fn,
        )
        # Setting up val data loader
        val_data = scknee_data.RawSliceData(
            val_and_test_path,
            activemri.envs.transform,
            custom_split="val",
            num_cols=self.IMAGE_WIDTH,
        )
        self._val_data_handler = DataHandler(
            val_data,
            self.seed + 1 if self.seed else None,
            batch_size=self._batch_size,
            loops=1,
            collate_fn=_env_collate_fn,
        )
        # Setting up test data loader
        test_data = scknee_data.RawSliceData(
            val_and_test_path,
            activemri.envs.transform,
            custom_split="test",
            num_cols=self.IMAGE_WIDTH,
        )
        self._test_data_handler = DataHandler(
            test_data,
            self.seed + 2 if self.seed else None,
            batch_size=self._batch_size,
            loops=1,
            collate_fn=_env_collate_fn,
        )

    def reset(self,) -> Dict[str, np.ndarray]:
        obs = super().reset()
        obs["mask"][:, self.START_PADDING : self.END_PADDING] = 1
        return obs

    def _compute_score_given_tensors(
        self, reconstruction: torch.Tensor, ground_truth: torch.Tensor
    ) -> Dict[str, np.ndarray]:
        # Compute magnitude (for metrics)
        reconstruction = sc_transforms.to_magnitude(reconstruction, dim=3)
        ground_truth = sc_transforms.to_magnitude(ground_truth, dim=3)

        reconstruction = sc_transforms.center_crop(
            reconstruction, [self.CENTER_CROP_SIZE, self.CENTER_CROP_SIZE]
        )
        ground_truth = sc_transforms.center_crop(
            ground_truth, [self.CENTER_CROP_SIZE, self.CENTER_CROP_SIZE]
        )

        mse = activemri.envs.util.compute_mse(reconstruction, ground_truth)
        nmse = activemri.envs.util.compute_nmse(reconstruction, ground_truth)
        ssim = activemri.envs.util.compute_ssim(reconstruction, ground_truth)
        psnr = activemri.envs.util.compute_psnr(reconstruction, ground_truth)

        return {"mse": mse, "nmse": nmse, "ssim": ssim, "psnr": psnr}
