import abc
import functools
import json
import pathlib
import warnings
from typing import (
    Any,
    Callable,
    Dict,
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
import activemri.envs.masks
import activemri.envs.util
import activemri.models.singlecoil_knee_transforms as scknee_transforms

DataInitFnReturnType = Tuple[
    torch.utils.data.Dataset, torch.utils.data.Dataset, torch.utils.data.Dataset
]


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

    def _iterator(self):
        for _ in range(self.loops):
            for j in self.order:
                yield j

    def __iter__(self):
        return iter(self._iterator())

    def __len__(self):
        return len(self.data_source) * self.loops


# TODO Add a fastMRI batch type (make it the return of void_transform)
# noinspection PyUnresolvedReferences
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
        self._iter = None
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
    _num_loops_train_data = 100000

    def __init__(
        self,
        img_width: int,
        img_height: int,
        batch_size: int = 1,
        budget: Optional[int] = None,
        seed: Optional[int] = None,
    ):
        # Default initialization
        self._cfg = None
        self._data_location = None
        self._reconstructor = None
        self._transform = None
        self._train_data_handler = None
        self._val_data_handler = None
        self._test_data_handler = None
        self._device = torch.device("cpu")
        self.batch_size = batch_size
        self.budget = budget

        self.action_space = gym.spaces.Discrete(img_width)

        self._seed = seed
        self._rng = np.random.RandomState(seed)
        self.reward_metric = "mse"

        # Init from provided configuration
        self.img_width = img_width
        self.img_height = img_height

        # Gym init
        # Observation is a dictionary
        self.observation_space = None
        self.action_space = gym.spaces.Discrete(img_width)

        # This is changed by `set_training()`, `set_val()`, `set_test()`
        self._current_data_handler = None

        # These are changed every call to `reset()`
        self._current_ground_truth = None
        self._transform_wrapper = None
        self._current_k_space = None
        self._did_reset = False
        self._steps_since_reset = 0
        # These three are changed every call to `reset()` and every call to `step()`
        self._current_reconstruction_numpy = None
        self._current_score = None
        self._current_mask = None

    # -------------------------------------------------------------------------
    # Protected methods
    # -------------------------------------------------------------------------
    def _setup(
        self, cfg_filename: str, data_init_func: Callable[[], DataInitFnReturnType],
    ):
        self._init_from_config_file(cfg_filename)
        self._setup_data_handlers(data_init_func)

    def _setup_data_handlers(
        self, data_init_func: Callable[[], DataInitFnReturnType],
    ):
        train_data, val_data, test_data = data_init_func()
        self._train_data_handler = DataHandler(
            train_data,
            self._seed,
            batch_size=self.batch_size,
            loops=self._num_loops_train_data,
            collate_fn=_env_collate_fn,
        )
        self._val_data_handler = DataHandler(
            val_data,
            self._seed + 1 if self._seed else None,
            batch_size=self.batch_size,
            loops=1,
            collate_fn=_env_collate_fn,
        )
        self._test_data_handler = DataHandler(
            test_data,
            self._seed + 2 if self._seed else None,
            batch_size=self.batch_size,
            loops=1,
            collate_fn=_env_collate_fn,
        )
        self._current_data_handler = self._train_data_handler

    def _init_from_config_dict(self, cfg: Mapping[str, Any]):
        self._cfg = cfg
        self._data_location = cfg["data_location"]
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
        checkpoint_path = pathlib.Path(reconstructor_cfg["checkpoint_path"])
        checkpoint = (
            torch.load(str(checkpoint_path)) if checkpoint_path.is_file() else None
        )
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
            reconstruction, self._current_ground_truth
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

    @abc.abstractmethod
    def _compute_score_given_tensors(
        self, reconstruction: torch.Tensor, ground_truth: torch.Tensor
    ) -> Dict[str, np.ndarray]:
        pass

    # -------------------------------------------------------------------------
    # Public methods
    # -------------------------------------------------------------------------
    def reset(self,) -> Tuple[Dict[str, Any], Dict[str, Any]]:
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

    # TODO look how to handle the batch_size=1 special case
    # TODO change batch_size by parallel episodes
    def step(
        self, action: Union[int, Sequence[int]]
    ) -> Tuple[Dict[str, Any], np.ndarray, List[bool], Dict]:
        if not self._did_reset:
            raise RuntimeError(
                "Attempting to call env.step() before calling env.reset()."
            )
        if isinstance(action, int):
            action = [action for _ in range(self.batch_size)]
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
        if not self._did_reset:
            raise RuntimeError(
                "Attempting to call env.try_action() before calling env.reset()."
            )
        if isinstance(action, int):
            action = [action for _ in range(self.batch_size)]
        new_mask = activemri.envs.masks.update_masks_from_indices(
            self._current_mask, action
        )
        obs, new_score = self._compute_obs_and_score(override_current_mask=new_mask)

        return obs, new_score

    def render(self, mode="human"):
        pass

    def seed(self, seed: Optional[int] = None):
        self._seed = seed
        self._rng = np.random.RandomState(seed)
        self._train_data_handler.seed(seed)
        self._val_data_handler.seed(seed)
        self._test_data_handler.seed(seed)

    def set_training(self, reset: bool = False):
        if reset:
            self._train_data_handler.reset()
        self._current_data_handler = self._train_data_handler
        self._clear_cache_and_unset_did_reset()

    def set_val(self, reset: bool = True):
        if reset:
            self._val_data_handler.reset()
        self._current_data_handler = self._val_data_handler
        self._clear_cache_and_unset_did_reset()

    def set_test(self, reset: bool = True):
        if reset:
            self._test_data_handler.reset()
        self._current_data_handler = self._test_data_handler
        self._clear_cache_and_unset_did_reset()

    @abc.abstractmethod
    def score_keys(self) -> List[str]:
        pass


# -----------------------------------------------------------------------------
#                             CUSTOM ENVIRONMENTS
# -----------------------------------------------------------------------------
class MICCAI2020Env(ActiveMRIEnv):
    IMAGE_WIDTH = scknee_data.MICCAI2020Data.IMAGE_WIDTH
    IMAGE_HEIGHT = scknee_data.MICCAI2020Data.IMAGE_HEIGHT
    START_PADDING = scknee_data.MICCAI2020Data.START_PADDING
    END_PADDING = scknee_data.MICCAI2020Data.END_PADDING
    CENTER_CROP_SIZE = scknee_data.MICCAI2020Data.CENTER_CROP_SIZE

    def __init__(self, batch_size: int = 1, budget: Optional[int] = None):
        super().__init__(
            self.IMAGE_WIDTH, self.IMAGE_HEIGHT, batch_size=batch_size, budget=budget
        )
        self._setup("configs/miccai-2020.json", self._create_dataset)

    # -------------------------------------------------------------------------
    # Protected methods
    # -------------------------------------------------------------------------
    def _create_dataset(self) -> DataInitFnReturnType:
        root_path = pathlib.Path(self._data_location)
        train_path = root_path / "knee_singlecoil_train"
        val_and_test_path = root_path / "knee_singlecoil_val"

        train_data = scknee_data.MICCAI2020Data(
            train_path, ActiveMRIEnv._void_transform, num_cols=self.IMAGE_WIDTH,
        )
        val_data = scknee_data.MICCAI2020Data(
            val_and_test_path,
            ActiveMRIEnv._void_transform,
            custom_split="val",
            num_cols=self.IMAGE_WIDTH,
        )
        test_data = scknee_data.MICCAI2020Data(
            val_and_test_path,
            ActiveMRIEnv._void_transform,
            custom_split="test",
            num_cols=self.IMAGE_WIDTH,
        )
        return train_data, val_data, test_data

    def _compute_score_given_tensors(
        self, reconstruction: torch.Tensor, ground_truth: torch.Tensor
    ) -> Dict[str, np.ndarray]:
        # Compute magnitude (for metrics)
        reconstruction = scknee_transforms.to_magnitude(reconstruction, dim=3)
        ground_truth = scknee_transforms.to_magnitude(ground_truth, dim=3)

        reconstruction = scknee_transforms.center_crop(
            reconstruction, (self.CENTER_CROP_SIZE, self.CENTER_CROP_SIZE)
        )
        ground_truth = scknee_transforms.center_crop(
            ground_truth, (self.CENTER_CROP_SIZE, self.CENTER_CROP_SIZE)
        )

        mse = activemri.envs.util.compute_mse(reconstruction, ground_truth)
        nmse = activemri.envs.util.compute_nmse(reconstruction, ground_truth)
        ssim = activemri.envs.util.compute_ssim(reconstruction, ground_truth)
        psnr = activemri.envs.util.compute_psnr(reconstruction, ground_truth)

        return {"mse": mse, "nmse": nmse, "ssim": ssim, "psnr": psnr}

    # -------------------------------------------------------------------------
    # Public methods
    # -------------------------------------------------------------------------
    def reset(self,) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        obs, meta = super().reset()
        if not obs:
            return obs, meta
        obs["mask"][:, self.START_PADDING : self.END_PADDING] = 1
        return obs, meta

    def render(self, mode="human"):
        img_tensor = self._current_reconstruction_numpy.cpu().unsqueeze(0)
        img_tensor = scknee_transforms.to_magnitude(img_tensor, dim=3)
        img_tensor = scknee_transforms.center_crop(
            img_tensor, (self.CENTER_CROP_SIZE, self.CENTER_CROP_SIZE)
        )
        img_tensor = img_tensor.view(self.CENTER_CROP_SIZE, self.CENTER_CROP_SIZE, 1)
        img = img_tensor.numpy()
        t_min = img.min()
        t_max = img.max()
        img = 255 * (img - t_min) / (t_max - t_min)
        return img.astype(np.uint8)

    def score_keys(self) -> List[str]:
        return ["mse", "nmse", "ssim", "psnr"]


class FastMRIEnv(ActiveMRIEnv):
    def __init__(
        self,
        config_path: str,
        dataset_name: str,
        challenge: str,
        batch_size: int = 1,
        budget: Optional[int] = None,
        num_cols: Sequence[int] = (368, 372),
    ):
        assert challenge in ["singlecoil", "multicoil"]
        super().__init__(320, 320, batch_size=batch_size, budget=budget)
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

    # TODO replace for the functions used in fastmri
    def _compute_score_given_tensors(
        self, reconstruction: torch.Tensor, ground_truth: torch.Tensor
    ) -> Dict[str, np.ndarray]:
        mse = activemri.envs.util.compute_mse(reconstruction, ground_truth)
        nmse = activemri.envs.util.compute_nmse(reconstruction, ground_truth)
        ssim = activemri.envs.util.compute_ssim(reconstruction, ground_truth)
        psnr = activemri.envs.util.compute_psnr(reconstruction, ground_truth)

        return {"mse": mse, "nmse": nmse, "ssim": ssim, "psnr": psnr}

    def score_keys(self) -> List[str]:
        return ["mse", "nmse", "ssim", "psnr"]

    def render(self, mode="human"):
        gt = self._current_ground_truth.cpu().numpy()
        rec = self._current_reconstruction_numpy

        frames = []
        for i in range(gt.shape[0]):
            scale = np.quantile(gt[i], 0.75)
            mask = (
                self._current_mask[i].cpu().repeat(self.img_height, 1).numpy() * scale
            )

            pad = 30
            mask_begin = pad
            mask_end = mask_begin + mask.shape[-1]
            gt_begin = mask_end + pad
            gt_end = gt_begin + self.img_width
            rec_begin = gt_end + pad
            rec_end = rec_begin + self.img_width
            frame = 0.4 * scale * np.ones((self.img_height, rec_end + pad))
            frame[:, mask_begin:mask_end] = mask
            frame[:, gt_begin:gt_end] = gt[i]
            frame[:, rec_begin:rec_end] = rec[i]

            frames.append(frame)
        return frames


class SingleCoilKneeEnv(FastMRIEnv):
    def __init__(
        self,
        batch_size: int = 1,
        budget: Optional[int] = None,
        num_cols: Sequence[int] = (368, 372),
    ):
        super().__init__(
            "configs/single-coil-knee.json",
            "knee_singlecoil",
            "singlecoil",
            batch_size=batch_size,
            budget=budget,
            num_cols=num_cols,
        )


class MultiCoilKneeEnv(FastMRIEnv):
    def __init__(
        self,
        batch_size: int = 1,
        budget: Optional[int] = None,
        num_cols: Sequence[int] = (368, 372),
    ):

        super().__init__(
            "configs/multi-coil-knee.json",
            "multicoil",
            "multicoil",
            batch_size=batch_size,
            budget=budget,
            num_cols=num_cols,
        )
