import importlib

import numpy as np
import skimage.metrics
import torch


def import_object_from_str(classname: str):
    the_module, the_object = classname.rsplit(".", 1)
    the_object = classname.split(".")[-1]
    module = importlib.import_module(the_module)
    return getattr(module, the_object)


def compute_ssim(xs: torch.Tensor, ys: torch.Tensor) -> np.ndarray:
    ssims = []
    for i in range(xs.shape[0]):
        ssim = skimage.metrics.structural_similarity(
            xs[i].cpu().numpy(),
            ys[i].cpu().numpy(),
            data_range=ys[i].cpu().numpy().max(),
        )
        ssims.append(ssim)
    return np.array(ssims, dtype=np.float32)


def compute_psnr(xs: torch.Tensor, ys: torch.Tensor) -> np.ndarray:
    psnrs = []
    for i in range(xs.shape[0]):
        psnr = skimage.metrics.peak_signal_noise_ratio(
            xs[i].cpu().numpy(),
            ys[i].cpu().numpy(),
            data_range=ys[i].cpu().numpy().max(),
        )
        psnrs.append(psnr)
    return np.array(psnrs, dtype=np.float32)


def compute_mse(xs: torch.Tensor, ys: torch.Tensor) -> np.ndarray:
    dims = tuple(range(1, len(xs.shape)))
    return np.mean((ys.cpu().numpy() - xs.cpu().numpy()) ** 2, axis=dims)


def compute_nmse(xs: torch.Tensor, ys: torch.Tensor) -> np.ndarray:
    ys_numpy = ys.cpu().numpy()
    nmses = []
    for i in range(xs.shape[0]):
        x = xs[i].cpu().numpy()
        y = ys_numpy[i]
        nmse = np.linalg.norm(y - x) ** 2 / np.linalg.norm(y) ** 2
        nmses.append(nmse)
    return np.array(nmses, dtype=np.float32)
