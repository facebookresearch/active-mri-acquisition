import importlib

import numpy as np
import skimage.metrics


def import_object_from_str(classname: str):
    the_module = ".".join(classname.split(".")[:-1])
    the_object = classname.split(".")[-1]
    module = importlib.import_module(the_module)
    return getattr(module, the_object)


def compute_ssim(xs, ys):
    ssims = []
    for i in range(xs.shape[0]):
        ssim = skimage.metrics.structural_similarity(
            xs[i, ..., 0].cpu().numpy(),
            ys[i, ..., 0].cpu().numpy(),
            data_range=ys[i, ..., 0].cpu().numpy().max(),
        )
        ssims.append(ssim)
    return np.array(ssims).mean()


def compute_psnr(xs, ys):
    psnrs = []
    for i in range(xs.shape[0]):
        psnr = skimage.metrics.peak_signal_noise_ratio(
            xs[i, ..., 0].cpu().numpy(),
            ys[i, ..., 0].cpu().numpy(),
            data_range=ys[i, ..., 0].cpu().numpy().max(),
        )
        psnrs.append(psnr)
    return np.array(psnrs).mean()


def compute_mse(xs, ys):
    return np.mean((ys.cpu().numpy() - xs.cpu().numpy()) ** 2)


def compute_nmse(xs, ys):
    ys_numpy = ys.cpu().numpy()
    return (
        np.linalg.norm(ys_numpy - xs.cpu().numpy()) ** 2 / np.linalg.norm(ys_numpy) ** 2
    )
