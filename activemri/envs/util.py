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
    return np.array(ssims, dtype=np.float32)


def compute_psnr(xs, ys):
    psnrs = []
    for i in range(xs.shape[0]):
        psnr = skimage.metrics.peak_signal_noise_ratio(
            xs[i, ..., 0].cpu().numpy(),
            ys[i, ..., 0].cpu().numpy(),
            data_range=ys[i, ..., 0].cpu().numpy().max(),
        )
        psnrs.append(psnr)
    return np.array(psnrs, dtype=np.float32)


def compute_mse(xs, ys):
    dims = tuple(range(1, len(xs.shape)))
    return np.mean((ys.cpu().numpy() - xs.cpu().numpy()) ** 2, axis=dims)


def compute_nmse(xs, ys):
    ys_numpy = ys.cpu().numpy()
    nmses = []
    for i in range(xs.shape[0]):
        x = xs[i, ..., 0].cpu().numpy()
        y = ys_numpy[i, ..., 0]
        nmse = np.linalg.norm(y - x) ** 2 / np.linalg.norm(y) ** 2
        nmses.append(nmse)
    return np.array(nmses, dtype=np.float32)
