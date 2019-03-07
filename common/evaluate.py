import logging
import time

import torch
import sys
import math
from timeit import default_timer as timer
import datetime
from torch.nn import functional as F
from torch.utils.data import DataLoader

from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist

from common import dicom_dataset, subsample
from common import pytorch_mssim as ssim_metric

def evaluate_model(which_dataset, model, args):
    """
    Evaluates the model on the validation data

    :param which_dataset: Which dataset to evaluate on. Choices are `val` or `public_leaderboard`
    :param model: The model to evaluate. The model should be callable with a batch of masked k-space images
                  and masks as inputs and return a batch of predicted images.
    :param args: Standard args struct containing batchsize, number of workers etc.
    :return: A dictionary with the computed evaluation metrics.
    """
    assert which_dataset in {'val', 'public_leaderboard'}
    start_time = time.perf_counter()
    mask_func = subsample.Mask(reuse_mask=True)
    dataset = dicom_dataset.Slice(
        mask_func=mask_func,
        args=args,
        which=which_dataset,
    )
    device = torch.cuda.current_device()

    if args.distributed:
        sampler = DistributedSampler(dataset)
        sampler.set_epoch(0)
        group = dist.new_group(range(dist.get_world_size()))
    else:
        sampler = None

    batch_size = args.eval_batch_size
    if batch_size == -1:
        batch_size = args.batch_size

    data_loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=args.workers,
        sampler=sampler,
        pin_memory=True
    )

    neval = len(data_loader)*batch_size

    metrics = {
        'mse': 0,
        'ssim': 0,
        'msssim': 0
    }

    if args.log_during_eval:
        logging.info(f"Evaluating, batchsize: {batch_size}")

    model.eval()
    tmptensor = torch.zeros(1).cuda()
    start = timer()

    with torch.no_grad():
        num_images = 0
        num_batches = 0
        for masked_kspace, image, mask, metadata in data_loader:
            #if not args.distributed:
            masked_kspace = masked_kspace.to(device)
            mask = mask.to(device)
            image = image.to(device)

            if args.apex:
                image = image.half()
                masked_kspace = masked_kspace.half()
                mask = mask.half()


            prediction = model(masked_kspace, mask)

            metrics['mse'] += F.mse_loss(prediction, image).item() * image.shape[0]
            metrics['ssim'] += ssim_metric.ssim(prediction, image).item() * image.shape[0]
            metrics['msssim'] += ssim_metric.msssim(prediction, image).item() * image.shape[0]
            num_images += image.shape[0]
            num_batches += 1

            if args.log_during_eval and num_images % args.log_every == 0:
                mid = timer()
                percent_done = 100. * num_images / neval
                if percent_done > 0:
                    inst_estimate = math.ceil((mid - start)*(100/percent_done))
                    inst_estimate = str(datetime.timedelta(seconds=inst_estimate))
                else:
                    inst_estimate = "unknown"
                logging.info("[{}/{} ({:.0f}%)] MSE: {:.4f} SSIM {:.4f} MSSSIM {:.4f}   Est. {}".format(
                num_images, neval, 100. * num_images / neval,
                metrics['mse']/num_images, metrics['ssim']/num_images, metrics['msssim']/num_images, inst_estimate
                ))
                sys.stdout.flush()

            if args.distributed and num_batches % 16 == 0:
                # Essentially a barrier, to prevent the pace from diverging too much between machines.
                #tmptensor[0] = metrics['mse']
                dist.all_reduce(tmptensor, op=dist.reduce_op.SUM, group=group)

    for k in sorted(metrics.keys()):
        metrics[k] = metrics[k] / num_images
        # All-reduce to communicate metrics between processes
        if args.distributed:
            logging.info(f"Before reduce: {metrics[k]}")
            sys.stdout.flush()
            tmptensor[0] = metrics[k]
            dist.all_reduce(tmptensor, op=dist.reduce_op.SUM, group=group)
            metrics[k] = tmptensor.item()/dist.get_world_size()
            logging.info(f"After reduce: {metrics[k]}")
            sys.stdout.flush()

    metrics_strs = [f'{key}: {value:.4f}' for key, value in metrics.items()]
    duration = time.perf_counter() - start_time
    logging.info(f'Evaluated model in {duration}s ({num_images} images). {metrics_strs}')

    return metrics
