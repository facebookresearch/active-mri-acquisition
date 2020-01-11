import argparse
import logging

import numpy as np
import tensorboardX
import torch.optim as optim
import torch.utils.data

import models.fft_utils
import models.reconstruction

from typing import Callable, Dict, Tuple


def split_masks_dict_train_val(masks_dict: Dict[int, np.ndarray], ratio: float = 0.1
                              ) -> Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray]]:
    n = len(masks_dict)
    val_indices = np.random.choice(list(masks_dict.keys()), int(n * ratio), replace=False)
    masks_dict_val = dict([(k, masks_dict[k]) for k in val_indices])
    masks_dict_train = {}
    for k in masks_dict.keys():
        if k not in val_indices:
            masks_dict_train[k] = masks_dict[k]
    return masks_dict_train, masks_dict_val


class DatasetFromActiveAcq(torch.utils.data.Dataset):
    # Creates a dataset of images loaded from a given dataset and masks from a given dictionary.
    # The dictionary associates image indices to an ndarray of several masks, from where masks
    # will be sampled from.

    def __init__(self, original_dataset: torch.utils.data.Dataset,
                 masks_dict: Dict[int, np.ndarray], dataroot: str):
        # masks_dict is [image_index, matrix of N * W columns of masks for each image]
        super(DatasetFromActiveAcq, self).__init__()
        self.original_dataset = original_dataset
        self.masks_dict = masks_dict
        self.image_indices = [int(x) for x in masks_dict.keys()]
        self.rng = np.random.RandomState()
        self.dataroot = dataroot

    def __getitem__(self, index):
        if self.rng.random_sample() < 0.01:
            # With small probability return a random image/mask pair from original dataset
            # as a form of regularization
            return self.original_dataset.__getitem__(self.rng.choice(len(self.original_dataset)))
        image_index = self.image_indices[index]
        masK_image_raw = self.original_dataset.__getitem__(image_index)
        if self.dataroot == 'KNEE':
            mask_index = int(self.rng.beta(1, 5) * self.masks_dict[image_index].shape[0])
            mask = torch.from_numpy(self.masks_dict[image_index][mask_index])
            return mask.view(1, 1, -1), masK_image_raw[1]
        elif self.dataroot == 'KNEE_RAW':
            mask_index = int(self.rng.beta(1, 10) * self.masks_dict[image_index].shape[0])
            mask = torch.from_numpy(self.masks_dict[image_index][mask_index])
            return mask.view(1, 1, -1), masK_image_raw[1], masK_image_raw[2]
        else:
            raise NotImplementedError('Only supported for options.dataroot KNEE or KNEE_RAW')

    def __len__(self):
        return len(self.image_indices)


class ReconstructorRLTrainer:

    def __init__(self, reconstructor: models.reconstruction.ReconstructorNetwork,
                 original_train_dataset: torch.utils.data.Dataset, options: argparse.Namespace,
                 reconstructor_checkpoint_callback: Callable[
                     [models.reconstruction.ReconstructorNetwork, int], None]):
        self.original_train_dataset = original_train_dataset
        # Use all GPUs except the last one, which is used for DQN
        self.reconstructor = torch.nn.DataParallel(reconstructor,
                                                   list(range(torch.cuda.device_count() - 1)))
        self.options = options
        self.optimizer = optim.Adam(
            self.reconstructor.parameters(), lr=self.options.reconstructor_lr)
        self.reconstructor_checkpoint_callback = reconstructor_checkpoint_callback

    def do_validation_loop(self, epoch: int, data_loader_validation: torch.utils.data.DataLoader,
                           logger: logging.Logger, writer: tensorboardX.SummaryWriter):
        self.reconstructor.eval()
        total_loss_valid = 0
        for batch in data_loader_validation:
            with torch.no_grad():
                zero_filled_image, target, mask = models.fft_utils.preprocess_inputs(
                    batch, self.options.dataroot, self.options.device)

                # Get reconstructor output
                reconstructed_image, uncertainty_map, mask_embedding = self.reconstructor(
                    zero_filled_image, mask)

                loss = models.fft_utils.gaussian_nll_loss(reconstructed_image, target,
                                                          uncertainty_map, self.options).mean()

                total_loss_valid += loss.item()

        logger.info(f'Reconstructor loss validation: {total_loss_valid}')
        writer.add_scalar('valid/reconstructor_loss', total_loss_valid, epoch)

        return total_loss_valid

    def __call__(self, start_epoch_for_logs: int, masks_dict: Dict[int, np.ndarray],
                 logger: logging.Logger, writer: tensorboardX.SummaryWriter):
        epoch = start_epoch_for_logs
        self.reconstructor.to(torch.device('cuda'))
        num_workers = 2 * len(self.reconstructor.device_ids)

        logger.info(f'Training reconstructor with device IDS: {self.reconstructor.device_ids}')
        masks_dict_train, masks_dict_valid = split_masks_dict_train_val(masks_dict, ratio=0.1)
        logger.info(masks_dict_train.keys())
        logger.info(masks_dict_valid.keys())

        # masks_dict is [image_index, matrix of N * W columns of masks for each image]
        # Only image indices appearing in this dictionary will be considered
        dataset_train = DatasetFromActiveAcq(self.original_train_dataset, masks_dict_train,
                                             self.options.dataroot)
        logger.info(f'Created train dataset of len {len(dataset_train)}.')
        data_loader_train = torch.utils.data.DataLoader(
            dataset_train,
            batch_size=self.options.alternate_opt_batch_size,
            num_workers=num_workers)
        logger.info(f'Created dataloader for this with {len(data_loader_train)} batches.')

        dataset_valid = DatasetFromActiveAcq(self.original_train_dataset, masks_dict_valid,
                                             self.options.dataroot)
        logger.info(f'Created validation dataset of len {len(dataset_train)}.')
        data_loader_validation = torch.utils.data.DataLoader(
            dataset_valid,
            batch_size=self.options.alternate_opt_batch_size,
            num_workers=num_workers)
        logger.info(f'Created validation dataloader with {len(data_loader_validation)} batches.')

        best_validation_score = self.do_validation_loop(epoch, data_loader_validation, logger,
                                                        writer)
        for i in range(self.options.num_epochs_train_reconstructor):
            epoch += 1
            logger.info(f'Starting epoch {i + 1}/{self.options.num_epochs_train_reconstructor}')
            total_loss = 0
            self.reconstructor.train()
            for batch in data_loader_train:
                self.optimizer.zero_grad()

                zero_filled_image, target, mask = models.fft_utils.preprocess_inputs(
                    batch, self.options.dataroot, self.options.device)

                # Get reconstructor output
                reconstructed_image, uncertainty_map, mask_embedding = self.reconstructor(
                    zero_filled_image, mask)

                loss = models.fft_utils.gaussian_nll_loss(reconstructed_image, target,
                                                          uncertainty_map, self.options).mean()
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

            logger.info(f'Reconstructor loss: {total_loss}')
            writer.add_scalar('train/reconstructor_loss', total_loss, epoch)

            validation_score = self.do_validation_loop(epoch, data_loader_validation, logger,
                                                       writer)

            if validation_score < best_validation_score:
                logger.info(f'Found a model with a better validation score of {validation_score}.')
                best_validation_score = validation_score
                self.reconstructor_checkpoint_callback(self.reconstructor, epoch)

        self.reconstructor.to('cpu')
