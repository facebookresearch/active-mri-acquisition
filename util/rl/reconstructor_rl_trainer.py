import argparse

import numpy as np
import tensorboardX
import torch.optim as optim
import torch.utils.data

import models.fft_utils
import models.reconstruction

from typing import Dict


class DatasetFromActiveAcq(torch.utils.data.Dataset):

    def __init__(self, original_dataset: torch.utils.data.Dataset,
                 masks_dict: Dict[int, np.ndarray]):
        # masks_dict is [image_index, matrix of N * W columns of masks for each image]
        super(DatasetFromActiveAcq, self).__init__()
        self.original_dataset = original_dataset
        self.masks_dict = masks_dict
        self.image_indices = [int(x) for x in masks_dict.keys()]
        self.rng = np.random.RandomState(1337)

    def __getitem__(self, index):
        if self.rng.random_sample() < 0.1:
            # With small probability return a random image/mask pair from original dataset
            # as a form of regularization
            return self.original_dataset.__getitem__(self.rng.choice(len(self.original_dataset)))
        image_index = self.image_indices[index]
        _, image = self.original_dataset.__getitem__(image_index)  # assumes DICOM
        mask_index = int(self.rng.beta(1, 5) * self.masks_dict[image_index].shape[0])
        mask = torch.from_numpy(self.masks_dict[image_index][mask_index])
        return mask.view(1, 1, -1), image

    def __len__(self):
        return len(self.image_indices)


class ReconstructorRLTrainer:

    def __init__(self, reconstructor: models.reconstruction.ReconstructorNetwork,
                 original_dataset: torch.utils.data.Dataset, options: argparse.Namespace):
        self.original_dataset = original_dataset
        self.reconstructor = reconstructor
        self.options = options
        self.optimizer = optim.Adam(
            self.reconstructor.parameters(), lr=self.options.reconstructor_lr)
        self.num_epochs = 0

    def __call__(self, masks_dict: Dict[int, np.ndarray], writer: tensorboardX.SummaryWriter):
        # masks_dict is [image_index, matrix of N * W columns of masks for each image]
        # Only image indices appearing in this dictionary will be considered
        dataset = DatasetFromActiveAcq(self.original_dataset, masks_dict)
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=40)
        self.reconstructor.train()
        for i in range(self.options.num_epochs_train_reconstructor):
            total_loss = 0
            for batch in data_loader:
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

            self.num_epochs += 1
            writer.add_scalar('reconstructor_loss', total_loss, self.num_epochs)

        self.reconstructor.eval()

        return 0
