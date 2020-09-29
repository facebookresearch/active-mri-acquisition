# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
cvpr19_models.models.evaluator.py
=================================
Active acquisition model as described in `Zhang, Zizhao, et al. "Reducing uncertainty in
undersampled mri reconstruction with active acquisition." Proceedings of the IEEE Conference on
Computer Vision and Pattern Recognition. 2019.`
"""
import functools
from typing import Optional

import torch
import torch.nn as nn

from . import fft_utils, reconstruction


class SimpleSequential(nn.Module):
    def __init__(self, net1, net2):
        super(SimpleSequential, self).__init__()
        self.net1 = net1
        self.net2 = net2

    def forward(self, x, mask):
        output = self.net1(x, mask)
        return self.net2(output, mask)


class SpectralMapDecomposition(nn.Module):
    def __init__(self):
        super(SpectralMapDecomposition, self).__init__()

    def forward(self, reconstructed_image, mask_embedding, mask):
        batch_size = reconstructed_image.shape[0]
        height = reconstructed_image.shape[2]
        width = reconstructed_image.shape[3]

        # create spectral maps in kspace
        kspace = fft_utils.fft(reconstructed_image)
        kspace = kspace.unsqueeze(1).repeat(1, width, 1, 1, 1)

        # separate image into spectral maps
        separate_mask = torch.zeros([1, width, 1, 1, width], dtype=torch.float32)
        for i in range(width):
            separate_mask[0, i, 0, 0, i] = 1

        separate_mask = separate_mask.to(reconstructed_image.device)

        masked_kspace = torch.where(
            separate_mask.byte(), kspace, torch.tensor(0.0).to(kspace.device)
        )
        masked_kspace = masked_kspace.view(batch_size * width, 2, height, width)

        # convert spectral maps to image space
        separate_images = fft_utils.ifft(masked_kspace)
        # result is (batch, [real_M0, img_M0, real_M1, img_M1, ...],  height, width]
        separate_images = separate_images.contiguous().view(
            batch_size, 2, width, height, width
        )

        # add mask information as a summation -- might not be optimal
        if mask is not None:
            separate_images = (
                separate_images + mask.permute(0, 3, 1, 2).unsqueeze(1).detach()
            )

        separate_images = separate_images.contiguous().view(
            batch_size, 2 * width, height, width
        )
        # concatenate mask embedding
        if mask_embedding is not None:
            spectral_map = torch.cat([separate_images, mask_embedding], dim=1)
        else:
            spectral_map = separate_images

        return spectral_map


class EvaluatorNetwork(nn.Module):
    """Evaluator network used in Zhang et al., CVPR'19.

    Args:
        number_of_filters(int): Number of filters used in convolutions. Defaults to 256. \n
        number_of_conv_layers(int): Depth of the model defined as a number of
                convolutional layers. Defaults to 4.
        use_sigmoid(bool): Whether the sigmoid non-linearity is applied to the
                output of the network. Defaults to False.
        width(int): The width of the image. Defaults to 128 (corresponds to DICOM).
        height(Optional[int]): The height of the image. If ``None`` the value of ``width``.
            is used. Defaults to ``None``.
        mask_embed_dim(int): Dimensionality of the mask embedding.
        num_output_channels(Optional[int]): The dimensionality of the output. If ``None``,
            the value of ``width`` is used. Defaults to ``None``.
    """

    def __init__(
        self,
        number_of_filters: int = 256,
        number_of_conv_layers: int = 4,
        use_sigmoid: bool = False,
        width: int = 128,
        height: Optional[int] = None,
        mask_embed_dim: int = 6,
        num_output_channels: Optional[int] = None,
    ):
        print(f"[EvaluatorNetwork] -> n_layers = {number_of_conv_layers}")
        super(EvaluatorNetwork, self).__init__()

        self.spectral_map = SpectralMapDecomposition()
        self.mask_embed_dim = mask_embed_dim

        if height is None:
            height = width

        number_of_input_channels = 2 * width + mask_embed_dim

        norm_layer = functools.partial(
            nn.InstanceNorm2d, affine=False, track_running_stats=False
        )

        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        sequence = [
            nn.Conv2d(
                number_of_input_channels,
                number_of_filters,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.LeakyReLU(0.2, True),
        ]

        in_channels = number_of_filters

        for n in range(1, number_of_conv_layers):
            if n < number_of_conv_layers - 1:
                if n <= 4:
                    out_channels = in_channels * 2
                else:
                    out_channels = in_channels // 2

            else:
                out_channels = in_channels

            sequence += [
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    bias=use_bias,
                ),
                norm_layer(out_channels),
                nn.LeakyReLU(0.2, True),
            ]

            in_channels = out_channels
        kernel_size_width = width // 2 ** number_of_conv_layers
        kernel_size_height = height // 2 ** number_of_conv_layers
        sequence += [nn.AvgPool2d(kernel_size=(kernel_size_height, kernel_size_width))]

        if num_output_channels is None:
            num_output_channels = width
        sequence += [
            nn.Conv2d(
                in_channels, num_output_channels, kernel_size=1, stride=1, padding=0
            )
        ]

        if use_sigmoid:
            sequence += [nn.Sigmoid()]

        self.model = nn.Sequential(*sequence)
        self.apply(reconstruction.init_func)

    def forward(
        self,
        input_tensor: torch.Tensor,
        mask_embedding: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ):
        """Computes scores for each k-space column.

        Args:
            input_tensor(torch.Tensor): Batch of reconstructed images,
                    as produced by :class:`models.reconstruction.ReconstructorNetwork`.
            mask_embedding(Optional[torch.Tensor]): Corresponding batch of mask embeddings
                    produced by :class:`models.reconstruction.ReconstructorNetwork`, if needed.
            mask(Optional[torch.Tensor]): Corresponding masks arrays, if needed.

        Returns:
            torch.Tensor: Evaluator score for each k-space column in each image in the batch.
        """
        spectral_map_and_mask_embedding = self.spectral_map(
            input_tensor, mask_embedding, mask
        )
        return self.model(spectral_map_and_mask_embedding).squeeze(3).squeeze(2)
