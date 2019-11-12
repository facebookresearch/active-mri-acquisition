from .fft_utils import fft, ifft
from .reconstruction import init_func

import functools
import torch
import torch.nn as nn
from models.fft_utils import center_crop

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
        """

        Args:
            reconstructed_image: image reconstructed by ReconstructorNetwork
                                    shape   :   (batch_size, 2, height, width)
            mask_embedding: mask embedding created by ReconstructorNetwork (Replicated along
                height and width)
            shape   :   (batch_size, embedding_dimension, height, width)

        Returns:    spectral map concatenated with mask embedding
                        shape   : (batch_size, width + embedding_dimension, height, width)

        """
        batch_size = reconstructed_image.shape[0]
        height = reconstructed_image.shape[2]
        width = reconstructed_image.shape[3]

        # create spectral maps in kspace
        kspace = fft(reconstructed_image)
        kspace = kspace.unsqueeze(1).repeat(1, width, 1, 1, 1)

        # separate image into spectral maps
        separate_mask = torch.zeros([1, width, 1, 1, width], dtype=torch.float32)
        for i in range(width):
            separate_mask[0, i, 0, 0, i] = 1

        separate_mask = separate_mask.to(reconstructed_image.device)

        masked_kspace = torch.where(separate_mask.byte(), kspace,
                                    torch.tensor(0.).to(kspace.device))
        masked_kspace = masked_kspace.view(batch_size * width, 2, height, width)

        # convert spectral maps to image space
        separate_images = ifft(masked_kspace)
        # result is (batch, [real_M0, img_M0, real_M1, img_M1, ...],  height, width]
        separate_images = separate_images.contiguous().view(batch_size, 2, width, height, width)

        # add mask information as a summation -- might not be optimal
        if mask is not None:
            separate_images = separate_images + mask.permute(0, 3, 1, 2).unsqueeze(1).detach()

        separate_images = separate_images.contiguous().view(batch_size, 2 * width, height, width)
        # concatenate mask embedding
        if mask_embedding is not None:
            spectral_map = torch.cat([separate_images, mask_embedding], dim=1)
        else:
            spectral_map = separate_images

        return spectral_map


class EvaluatorNetwork(nn.Module):

    def __init__(self,
                 number_of_filters=256,
                 number_of_conv_layers=4,
                 use_sigmoid=False,
                 width=128,
                 mask_embed_dim=6,
                 num_output_channels=None):
        print(f'[EvaluatorNetwork] -> n_layers = {number_of_conv_layers}')
        super(EvaluatorNetwork, self).__init__()

        self.spectral_map = SpectralMapDecomposition()
        self.mask_embed_dim = mask_embed_dim

        number_of_input_channels = 2 * width + mask_embed_dim

        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)

        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        sequence = [
            nn.Conv2d(
                number_of_input_channels, number_of_filters, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, True)
        ]

        in_channels = number_of_filters

        for n in range(1, number_of_conv_layers):
            if n < number_of_conv_layers - 1:
                out_channels = in_channels * 2
            else:
                out_channels = in_channels

            sequence += [
                nn.Conv2d(
                    in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=use_bias),
                norm_layer(out_channels),
                nn.LeakyReLU(0.2, True)
            ]

            in_channels = out_channels

        kernel_size = width // 2**number_of_conv_layers
        sequence += [nn.AvgPool2d(kernel_size=kernel_size)]

        if num_output_channels is None:
            num_output_channels = width
        sequence += [
            nn.Conv2d(in_channels, num_output_channels, kernel_size=1, stride=1, padding=0)
        ]

        if use_sigmoid:
            sequence += [nn.Sigmoid()]

        self.model = nn.Sequential(*sequence)
        self.apply(init_func)

    def forward(self, input, mask_embedding=None, mask=None):
        """

        Args:
            input: reconstructed image as returned by the reconstruction network
                        shape   :   (batch_size, 2, height, width)
            mask_embedding:     mask embedding returned by the reconstructor
                        shape   :   (batch_size, embedding_dimension, height, width)

        Returns: evaluator score for each measurement
                    shape   :   (batch_size, width)

        """
        spectral_map_and_mask_embedding = self.spectral_map(input, mask_embedding, mask)
        return self.model(spectral_map_and_mask_embedding).squeeze(3).squeeze(2)


def test_evaluator(height, width, number_of_filters, number_of_conv_layers, use_sigmoid,
                   mask_embed_dim):
    batch = 4

    image = torch.rand(batch, 2, height, width)
    image = image.type(torch.FloatTensor)

    if mask_embed_dim > 0:
        mask_embedding = torch.rand(batch, mask_embed_dim, height, width)
        mask_embedding.type(torch.FloatTensor)
    else:
        mask_embedding = None

    evaluator = EvaluatorNetwork(
        number_of_filters=number_of_filters,
        number_of_conv_layers=number_of_conv_layers,
        use_sigmoid=use_sigmoid,
        width=width,
        mask_embed_dim=mask_embed_dim)
    output = evaluator(image, mask_embedding)
    print('evaluator output shape :', output.shape)


if __name__ == '__main__':

    print('DICOM :')
    test_evaluator(
        height=128,
        width=128,
        number_of_filters=256,
        number_of_conv_layers=4,
        use_sigmoid=False,
        mask_embed_dim=6)

    print('RAW:')
    test_evaluator(
        height=640,
        width=368,
        number_of_filters=64,
        number_of_conv_layers=5,
        use_sigmoid=False,
        mask_embed_dim=0)
