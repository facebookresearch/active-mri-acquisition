from fft_utils import RFFT, IFFT, FFT
from reconstruction import get_norm_layer, init_net, init_func

import functools
import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleSequential(nn.Module):
    def __init__(self, net1, net2):
        super(SimpleSequential, self).__init__()
        self.net1 = net1
        self.net2 = net2

    def forward(self, x, mask):
        output = self.net1(x,mask)
        return self.net2(output,mask)


def define_D(input_nc, ndf, which_model_netD,
             n_layers_D=3, norm='batch', use_sigmoid=False, init_type='normal', gpu_ids=[], preprocess_module=None):
    netD = None
    norm_layer = get_norm_layer(norm_type=norm)

    netD = EvaluatorNetwork(input_nc, ndf, n_layers_D, norm_layer=norm_layer, use_sigmoid=use_sigmoid)

    if preprocess_module is not None:
        netD = SimpleSequential(preprocess_module, netD)

    return init_net(netD, init_type, gpu_ids)


class SpectralMapDecomposition(nn.Module):
    def __init__(self):
        super(SpectralMapDecomposition, self).__init__()

        self.RFFT = RFFT()
        self.IFFT = IFFT()

    def forward(self, reconstructed_image, mask_embedding):
        """

        Args:
            reconstructed_image: image reconstructed by ReconstructorNetwork
                                    shape   :   (batch_size, 1, height, width)
            mask_embedding: mask embedding created by ReconstructorNetwork (Replicated along height and width)
                                    shape   :   (batch_size, embedding_dimension, height, width)

        Returns:    spectral map concatenated with mask embedding
                        shape   : (batch_size, width + embedding_dimension, height, width)

        """
        batch_size = reconstructed_image.shape[0]
        height = reconstructed_image.shape[2]
        width = reconstructed_image.shape[3]

        # create spectral maps in kspace
        kspace = self.RFFT(reconstructed_image)
        kspace = kspace.unsqueeze(1).repeat(1, width, 1, 1, 1)

        # seperate image into spectral maps
        separate_mask = torch.zeros([1, width, 1, 1, width], dtype=torch.float32)
        for i in range(width):
            separate_mask[0, i, 0, 0, i] = 1

        masked_kspace = separate_mask * kspace
        masked_kspace = masked_kspace.view(batch_size * width, 2, height, width)

        # convert spectral maps to image space
        # discard the imaginary part
        separate_images = self.IFFT(masked_kspace)[:, 0, :, :].view(batch_size, width, height, width)

        # concatenate mask embedding
        if mask_embedding is not None:
            spectral_map = torch.cat([separate_images, mask_embedding], dim=1)
        else:
            spectral_map = separate_images

        return spectral_map


class EvaluatorNetwork(nn.Module):
    def __init__(self, number_of_filters=256,
                 number_of_conv_layers=4,
                 use_sigmoid=False,
                 width=128,
                 mask_embed_dim=6):
        print(f'[NLayerDiscriminatorChannel] -> n_layers = {number_of_conv_layers}')
        super(EvaluatorNetwork, self).__init__()

        self.spectral_map = SpectralMapDecomposition()
        self.mask_embed_dim = mask_embed_dim

        number_of_input_channels = width + mask_embed_dim

        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)

        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        sequence = [
            nn.Conv2d(number_of_input_channels, number_of_filters, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, True)
        ]

        in_channels = number_of_filters

        for n in range(1, number_of_conv_layers):
            if n < number_of_conv_layers - 1:
                out_channels = in_channels * 2
            else:
                out_channels = in_channels

            sequence += [
                nn.Conv2d(in_channels, out_channels,
                          kernel_size=4, stride=2, padding=1, bias=use_bias),
                norm_layer(out_channels),
                nn.LeakyReLU(0.2, True)
            ]

            in_channels = out_channels

        kernel_size = width // 2 ** number_of_conv_layers
        sequence += [nn.AvgPool2d(kernel_size=kernel_size)]
        sequence += [nn.Conv2d(in_channels, width, kernel_size=1, stride=1, padding=0)]

        if use_sigmoid:
            sequence += [nn.Sigmoid()]

        self.model = nn.Sequential(*sequence)
        self.apply(init_func)

    def forward(self, input, mask_embedding):
        """

        Args:
            input: reconstructed image as returned by the reconstruction network
                        shape   :   (batch_size, 2, height, width)
            mask_embedding:     mask embedding returned by the reconstructor
                        shape   :   (batch_size, embedding_dimension, height, width)

        Returns: evaluator score for each measurement
                    shape   :   (batch_size, width)

        """

        spectral_map_and_mask_embedding = self.spectral_map(input, mask_embedding)

        return self.model(spectral_map_and_mask_embedding).squeeze()

#TODO: we might consider moving this to losses
class GANLossKspace(nn.Module):
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0,
                 use_mse_as_energy=False, grad_ctx=False, gamma=100):
        super(GANLossKspace, self).__init__()
        # self.register_buffer('real_label', torch.ones(imSize, imSize))
        # self.register_buffer('fake_label', torch.zeros(imSize, imSize))
        self.grad_ctx = grad_ctx
        if use_lsgan:
            self.loss = nn.MSELoss(size_average=False)
        else:
            self.loss = nn.BCELoss(size_average=False)
        self.use_mse_as_energy = use_mse_as_energy
        if use_mse_as_energy:
            self.RFFT = RFFT()
            self.gamma = gamma
            self.bin = 5

    def get_target_tensor(self, input, target_is_real, degree, mask, pred_gt=None):

        if target_is_real:
            target_tensor = torch.ones_like(input)
            target_tensor[:] = degree

        else:
            target_tensor = torch.zeros_like(input)
            if not self.use_mse_as_energy:
                if degree != 1:
                    target_tensor[:] = degree
            else:
                pred, gt = pred_gt
                w = gt.shape[2]
                ks_gt = self.RFFT(gt[:,:1,:,:], normalized=True)
                ks_input = self.RFFT(pred, normalized=True)
                ks_row_mse = F.mse_loss(
                    ks_input, ks_gt, reduce=False).sum(1, keepdim=True).sum(2, keepdim=True).squeeze() / (2*w)
                energy = torch.exp(-ks_row_mse * self.gamma)

                # do some bin process
                # import pdb; pdb.set_trace()
                # energy = torch.floor(energy * 10 / self.bin) * self.bin / 10

                target_tensor[:] = energy
            # force observed part to always
            for i in range(mask.shape[0]):
                idx = torch.nonzero(mask[i, 0, 0, :])
                target_tensor[i,idx] = 1
        return target_tensor

    def __call__(self, input, target_is_real, mask, degree=1, updateG=False, pred_gt=None):
        # input [B, imSize]
        # degree is the realistic degree of output
        # set updateG to True when training G.
        target_tensor = self.get_target_tensor(input, target_is_real, degree, mask, pred_gt)
        b,w = target_tensor.shape
        if updateG and not self.grad_ctx:
            mask_ = mask.squeeze()
            # maskout the observed part loss
            return self.loss(input * (1-mask_), target_tensor * (1-mask_)) / (1-mask_).sum()
        else:
            return self.loss(input, target_tensor) / (b*w)

def test_evaluator(height, width, number_of_filters, number_of_conv_layers, use_sigmoid, mask_embed_dim):
    batch = 4

    image = torch.rand(batch, 1, height, width)
    image = image.type(torch.FloatTensor)

    if mask_embed_dim > 0:
        mask_embedding = torch.rand(batch, mask_embed_dim, height, width)
        mask_embedding.type(torch.FloatTensor)
    else:
        mask_embedding = None

    evaluator = EvaluatorNetwork(number_of_filters=number_of_filters,
                                 number_of_conv_layers=number_of_conv_layers,
                                 use_sigmoid=use_sigmoid,
                                 width=width,
                                 mask_embed_dim=mask_embed_dim)
    output = evaluator(image, mask_embedding)
    print('evaluator output shape :', output.shape)


if __name__ == '__main__':

    print('DICOM :')
    test_evaluator(height=128,
                   width=128,
                   number_of_filters=256,
                   number_of_conv_layers=4,
                   use_sigmoid=False,
                   mask_embed_dim=6)

    print('RAW:')
    test_evaluator(height=640,
                   width=368,
                   number_of_filters=64,
                   number_of_conv_layers=5,
                   use_sigmoid=False,
                   mask_embed_dim=0)

