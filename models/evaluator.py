import torch
import torch.nn as nn
import torch.nn.functional as F
from fft_utils import RFFT, IFFT, FFT
from torch.nn import init
from reconstruction import get_norm_layer, init_net



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

    netD = NLayerDiscriminatorChannel(input_nc, ndf, n_layers_D, norm_layer=norm_layer, use_sigmoid=use_sigmoid)

    if preprocess_module is not None:
        netD = SimpleSequential(preprocess_module, netD)

    return init_net(netD, init_type, gpu_ids)


class NLayerDiscriminatorChannel(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3,
            norm_layer=nn.BatchNorm2d, use_sigmoid=False, imSize=128):
        print(f'[NLayerDiscriminatorChannel] -> n_layers = {n_layers}, n_channel {input_nc}')
        super(NLayerDiscriminatorChannel, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True)
        ]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 4)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                          kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        kw = imSize//2**n_layers
        sequence += [nn.AvgPool2d(kernel_size=kw)]
        sequence += [nn.Conv2d(ndf * nf_mult, imSize, kernel_size=1, stride=1, padding=0)]

        if use_sigmoid:
            sequence += [nn.Sigmoid()]

        self.model = nn.Sequential(*sequence)

    def forward(self, input, mask):
        # mask is not used
        return self.model(input).squeeze()

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
