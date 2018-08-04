import torch
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
from util import util
import torchvision.utils as tvutil
import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
from .fft_utils import create_mask, roll_n
from torch.autograd import Variable
import functools


def fftshift(batch):
    results = []
    for i in range(batch.shape[1]):
        data = batch[:,i,:,:]
        for dim in range(1, len(data.size())):
            data = roll_n(data, axis=dim, n=data.size(dim)//2)
        results.append(data)
    batch = torch.stack(results, 1)
    return batch

def ifftshift(batch):
    results = []
    for i in range(batch.shape[1]):
        data = batch[:,i,:,:]
        for dim in range(len(data.size()) - 1, 0, -1):
            data = roll_n(data, axis=dim, n=data.size(dim)//2)
        results.append(data)
    batch = torch.stack(results, 1)
    return batch   

class IFFT(nn.Module):
    def forward(self, x, normalized=False, shift=False):
        x = x.permute(0, 2, 3, 1)
        y = torch.ifft(x, 2, normalized=normalized)
        if shift:
            y = ifftshift(y)

        return y.permute(0, 3, 1, 2)

    def __repr__(self):
        return 'IFFT()'

class RFFT(nn.Module):
    def forward(self, x, normalized=False, shift=False):
        # x is in gray scale and has 1-d in the 1st dimension
        x = x.squeeze(1)
        y = torch.rfft(x, 2, onesided=False, normalized=normalized)
        if shift:
            y = fftshift(y)
        return y.permute(0, 3, 1, 2)

    def __repr__(self):
        return 'RFFT()'

class FFT(nn.Module):
    def forward(self, x, normalized=False, shift=False):
        x = x.permute(0, 2, 3, 1)
        y = torch.fft(x, 2, normalized=normalized)
        if shift:
            y = fftshift(y)

        return y.permute(0, 3, 1, 2)

    def __repr__(self):
        return 'FFT()'

from .coordconv import CoordConv
class kspace_unet(nn.Module):
    def __init__(self, fm_in, out_dim, fm=64, use_coordconv=False):
        super(kspace_unet, self).__init__()
        conv = lambda fm_in, fm_out, stride=2: nn.Conv2d(fm_in, fm_out, 4, stride, 1)
        convT = lambda fm_in, fm_out: nn.ConvTranspose2d(fm_in, fm_out, 4, 2, 1)

        if use_coordconv:
            print('[kspace_unet] -> use CoordConv')
            self.in_conv = nn.Sequential(*[CoordConv(fm_in, fm, kernel_size=1, stride=1, padding=0), 
                                nn.LeakyReLU(0.2, True), 
                                conv(fm, fm)])
        else:
            self.in_conv = nn.Sequential(*[conv(fm_in, fm)])
        
        self.conv1 = conv(fm*1, fm*2)
        self.conv2 = conv(fm*2, fm*4)
        self.conv3 = conv(fm*4, fm*4)
        self.deconv1 = convT(fm*4*1, fm*4)
        self.deconv2 = convT(fm*4*2, fm*2)
        self.deconv3 = convT(fm*2*2, fm*1)

        self.out_conv = convT(fm*2*1, out_dim)
        
        self.leaky_relu = functools.partial(F.leaky_relu, negative_slope=0.2, inplace=True)

        self.relu = functools.partial(F.relu, inplace=True)

    def forward(self, x):
        
        x = fftshift(x)
        d1 = self.leaky_relu(self.in_conv(x))
        d2 = self.leaky_relu(self.conv1(d1))
        d3 = self.leaky_relu(self.conv2(d2))
        d4 = self.relu(self.conv3(d3))

        d5 = self.deconv1(d4)
        d5 = self.relu(torch.cat([d5, d3], 1))
        d6 = self.deconv2(d5)
        d6 = self.relu(torch.cat([d6, d2], 1))
        d7 = self.deconv3(d6)
        d7 = self.relu(torch.cat([d7, d1], 1))

        out = self.out_conv(d7)
        out = ifftshift(out)

        return out

from math import sqrt
class Conv_ReLU_Block(nn.Module):
    def __init__(self, dilation):
        super(Conv_ReLU_Block, self).__init__()
        self.conv = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, dilation=dilation, stride=1, padding=dilation, bias=False)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        return self.relu(self.conv(x))
        
class Net(nn.Module):
    r"got from this implementation https://github.com/twtygqyy/pytorch-vdsr"
    def __init__(self, input_nc, output_nc, use_coordconv=False, num_reblocks=18, use_dilation=False):
        super(Net, self).__init__()
        if use_dilation:
            dilation_sizes = [1,1,2,4,8,16,2,4,8,1,1]
            num_reblocks = len(dilation_sizes)
            print('--> use DilatedConv ', dilation_sizes)
        else:
            dilation_sizes = None
        self.residual_layer = self.make_layer(Conv_ReLU_Block, num_reblocks, dilation_sizes)
        if use_coordconv:
            print('--> use CoordConv')
            self.input = nn.Sequential(*[CoordConv(input_nc, 64, kernel_size=1, stride=1, padding=0), 
                                nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)])
        else:
            self.input = nn.Conv2d(in_channels=input_nc, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        
        self.output = nn.Conv2d(in_channels=64, out_channels=output_nc, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
    
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, sqrt(2. / n))
                
    def make_layer(self, block, num_of_layer, dilation_sizes=None):
        layers = []
        for i in range(num_of_layer):
            dil = 1 if dilation_sizes is None else dilation_sizes[i]
            layers.append(block(dilation=dil))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = fftshift(x)
        out = self.relu(self.input(x))
        out = self.residual_layer(out)
        out = self.output(out)
        out = ifftshift(out)

        return out

class FTKSPACEModel(BaseModel):
    def name(self):
        return 'FTKSPACEModel'

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.add_argument('--use_coordconv', action='store_true', help='use CoordConv.')
        parser.add_argument('--use_stage1_skip', action='store_true', help='use stage1 output as skip connection of stage 2.')
        parser.add_argument('--use_dilation', action='store_true', help='use CoordConv.')
        return parser

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.isTrain = opt.isTrain
        # specify the training losses you want to print out. The program will call base_model.get_current_losses
        self.loss_names = ['G', 'FFTVisiable', 'FFTInvisiable', 'G_k', 'G_S1']
        # specify the images you want to save/display. The program will call base_model.get_current_visuals
        self.visual_names = ['real_A', 'fake_B', 'real_B']
        # specify the models you want to save to the disk. The program will call base_model.save_networks and base_model.load_networks
        if self.isTrain:
            self.model_names = ['G']
        else:  # during test time, only load Gs
            self.model_names = ['G']
        # load/define networks
        self.netG_basis = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf,
                                    opt.which_model_netG, opt.norm, not opt.no_dropout, opt.init_type, self.gpu_ids, no_last_tanh=True)

        if self.isTrain and not opt.continue_train:
            self.model_names = ['G_basis']
            self.load_networks('0')
            self.model_names = ['G']

        # conditioned on mask
        from .networks import init_net
        self.netG = init_net(Net(opt.input_nc+1, opt.output_nc, use_coordconv=opt.use_coordconv, num_reblocks=9, use_dilation=opt.use_dilation), opt.init_type, self.gpu_ids)
        self.RFFT = RFFT().to(self.device)
        self.mask = create_mask(opt.fineSize).to(self.device)
        self.IFFT = IFFT().to(self.device)
        # for evaluation
        self.FFT = FFT().to(self.device)
        
        if self.isTrain:
            # define loss functions
            self.criterion = torch.nn.MSELoss()

            # initialize optimizers
            self.optimizers = []
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))

            self.optimizers.append(self.optimizer_G)
        
        self.factor = self.opt.fineSize ** 2
        self.kspace_lowv = 3 if self.mri_data else 2 
        
        self.imag_gt = torch.cuda.FloatTensor(opt.batchSize, 1, opt.fineSize, opt.fineSize)

    def to_net_input(self, input):
        # to go log space
        out = input.div(self.factor) # normalize kspace
        # out = out.add(self.kspace_lowv)
        # if out.min() <=0:
        #     ValueError('the minimum value of shifted kspace is >= 0, which is '+str(out.min().item()))
        # out = out.log()
        return out

    def to_net_output(self, input):
        # copy data
        # out = input.exp()
        # out = out.add(-self.kspace_lowv)
        # out = out.mul(self.factor)
        out = input.mul(self.factor)
        return out
 
    def set_input(self, input):
        if self.mri_data:
            if len(input) == 4:
                input = input[1:]
            self.set_input2(input)
            self.real_B_k = self.FFT(self.real_B)
        else:
            self.set_input1(input)

    def forward(self):
        # conditioned on mask
        
        h, b = self.real_A.shape[2], self.real_A.shape[0]
        mask = Variable(self.mask)
        with torch.no_grad():
            fake_B_Gs, _ = self.netG_basis(self.real_A, mask)
            self.fake_B_G = fake_B_Gs[-1]

        fake_B_G = self.to_net_input(self.RFFT(self.fake_B_G[:,:1,:,:]))
        
        mask = self.mask.repeat(1,1,1,h).detach()
        
        self.fake_B_k = self.netG(torch.cat([fake_B_G, mask], 1))
        # residual training
        
        if self.opt.use_stage1_skip:
            self.fake_B_k = self.fake_B_k + self.to_net_input(self.FFT(self.fake_B_G)) # use skip connection

        self.fake_B_k = self.to_net_output(self.fake_B_k * (1-self.mask)) + self.FFT(self.real_A) * self.mask

        # self.fake_B_k = self.to_net_output(self.to_net_input(self.real_B_k) * (1-self.mask) + self.real_A_k * self.mask)
        # go back to image space
        self.fake_B = self.IFFT(self.fake_B_k)
        
    def backward_G(self):
        # MSE on kspace
        self.loss_G_S1 = self.criterion(self.fake_B_G, self.real_B) 
        self.loss_G = self.criterion(self.fake_B, self.real_B) 

        self.loss_G_k = self.criterion(self.fake_B_k, self.real_B_k) + self.loss_G
        self.loss_G_k.backward()

        self.compute_special_losses()

    def optimize_parameters(self):
        self.forward()

        # update G
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()

