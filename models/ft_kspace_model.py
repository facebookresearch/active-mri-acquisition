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
from .fft_utils import *
from torch.autograd import Variable

class FTKSPACEModel(BaseModel):
    def name(self):
        return 'FTKSPACEModel'

    @staticmethod
    def modify_commandline_options(parser, is_train=True):

        return parser

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.isTrain = opt.isTrain
        # specify the training losses you want to print out. The program will call base_model.get_current_losses
        self.loss_names = ['G', 'FFTVisiable', 'FFTInvisiable']
        # specify the images you want to save/display. The program will call base_model.get_current_visuals
        self.visual_names = ['real_A', 'fake_B', 'real_B']
        # specify the models you want to save to the disk. The program will call base_model.save_networks and base_model.load_networks
        if self.isTrain:
            self.model_names = ['G']
        else:  # during test time, only load Gs
            self.model_names = ['G']
        # load/define networks
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf,
                                    opt.which_model_netG, opt.norm, not opt.no_dropout, opt.init_type, self.gpu_ids, no_last_tanh=True)
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

    def to_net_input(self, input):
        # to go log space
        input = input.div(self.factor) # normalize kspace
        out = input.add_(self.kspace_lowv).log_()
        return out

    def to_net_output(self, input):
        # copy data
        out = input.exp().add_(-self.kspace_lowv)
        out.mul_(self.factor)

        return out

    def set_input(self, input):
        # output from FT loader
        # BtoA is used to test if the network is identical
        AtoB = self.opt.which_direction == 'AtoB'
        img, _, _ = input
        img = img.to(self.device)
        # doing FFT
        # if has two dimension output, 
        # we actually want the imagary part is also supervised, which should be all zero
        fft_kspace = self.RFFT(img) #[B, 2, H, W]

        fft_kspace = self.to_net_input(fft_kspace)

        self.real_A_k = fft_kspace * self.mask
        self.real_B_k = fft_kspace

        # for visualization only
        rn_real_A_k = self.to_net_output(self.real_A_k) * self.mask
        rn_real_B_k = self.to_net_output(self.real_B_k)

        self.real_A = self.IFFT(rn_real_A_k)
        self.real_B = self.IFFT(rn_real_B_k)
                
    def compute_special_losses(self):
    
        if self.fake_B.shape[1] == 1:
            _k_fakeB = self.RFFT(self.fake_B)
            _k_realB = self.RFFT(self.real_B)
        else:
            # if output are two dimensional
            _k_fakeB = self.FFT(self.fake_B)
            _k_realB = self.FFT(self.real_B)
        
        mask_deno = self.mask.sum() * _k_fakeB.shape[0] * _k_fakeB.shape[2] 
        invmask_deno = (1-self.mask).sum() * _k_fakeB.shape[0] * _k_fakeB.shape[2] 

        self.loss_FFTVisiable = F.mse_loss(_k_fakeB*self.mask, _k_realB*self.mask, reduce=False).sum().div(mask_deno)
        self.loss_FFTInvisiable = F.mse_loss(_k_fakeB*(1-self.mask), _k_realB*(1-self.mask), reduce=False).sum().div(invmask_deno)
        
        return float(self.loss_FFTVisiable), float(self.loss_FFTInvisiable)

    def forward(self):
        # conditioned on mask
        # h, b = self.mask.shape[1], self.real_A.shape[0]
        # mask = Variable(self.mask.expand(b,h,1))
        self.fake_B_k = self.netG(self.real_A_k)
        
        # residual training
        self.fake_B_k = self.fake_B_k * (1-self.mask) + self.real_A_k * self.mask

        real_B_k = self.real_B_k * (1-self.mask) + self.real_A_k * self.mask
        
        real_B = self.IFFT(self.to_net_output(real_B_k))
        
        rn_fake_B_k = self.to_net_output(self.fake_B_k)

        self.fake_B = self.IFFT(rn_fake_B_k)
        
    def backward_G(self):
        # MSE on kspace
        self.loss_G = self.criterion(self.fake_B, self.real_B) 

        loss_G_k = self.criterion(self.fake_B_k, self.real_B_k) 
        loss_G_k.backward()

        self.compute_special_losses()

    def optimize_parameters(self):
        self.forward()

        # update G
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()

