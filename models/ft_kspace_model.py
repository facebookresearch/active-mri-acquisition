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
        self.visual_names = ['real_A_vis', 'fake_B_vis', 'real_B_vis']
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
        
    def to_net_input(self, data):
        # input [B, 2, H, W]
        # output [B,2*W, H, 1]
        b,c,h,w = data.shape
        # data = data.transpose(2,3).contiguous().view(b,2*w,h,1)
        data = data.contiguous().view(b,2*h,w,1)


        return data

    def to_image_output(self, data):
        # input [B,2*W, H, 1] 
        # output [B, 2, H, W]
        b,w,h,_ = data.shape
        w = w//2
        data = data.view(b,2,h,w)

        return data

    def set_input(self, input):
        # output from FT loader
        # BtoA is used to test if the network is identical
        AtoB = self.opt.which_direction == 'AtoB'
        img, _, _ = input
        img = img.to(self.device)
        # doing FFT
        # if has two dimension output, 
        # we actually want the imagary part is also supervised, which should be all zero
        fft_kspace = self.RFFT(img, normalized=True) #[B, 2, H, W]
       
        self.real_A = self.to_net_input(fft_kspace * self.mask)
        self.real_B = self.to_net_input(fft_kspace)

        # for visualization only
        self.real_A_vis = self.IFFT(fft_kspace * self.mask, normalized=True)
        self.real_B_vis = self.IFFT(fft_kspace, normalized=True)
        
    def compute_special_losses(self):
        # compute losses between fourier spaces of fake_B and real_B
        # if output one dimension
        # if self.fake_B.shape[1] == 1:
        #     _k_fakeB = self.RFFT(self.fake_B)
        #     _k_realB = self.RFFT(self.real_B)
        # else:
        #     # if output are two dimensional
        #     _k_fakeB = self.FFT(self.fake_B)
        #     _k_realB = self.FFT(self.real_B)

        _k_fakeB = self.fake_B
        _k_realB = self.real_B
        
        mask_deno = self.mask.sum() * _k_fakeB.shape[0] * _k_fakeB.shape[2] 
        invmask_deno = (1-self.mask).sum() * _k_fakeB.shape[0] * _k_fakeB.shape[2] 

        self.loss_FFTVisiable = F.mse_loss(_k_fakeB*self.mask, _k_realB*self.mask, reduce=False).sum().div(mask_deno)
        self.loss_FFTInvisiable = F.mse_loss(_k_fakeB*(1-self.mask), _k_realB*(1-self.mask), reduce=False).sum().div(invmask_deno)
        
        return float(self.loss_FFTVisiable), float(self.loss_FFTInvisiable)

    def test(self):
        with torch.no_grad():
            self.forward(test_mode=True)

    def forward(self, test_mode=False):
        # conditioned on mask
        # import pdb ; pdb.set_trace()
        # h, b = self.mask.shape[1], self.real_A.shape[0]
        # mask = Variable(self.mask.expand(b,h,1))
        self.fake_B, self.fake_B_res = self.netG(self.real_A)

        self.fake_B_vis = self.IFFT(self.to_image_output(self.fake_B), normalized=True)
        if test_mode:
            # to adapt the code of base_model for evaluation
            self.real_A = self.real_A_vis
            self.real_B = self.real_B_vis
            self.fake_B = self.fake_B_vis

    def backward_G(self):
        # First, G(A) should fake the discriminator
        self.loss_G_k = self.criterion(self.fake_B, self.real_B) 
        # for visualization only
        self.loss_G = self.criterion(self.IFFT(self.to_image_output(self.fake_B), normalized=True), 
                                self.IFFT(self.to_image_output(self.real_B), normalized=True))
            
        # l2 regularization 
        if self.opt.l2_weight:
            l2_loss = 0
            for param in self.netG.parameters():
                if len(param.shape) != 1: # no regualize bias term
                    l2_loss += param.norm(2)
            self.loss_G_k += l2_loss * 0.00001

        self.loss_G_k.backward()

        self.compute_special_losses()

    def optimize_parameters(self):
        self.forward()

        # update G
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()

