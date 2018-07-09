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

class FTGCNModel(BaseModel):
    def name(self):
        return 'FTGCNModel'

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
        assert(opt.output_nc == 2)
        # load/define networks
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf,
                                    opt.which_model_netG, opt.norm, not opt.no_dropout, opt.init_type, self.gpu_ids, no_last_tanh=True)
        self.RFFT = RFFT().to(self.device)
        self.mask = create_mask(opt.fineSize).to(self.device)
        self.mask = self.mask.view(1, self.mask.shape[2], 1)
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

        if self.opt.output_nc == 2:
            # the imagnary part of reconstrued data
            self.imag_gt = torch.cuda.FloatTensor(opt.batchSize, 1, opt.fineSize, opt.fineSize)
        
        self._adj = Variable(torch.ones(1, opt.fineSize, opt.fineSize).float())
        self._adj.div_(opt.fineSize)

    def to_graph_space(self, data):
        # data [B, 2, H, W]
        b,c,h,w = data.shape
        data = data.permute(0,2,3,1).contiguous().view(b,h,w*c)

        return data

    def to_image_space(self, data):
        # data [B, H, W*2]
        b,h,w = data.shape
        w = w//2
        data = data.view(b,h,w,2).permute(0,3,1,2)

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
        if self.opt.output_nc == 2:
            if self.imag_gt.shape[0] != img.shape[0]:
                # imagnary part is all zeros
                self.imag_gt = torch.zeros_like(img)
            img = torch.cat([img, self.imag_gt], dim=1)

        self.real_A = self.to_graph_space(fft_kspace * self.mask)
        self.real_B = self.to_graph_space(fft_kspace)

        # for visualization only
        self.real_A_vis = self.IFFT(fft_kspace * self.mask, normalized=True)
        self.real_B_vis = img

        self.adj = self._adj.expand(self.real_A.shape[0], self._adj.shape[1], self._adj.shape[2])
        
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
        mask_deno = self.mask.sum() * self.fake_B.shape[0] * self.fake_B.shape[2] 
        invmask_deno = (1-self.mask).sum() * self.fake_B.shape[0] * self.fake_B.shape[2] 

        self.loss_FFTVisiable = F.mse_loss(_k_fakeB * self.mask, _k_realB*self.mask, reduce=False).sum().div(mask_deno)
        self.loss_FFTInvisiable = F.mse_loss(_k_fakeB * (1-self.mask), _k_realB*(1-self.mask), reduce=False).sum().div(invmask_deno)
        
        return float(self.loss_FFTVisiable), float(self.loss_FFTInvisiable)

    def test(self):
        with torch.no_grad():
            self.forward(test_mode=True)

    def forward(self, test_mode=False):
        # conditioned on mask
        h, b = self.mask.shape[1], self.real_A.shape[0]
        mask = Variable(self.mask.expand(b,h,1))
        self.fake_B, self.fake_B_res = self.netG(self.real_A, self.adj, mask)

        self.fake_B_vis = self.IFFT(self.to_image_space(self.fake_B), normalized=True)

        if test_mode:
            self.real_A = self.real_A_vis
            self.real_B = self.real_B_vis
            self.fake_B = self.fake_B_vis

    def backward_G(self):
        # First, G(A) should fake the discriminator
        self.loss_G_k = self.criterion(self.fake_B, self.real_B) 
        # for visualization only
        self.loss_G = self.criterion(self.IFFT(self.to_image_space(self.fake_B), normalized=True), 
                                self.IFFT(self.to_image_space(self.real_B), normalized=True))
            
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

