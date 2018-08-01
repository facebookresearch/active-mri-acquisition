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

class FTIMKSPACEModel(BaseModel):
    def name(self):
        return 'FTIMKSPACEModel'

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.add_argument('--preload_G', action='store_true', help='use posterior and use input of det net.')

        return parser

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.isTrain = opt.isTrain
        # specify the training losses you want to print out. The program will call base_model.get_current_losses
        self.loss_names = ['G', 'FFTVisiable', 'FFTInvisiable', 'G_s1']
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
        
        # if self.isTrain and not opt.continue_train:
        #     assert(opt.preload_G or opt.train_G)
        #     if opt.preload_G:
        #         self.model_names = ['G']
        #         self.load_networks('0')
        #         self.model_names = ['G', 'CVAE']

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
        if self.opt.output_nc == 2:
            # the imagnary part of reconstrued data
            self.imag_gt = torch.cuda.FloatTensor(opt.batchSize, 1, opt.fineSize, opt.fineSize)

    def set_input(self, input):
        # output from FT loader
        img, _, _ = input
        img = img.to(self.device)
        # doing FFT
        # if has two dimension output, 
        # we actually want the imagary part is also supervised, which should be all zero
        fft_kspace = self.RFFT(img)
        if self.opt.output_nc == 2:
            if self.imag_gt.shape[0] != img.shape[0]:
                # imagnary part is all zeros
                self.imag_gt = torch.zeros_like(img)
            img = torch.cat([img, self.imag_gt], dim=1)

        self.real_A = self.IFFT(fft_kspace * self.mask)
        self.real_B = img
    
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
        h, b = self.mask.shape[2], self.real_A.shape[0]
        mask = Variable(self.mask.expand(b,1,h,1))
        self.fake_B, self.fake_B_stage1 = self.netG(self.real_A, mask)
        
    def backward_G(self):
        # MSE on kspace
        self.loss_G = self.criterion(self.fake_B, self.real_B) 

        self.loss_G_s1 = self.criterion(self.fake_B_stage1, self.real_B) 

        self.loss_G.backward()

        self.compute_special_losses()

    def optimize_parameters(self):
        self.forward()

        # update G
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()

