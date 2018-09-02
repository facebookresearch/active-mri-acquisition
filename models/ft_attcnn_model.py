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
import inspect

class FTATTCNNModel(BaseModel):
    def name(self):
        return 'FTATTCNNModel'

    @staticmethod
    def modify_commandline_options(parser, is_train=True):

        parser.set_defaults(no_dropout=True)

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
        # assert(opt.output_nc == 2)
        # load/define networks
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf,
                                    opt.which_model_netG, opt.norm, not opt.no_dropout, opt.init_type, self.gpu_ids, no_last_tanh=True)
        self.RFFT = RFFT().to(self.device)
        self.mask = create_mask(opt.fineSize, mask_fraction=self.opt.kspace_keep_ratio).to(self.device)
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
        self.zscore = 3

    def set_input(self, input):
        if self.mri_data:
            if len(input) == 4:
                input = input[1:]
            self.set_input2(input)
        else:
            self.set_input1(input)
            
    def _clamp(self, data):
        # process data for D input
        # make consistent range with real_B for inputs of D
        assert self.mri_data
        if self.mri_data:
            if self.zscore != 0:
                data = data.clamp(-self.zscore, self.zscore) 
        else:
            data = data.clamp(-1, 1) 
        return data

    def forward(self):
        # conditioned on mask
        if self.opt.output_nc == 1:
            self.fake_B = self.netG(self.real_A)
        else:
            h, b = self.mask.shape[2], self.real_A.shape[0]
            # mask = Variable(self.mask.view(self.mask.shape[0],h,1,1).expand(b,h,1,1))
            mask = self.mask
            self.fake_B = self.netG(self.real_A, mask)

    def backward_G(self):
        # First, G(A) should fake the discriminator
        self.loss_G = self.criterion(self.fake_B, self.real_B) 
        self.loss_G.backward()

        self.compute_special_losses()

    def optimize_parameters(self):
        self.forward()

        # update G
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()

    def set_input_exp(self, input, mask, zscore=3):
        assert self.mri_data
        # used for test kspace scanning line recommentation
        target, _, metadata = input
        target = target.to(self.device)
        self.metadata = self.metadata2onehot(metadata, dtype=type(target)).to(self.device)
        target = self._clamp(target).detach()

        self.mask = mask

        fft_kspace = self.RFFT(target)
        ifft_img = self.IFFT(fft_kspace * self.mask)

        if self.opt.output_nc >= 2:
            if self.imag_gt.shape[0] != target.shape[0]:
                # imagnary part is all zeros
                self.imag_gt = torch.zeros_like(target)
            target = torch.cat([target, self.imag_gt], dim=1)

        self.real_A = ifft_img
        self.real_B = target