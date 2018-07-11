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


        parser.set_defaults(pool_size=0)
        parser.set_defaults(no_lsgan=True)
        parser.set_defaults(norm='batch')
        parser.set_defaults(dataset_mode='aligned')
        parser.set_defaults(which_model_netG='unet_256')
        if is_train:
            parser.add_argument('--lambda_L1', type=float, default=100.0, help='weight for L1 loss')

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
        assert(opt.output_nc == 2)
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

        if self.opt.output_nc == 2:
            # the imagnary part of reconstrued data
            self.imag_gt = torch.cuda.FloatTensor(opt.batchSize, 1, opt.fineSize, opt.fineSize)

    def set_input(self, input):
        # output from FT loader
        # BtoA is used to test if the network is identical
        AtoB = self.opt.which_direction == 'AtoB'
        img, _, _ = input
        img = img.to(self.device)

        caller = inspect.stack()[1][3]

        if self.opt.dynamic_mask_type in ['random','random_plus','curriculum']:
            # TODO: implement curriculum, random_plus
            self.mask = create_mask((img.shape[0], img.shape[2]), mask_fraction=self.opt.kspace_keep_ratio, mask_low_freqs=5).to(self.device)

        # doing FFT
        # if has two dimension output, 
        # we actually want the imagary part is also supervised, which should be all zero
        fft_kspace = self.RFFT(img)
        if self.opt.output_nc == 2:
            if self.imag_gt.shape[0] != img.shape[0]:
                # imagnary part is all zeros
                self.imag_gt = torch.zeros_like(img)
            img = torch.cat([img, self.imag_gt], dim=1)

        if AtoB:
            self.real_A = self.IFFT(fft_kspace * self.mask)
            self.real_B = img
        else:
            self.real_A = self.IFFT(fft_kspace)
            self.real_B = img
    
    def compute_special_losses(self):
        # compute losses between fourier spaces of fake_B and real_B
        # if output one dimension
        if self.fake_B.shape[1] == 1:
            _k_fakeB = self.RFFT(self.fake_B)
            _k_realB = self.RFFT(self.real_B)
        else:
            # if output are two dimensional
            _k_fakeB = self.FFT(self.fake_B)
            _k_realB = self.FFT(self.real_B)

        mask_deno = self.mask.sum() * self.fake_B.shape[0] * self.fake_B.shape[1] * self.fake_B.shape[3]
        invmask_deno = (1-self.mask).sum() * self.fake_B.shape[0] * self.fake_B.shape[1] * self.fake_B.shape[3]

        self.loss_FFTVisiable = F.mse_loss(_k_fakeB * self.mask, _k_realB*self.mask, reduce=False).sum().div(mask_deno)
        self.loss_FFTInvisiable = F.mse_loss(_k_fakeB * (1-self.mask), _k_realB*(1-self.mask), reduce=False).sum().div(invmask_deno)
        
        return float(self.loss_FFTVisiable), float(self.loss_FFTInvisiable)

    def forward(self):
        # conditioned on mask
        h, b = self.mask.shape[2], self.real_A.shape[0]
        mask = Variable(self.mask.view(self.mask.shape[0],h,1,1).expand(b,h,1,1))
        self.fake_B, self.fake_B_res = self.netG(self.real_A, mask)

    def backward_G(self):
        # First, G(A) should fake the discriminator
        self.loss_G = self.criterion(self.fake_B, self.real_B) 
        if self.opt.consistency_loss:
            self.loss_G = self.loss_G*0.9 + self.criterion(self.fake_B2, self.real_B) * 0.1 
        
        # residual loss
        # observed part should be all zero during residual training (y(x)+x)
        if self.opt.residual_loss:
            _k_fake_B_res = self.FFT(self.fake_B_res)
            if not hasattr(self, '_residual_gt') or (self._residual_gt.shape[0] != _k_fake_B_res.shape[0]):
                self._residual_gt = torch.zeros_like(_k_fake_B_res)
            loss_residual = self.criterion(_k_fake_B_res * self.mask, self._residual_gt)
            self.loss_G += loss_residual * 0.01 # around 100 smaller
            
        # l2 regularization 
        if self.opt.l2_weight:
            l2_loss = 0
            for param in self.netG.parameters():
                if len(param.shape) != 1: # no regualize bias term
                    l2_loss += param.norm(2)
            self.loss_G += l2_loss * 0.00001

        self.loss_G.backward()

        self.compute_special_losses()

    def optimize_parameters(self):
        self.forward()

        # update G
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()

