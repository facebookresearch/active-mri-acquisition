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

class FTRECURNNModel(BaseModel):
    def name(self):
        return 'FTRECURNNModel'

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

    def set_input1(self, input):
        # output from FT loader
        # BtoA is used to test if the network is identical
        AtoB = self.opt.which_direction == 'AtoB'
        img, _, _ = input
        img = img.to(self.device)

        self.mask = self.gen_random_mask(batchSize=img.shape[0])
        
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

    def gen_random_mask(self, batchSize=1):
        if self.isTrain and self.opt.dynamic_mask_type != 'None' and not self.validation_phase:
            if self.opt.dynamic_mask_type == 'random':
                mask = create_mask((batchSize, self.opt.fineSize), random_frac=True, mask_fraction=self.opt.kspace_keep_ratio).to(self.device)
            elif self.opt.dynamic_mask_type == 'random_lines':
                seed = np.random.randint(10000)
                mask = create_mask((batchSize, self.opt.fineSize), random_frac=False, mask_fraction=self.opt.kspace_keep_ratio, seed=seed).to(self.device)
        else:
            mask = create_mask(self.opt.fineSize, random_frac=False, mask_fraction=self.opt.kspace_keep_ratio).to(self.device)
            
        return mask

    def set_input2(self, input):
        # for MRI data
        input, target, mask, metadata = input
        input = input.to(self.device)
        input = input.squeeze(1).permute(0,3,1,2)
        target = target.to(self.device)
        mask = mask.to(self.device)
        ifft_img = self.IFFT(input, normalized=True) # this has to be normalized IFFT
        
        if self.isTrain and self.opt.dynamic_mask_type != 'None' and not self.validation_phase:
            self.mask = self.gen_random_mask(batchSize=ifft_img.shape[0])
            fft_kspace = self.RFFT(target)
            ifft_img = self.IFFT(fft_kspace * self.mask)
        else:
            # use masked as provided
            self.mask = mask[:1,:1,:,:1,0] #(1,1,h,1)

        if self.opt.output_nc == 2:
            if self.imag_gt.shape[0] != target.shape[0]:
                # imagnary part is all zeros
                self.imag_gt = torch.zeros_like(target)
            target = torch.cat([target, self.imag_gt], dim=1)

        self.real_A = ifft_img
        self.real_B = target

    def set_input(self, input):
        if self.mri_data:
            self.set_input2(input)
        else:
            self.set_input1(input)

    def compute_special_losses(self):
        # compute losses between fourier spaces of fake_B and real_B
        # if output one dimension
        # import pdb ; pdb.set_trace()
        if self.fake_B.shape[1] == 1:
            _k_fakeB = self.RFFT(self.fake_B)
            _k_realB = self.RFFT(self.real_B)
        else:
            # if output are two dimensional
            _k_fakeB = self.FFT(self.fake_B)
            _k_realB = self.FFT(self.real_B)

        b = self.fake_B.shape[0] if self.mask.shape[0] == 1 else 1
    
        mask_deno = self.mask.sum() * b * self.fake_B.shape[1] * self.fake_B.shape[3]
        invmask_deno = (1-self.mask).sum() * b * self.fake_B.shape[1] * self.fake_B.shape[3]

        self.loss_FFTVisiable = F.mse_loss(_k_fakeB * self.mask, _k_realB*self.mask, reduce=False).sum().div(mask_deno)
        self.loss_FFTInvisiable = F.mse_loss(_k_fakeB * (1-self.mask), _k_realB*(1-self.mask), reduce=False).sum().div(invmask_deno)
        
        return float(self.loss_FFTVisiable), float(self.loss_FFTInvisiable)

    def forward(self):
        # conditioned on mask
        h, b = self.mask.shape[2], self.real_A.shape[0]
        mask = Variable(self.mask.view(self.mask.shape[0],1,h,1).expand(b,1,h,1))

        fake_Bs, _ = self.netG(self.real_A, mask)

        # the condition depends on the network output
        self.fake_Bs = fake_Bs
        self.fake_B = fake_Bs[-1]
        
    def backward_G(self):
        # First, G(A) should fake the discriminator
        self.loss_G = 0
        for fake_B in self.fake_Bs:
            self.loss_G += self.criterion(fake_B, self.real_B)
        
        self.loss_G.backward()

        self.compute_special_losses()

    def optimize_parameters(self):
        self.forward()

        # update G
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()

