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

class FTATTGANModel(BaseModel):
    def name(self):
        return 'FTATTGANModel'

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.set_defaults(pool_size=0)
        parser.set_defaults(no_lsgan=False)
        parser.set_defaults(norm='instance')
        parser.set_defaults(dataset_mode='aligned')
        # parser.set_defaults(which_model_netG='unet_256')
        if is_train:
            parser.add_argument('--lambda_L1', type=float, default=100.0, help='weight for L1 loss')

        return parser

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.isTrain = opt.isTrain
        # specify the training losses you want to print out. The program will call base_model.get_current_losses
        self.loss_names = ['G', 'FFTVisiable', 'FFTInvisiable', 'G_GAN', 'G_all', 'D_real', 'D_fake']
        # specify the images you want to save/display. The program will call base_model.get_current_visuals
        self.visual_names = ['real_A', 'fake_B', 'real_B']
        # specify the models you want to save to the disk. The program will call base_model.save_networks and base_model.load_networks
        if self.isTrain:
            self.model_names = ['G','D']
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
        self.IRFFT = IRFFT().to(self.device)

        if self.isTrain:
            use_sigmoid = opt.no_lsgan
            self.netD = networks.define_D((opt.input_nc + opt.output_nc) if not self.opt.no_cond_gan else opt.output_nc, opt.ndf,
                                          opt.which_model_netD,
                                          opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, self.gpu_ids)

            self.fake_AB_pool = ImagePool(opt.pool_size)
            # define loss functions
            self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan).to(self.device)

        if self.isTrain:
            # define loss functions
            self.criterionL1 = torch.nn.MSELoss()

            # initialize optimizers
            self.optimizers = []
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

        if self.opt.output_nc == 2:
            # the imagnary part of reconstrued data
            self.imag_gt = torch.cuda.FloatTensor(opt.batchSize, 1, opt.fineSize, opt.fineSize)

    def set_input(self, input):
        # output from FT loader
        # BtoA is used to test if the network is identical
        AtoB = self.opt.which_direction == 'AtoB'
        img, _, _ = input
        img = img.to(self.device)
        
        if self.isTrain and self.opt.dynamic_mask_type != 'None' and not self.validation_phase:
            if self.opt.dynamic_mask_type == 'random':
                self.mask = create_mask(self.opt.fineSize, random_frac=True, mask_fraction=self.opt.kspace_keep_ratio).to(self.device)
            elif self.opt.dynamic_mask_type == 'random_plus':
                seed = np.random.randint(100)
                self.mask = create_mask(self.opt.fineSize, random_frac=False, mask_fraction=self.opt.kspace_keep_ratio, seed=seed).to(self.device)
        else:
            self.mask = create_mask(self.opt.fineSize, random_frac=False, mask_fraction=self.opt.kspace_keep_ratio).to(self.device)

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
        mask = Variable(self.mask.view(self.mask.shape[0],1,h,1))#.expand(b,h,1,1))

        self.fake_B = self.netG(self.real_A) # two channels

        ft_x = self.FFT(self.fake_B)
        self.fake_B = self.IFFT((1 - mask) * ft_x) + self.real_A

        # self.real_A = self.real_A[:,:1,:,:]
        # self.fake_B = self.fake_B2[:,:1,:,:]
        # self.real_B2 = self.real_B
        # self.real_B = self.real_B[:,:1,:,:]
    
    def _clamp(self, data):
        data = data.clamp(-1,1)
        data[:,1,:,:] = 0
        return data
        
    def backward_D(self):
        # Fake
        # stop backprop to the generator by detaching fake_B
        if self.opt.no_cond_gan:
            fake_AB = self.fake_AB_pool.query(self._clamp(self.fake_B))
        else:
            fake_AB = self.fake_AB_pool.query(torch.cat((self.real_A, self._clamp(self.fake_B)), 1))
        pred_fake = self.netD(fake_AB.detach()) # 14x14 for 128x128 # TODO clamp to prevent IIFT floating issue
        self.loss_D_fake = self.criterionGAN(pred_fake, False)

        # Real
        if self.opt.no_cond_gan:
            real_AB = self.real_B
        else:
            real_AB = torch.cat((self.real_A, self.real_B), 1)
            
        pred_real = self.netD(real_AB)
        self.loss_D_real = self.criterionGAN(pred_real, True)

        # Combined loss
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5

        self.loss_D.backward()

    def backward_G(self):
        # First, G(A) should fake the discriminator
        
        if self.opt.no_cond_gan:
            fake_AB = self._clamp(self.fake_B) 
        else:
            fake_AB = torch.cat((self.real_A, self._clamp(self.fake_B)), 1)

        pred_fake = self.netD(fake_AB)
        self.loss_G_GAN = self.criterionGAN(pred_fake, True) * 1/self.opt.lambda_L1

        # Second, G(A) = B
        self.loss_G = self.criterionL1(self.fake_B, self.real_B) 

        # third
        self.loss_G_all = self.loss_G_GAN + self.loss_G

        self.loss_G_all.backward()

        self.compute_special_losses()

    def optimize_parameters(self):
        self.forward()
        # update D
        self.set_requires_grad(self.netD, True)
        self.optimizer_D.zero_grad()
        self.backward_D()
        self.optimizer_D.step()

        # update G
        self.set_requires_grad(self.netD, False)
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()
