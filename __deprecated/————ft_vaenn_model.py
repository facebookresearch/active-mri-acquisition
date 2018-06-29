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


class FTVAENNModel(BaseModel):
    def name(self):
        return 'FTVAENNModel'

    @staticmethod
    def modify_commandline_options(parser, is_train=True):

        parser.set_defaults(nz=8)
        parser.set_defaults(which_model_netG='jure_unet_vae_residual')
        parser.set_defaults(output_nc=1) # currently, use 2
        if is_train:
            parser.add_argument('--lambda_KL', type=float, default=1, help='weight for KL loss')
            parser.add_argument('--kl_min_clip', type=float, default=0.0, help='clip value for KL loss')

        return parser

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.isTrain = opt.isTrain
        # specify the training losses you want to print out. The program will call base_model.get_current_losses
        self.loss_names = ['G', 'FFTVisiable', 'FFTInvisiable', 'KL']
        # specify the images you want to save/display. The program will call base_model.get_current_visuals
        self.visual_names = ['real_A', 'fake_B', 'real_B']
        # specify the models you want to save to the disk. The program will call base_model.save_networks and base_model.load_networks
        if self.isTrain:
            self.model_names = ['G', 'E_prior','E_posterior']
        else:  # during test time, only load Gs
            self.model_names = ['G', 'E_prior']
        # load/define networks
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf,
                                    opt.which_model_netG, opt.norm, not opt.no_dropout, opt.init_type, self.gpu_ids, no_last_tanh=True)

        norm = 'none' if 'jure_unet' in opt.which_model_netG else opt.norm
        self.netE_prior = networks.define_E(opt.input_nc, opt.nz, opt.ngf, 'conv_128', 
                                    norm, 'lrelu', opt.init_type, self.gpu_ids, vaeLike=True)
        self.netE_posterior = networks.define_E(opt.input_nc, opt.nz, opt.ngf, 'conv_128', 
                                    norm, 'lrelu', opt.init_type, self.gpu_ids, vaeLike=True)  

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
        # doing FFT
        # if has two dimension output, 
        # we actually want toe imagary part is also supervised, which should be all zero
        fft_kspace = self.RFFT(img)
        if self.opt.output_nc == 2:
            if self.imag_gt.shape[0] != img.shape[0]:
                # imagnary part is all zeros
                self.imag_gt = torch.zeros_like(img)
            img = torch.cat([img, self.imag_gt], dim=1)

        if AtoB:
            self.real_A = self.IFFT(fft_kspace * self.mask)
            self.real_B = img
            self.real_A_ = self.IFFT(fft_kspace * (1-self.mask))
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

    # def get_z_random(self, batchSize, nz, random_type='gauss'):
    #     if random_type == 'uni':
    #         self.eps = torch.rand(batchSize, nz) * 2.0 - 1.0
    #     elif random_type == 'gauss':
    #         self.eps = torch.randn(batchSize, nz)
        
    def reparam(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = Variable(torch.cuda.FloatTensor(mu.size()).normal_())
        q_z = eps.mul(std).add_(mu)

        return q_z

    def decode_distribution(self, sampling):
        
        p_mu, p_logvar = self.netE_prior.forward(self.real_A)
        p_z = self.reparam(p_mu, p_logvar)
        
        # compute KL loss
        if self.isTrain:
            q_mu, q_logvar = self.netE_posterior.forward(self.real_A_)
            q_z = self.reparam(q_mu, q_logvar)
            self.loss_KL = self.kld(q_mu, q_logvar, p_mu, p_logvar)
        else:
            self.loss_KL = 0
        if sampling:
            # in the training stage, return posteriors
            return p_z
        else:
            # in testing, return learned priors
            return q_z

    def test(self):
        with torch.no_grad():
            self.forward(sampling=True)

    def kld(self, mu1, logvar1, mu2, logvar2):
        
        prior = torch.distributions.Normal(mu2, torch.exp(logvar2))
        posterior = torch.distributions.Normal(mu1, torch.exp(logvar1))

        z = posterior.rsample()
        logqs = posterior.log_prob(z)
        logps = prior.log_prob(z)

        kl_obj = logqs - logps
        kl_obj = kl_obj.sum(1).mean()
        
        return kl_obj

    def forward(self, sampling=False):
        
        z = self.decode_distribution(sampling)

        if 'residual' in self.opt.which_model_netG:
            self.fake_B, self.fake_B_res = self.netG(self.real_A, z=z)
        else: 
            self.fake_B = self.netG(self.real_A, z=z)

    def backward_G(self):
        self.loss_G = self.criterion(self.fake_B, self.real_B) 
        
        # kl divergence
        self.loss_G += self.loss_KL * self.opt.lambda_KL

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
            self.loss_G += l2_loss * 0.0001

        self.loss_G.backward()

        self.compute_special_losses()

    def optimize_parameters(self):
        self.forward()

        # update G
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()

    def sampling(self, data_list, n_samples=9):
        def replicate_tensor(data, times):
            ks = list(data.shape)
            data = data.view(ks[0], 1, ks[1], ks[2], ks[3])
            data = data.repeat(1, times, 1, 1, 1) # repeat will copy memories which we do not need here
            data = data.view(ks[0]*times, ks[1], ks[2], ks[3])
            return data

        # data points [N,2,H,W] for multiple sampling and observe sample difference
        data = data_list[0]
        assert(data.shape[0] >= n_samples)
        data = data[:n_samples]
        b,c,h,w = data.shape
        all_pixel_diff = []
        all_pixel_avg = []

        data = replicate_tensor(data, n_samples)
        repeated_data = [data] + data_list[1:] # concat extra useless output

        self.set_input(repeated_data)
        self.test()

        input_imgs = data.cpu().view(b,n_samples,c,h,w)
        sample_x = self.fake_B.cpu() # bxn_samples

        all_pixel_diff = (input_imgs - sample_x.view(b,n_samples,c,h,w)).abs()

        pixel_diff_mean = torch.mean(all_pixel_diff, dim=1) # n_samples
        pixel_diff_std = torch.std(all_pixel_diff, dim=1) # n_samples

        return sample_x, pixel_diff_mean, pixel_diff_std
