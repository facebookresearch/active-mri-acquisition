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
import torch.nn.utils.weight_norm as weightNorm
import argparse
from .networks import init_net
from .ft_caenn_model import *

class FTCAE3NNModel(BaseModel):
    def name(self):
        return 'FTCAE3NNModel'

    @staticmethod
    def modify_commandline_options(parser, is_train=True):

        def str2bool(v):
            if v.lower() in ('yes', 'true', 't', 'y', '1'):
                return True
            elif v.lower() in ('no', 'false', 'f', 'n', '0'):
                return False
            else:
                raise argparse.ArgumentTypeError('Boolean value expected.')
                
        parser.add_argument("--ctx_gen_det_skip", type=str2bool, nargs='?', const=True, default=True,
                            help="Add deterministic skip connections from context path to generator path?")
        parser.add_argument("--ctx_mask_loss", type=str2bool, nargs='?', const=True, default=False,
                            help="Mask loss to context info for samples from model prior?")
        parser.add_argument("--ctx_gen_det_conn", type=str2bool, nargs='?', const=True, default=True,
                            help="Connect ctx to generator via deterministic skip in the bottleneck?")
                    
        parser.add_argument("--std_in_nll_dep_on_z", type=str2bool, nargs='?', const=True, default=False,
                    help="Std in generator output depends on z?")             
        parser.add_argument("--use_ctx_as_input", type=str2bool, nargs='?', const=True, default=False,
                    help="Use context information as input instead of the images. This flag allows to train the model from ctx to x direclty. Note that this flag ignores conditioning the learnable prior on additional context.")
        parser.add_argument('--n_ctx', type=int, default = 1,
                    help='Number of context frames to use. Set to 0 if no context is required.')
        parser.add_argument("--sum_m_s_from_gen", type=str2bool, nargs='?', const=True, default=True,
                    help="sum mean and std in prior and posterior to values comming from generator network?")
        parser.add_argument("--weight_norm", type=str2bool, nargs='?', const=True, default=True,
                    help="Use weight norm?")
        parser.add_argument('--beta', type=float, default=4.,
                    help='Weight on KL term.')
        parser.add_argument('--no_context_mask_cond', action='store_false', help='if mask is used as the input of conditional input')
        parser.add_argument("--h_size", type=int, default=160,
                            help="Size of resnet block")
        parser.add_argument("--depth", type=int, default=1,
                            help="Number of downsampling blocks.")
        parser.add_argument("--num_blocks", type=int, default=5,
                    help="Number of resnet blocks for each downsampling layer.")
        parser.add_argument("--loss_rec", type=str, choices=['MSE','NLL'], default='NLL', help="Which loss for p(x).")
        parser.add_argument("--z_size", type=int, default=32,
                    help="Size of z variables.")
        parser.add_argument("--k", type=int, default=1,
                    help="Number of samples for IS objective.")
        parser.add_argument("--loss_div", type=str, choices=['KL','MMD', 'GAN'], default='KL', help="Which loss for q(z|x).")
        parser.add_argument("--kl_min", type=float, default=0.25,
                    help="Number of free bits/nats.")
        parser.add_argument('--alpha', type=float, default = 0.,
                    help='Ratio of samples from the prior at training time.')
        parser.add_argument("--model_vae", type=str, default='VAE', choices=['VAE', 'IAF', 'WAE'],
                    help="Use flow, VAE model or WAE model.")

        parser.add_argument('--preload_G', action='store_true', help='pre_load Generator')
        parser.add_argument('--train_G', action='store_true', help='also train Generator')
        parser.add_argument('--cvae_attention', type=str, default='mask', choices=['mask', 'softatt','None'], help='use attention')

        parser.add_argument('--optimize_beta', action='store_true', help='increase KL weight during training')
        parser.add_argument('--ctx_as_residual', action='store_true', help='use context as residual of cvae')

        return parser

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.isTrain = opt.isTrain
        # specify the training losses you want to print out. The program will call base_model.get_current_losses
        self.loss_names = ['G', 'FFTVisiable', 'FFTInvisiable', 'KL', 'VAE', 'div', 'bits_per_dim', 'beta']
        # specify the images you want to save/display. The program will call base_model.get_current_visuals
        self.visual_names = ['real_A', 'fake_B', 'real_B']
        # specify the models you want to save to the disk. The program will call base_model.save_networks and base_model.load_networks
        if self.isTrain:
            self.model_names = ['G', 'CVAE']
        else:  # during test time, only load Gs
            self.model_names = ['G', 'CVAE']
        # load/define networks
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf,
                        opt.which_model_netG, opt.norm, not opt.no_dropout, opt.init_type, self.gpu_ids, no_last_tanh=True)
        
        if self.isTrain and not opt.continue_train:
            assert(opt.preload_G or opt.train_G)
            if opt.preload_G:
                self.model_names = ['G']
                self.load_networks('0')
                self.model_names = ['G', 'CVAE']

        self.netCVAE = init_net(CVAE(opt, in_channels=6, ctx_in_channels=2, out_channels=2), opt.init_type, self.gpu_ids) 

        self.loss_beta = self.opt.beta if not self.opt.optimize_beta else 0.5
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
            params = list(self.netCVAE.parameters()) + (list(self.netG.parameters()) if opt.train_G else [])
            # self.optimizer_G = torch.optim.Adam(params, lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_G = torch.optim.Adamax(params, lr=opt.lr)

            self.optimizers.append(self.optimizer_G)

        assert not opt.ctx_gen_det_skip
        
    def set_input(self, input):
        # output from FT loader
        # BtoA is used to test if the network is identical
        img, _, _ = input
        img = img.to(self.device)
        # doing FFT
        # if has two dimension output, 
        # we actually want toe imagary part is also supervised, which should be all zero
        fft_kspace = self.RFFT(img)
        # TODO: be careful of the normalization [-0.5, 0.5]
        ifft_img = self.IFFT(fft_kspace * self.mask)

        if not self.opt.no_context_mask_cond:
            b,c,h,w = fft_kspace.shape
            mask = self.mask.expand(b,c,h,w)
            fft_mask = self.RFFT(mask)
            self.real_A = torch.cat([ifft_img, fft_mask], 1) # has four channels
        else:
            self.real_A = ifft_img   

        self.real_B = self.IFFT(fft_kspace)
        # self.real_A_ = self.IFFT(fft_kspace * (1-self.mask)) # inversed masked ifft, the input of encoder of cvae 
 
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

    def test(self, sampling=False):
        with torch.no_grad():
            self.forward(sampling=sampling)

    def forward(self, sampling=False):
        ## CNN part
        h, b = self.mask.shape[2], self.real_A.shape[0]
        mask = Variable(self.mask.view(1,h,1,1).expand(b,h,1,1))
        self.fake_B_G, kspace_attention = self.netG(self.real_A, mask)
        self.fake_A_ = self.fake_B_G - self.real_A # estimated unobserved counterpart of real_A
        if not self.opt.train_G:
            self.fake_A_ = self.fake_A_.detach()
            kspace_attention = kspace_attention.detach()
        
        ## CVAE part
        sample_from = 'prior' if sampling else 'posterior'

        # normalize to [-0.5, 0.5] by dividing 2
        # then normalize back
        if self.opt.cvae_attention == 'None':
            att_w = None
        elif self.opt.cvae_attention == 'mask':
            att_w = 1 - self.mask.view(1,1,h,1)
        elif self.opt.cvae_attention == 'softatt':
            att_w = kspace_attention
        else:
            ValueError('Unknown cvae attention type')
        
        # errors from the determnistic mdoel
        E = self.real_B - self.fake_B_G 
        # unobserved part
        U = self.real_B - self.real_A 
        X = torch.cat([self.real_B, U, E], 1)
        # observed part as 
        ctx = self.real_A
        # added at output
        if self.opt.ctx_as_residual:
            residual = self.real_A
        else:
            residual = self.fake_B_G # take the determnistic output
        
        # self.fake_B, self.div_obj, self.aux_loss, self.dec_log_stdv, self.ft_stdv  = self.netCVAE(X, ctx, att_w, sample_from=sample_from)
        self.fake_B, self.div_obj, self.aux_loss, self.dec_log_stdv, self.ft_stdv, beta = self.netCVAE(X, ctx, sample_from)
        
        # add residual 
        if att_w is None:
            ## way 1 of fusing ctx on pixelspace
            self.fake_B = self.fake_B + residual[:,:2,:,:]
        else:
            ## way 2 of fusing ctx on kspace
            ft_x = self.FFT(self.fake_B)
            self.fake_B = self.IFFT(att_w * ft_x) + residual

        if len(self.opt.gpu_ids) > 1:
            self.dec_log_stdv = self.dec_log_stdv.mean()
            self.ft_stdv = self.ft_stdv.mean()

        if self.opt.optimize_beta:
            self.loss_beta = beta.mean().clamp_(0.2, 0.8)

        self.loss_div = self.div_obj.mean()

    def compute_KL(self):
        return float(self.loss_div)

    def compute_logistic(self):
        avg_batch = True
        c = 1 # the dimension to compute loss

        # TODO: we current only compute logistic on the real part
        KL_div = self.div_obj.mul(self.loss_beta)
        logp_beta = (1-self.loss_beta) if self.opt.optimize_beta else 1
        self.loss_VAE, _loss = criteria(self.fake_B[:,:c,:,:].div(2), self.real_B[:,:c,:,:].div(2), KL_div, 
                                    self.aux_loss, self.dec_log_stdv, self.opt, 
                                    ctx=None, ids=[], mask=None, ft_stdv=self.ft_stdv, avg_batch=avg_batch, logp_beta=logp_beta)
        b,_,h,w = self.fake_B.shape
        cst = np.log(2.) if self.opt.loss_rec == 'NLL' else 1.
        self.loss_bits_per_dim = _loss.item() / (cst*c*h*w) / (1 if avg_batch else b)
        
        return float(self.loss_bits_per_dim)

    def backward_G(self):
        
        self.loss_G = self.criterion(self.fake_B_G, self.real_B) # from resnet
        # normalize loss_VAE
        self.compute_logistic()

        loss_total = self.loss_VAE 
        if self.opt.train_G:
            loss_total += self.loss_G
        loss_total.backward()

        self.compute_special_losses()

    def optimize_parameters(self):
        self.forward()

        # update G
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()

    def sampling(self, data_list, n_samples=8, max_display=8, return_all=False):
        def replicate_tensor(data, times, expand=False):
            ks = list(data.shape)
            data = data.view(ks[0], 1, ks[1], ks[2], ks[3])
            data = data.repeat(1, times, 1, 1, 1) # repeat will copy memories which we do not need here
            if not expand:
                data = data.view(ks[0]*times, ks[1], ks[2], ks[3])
            return data
        
        max_display = min(max_display, n_samples)
        # data points [N,2,H,W] for multiple sampling and observe sample difference
        data = data_list[0]
        assert(data.shape[0] >= max_display)
        data = data[:max_display]
        b,c,h,w = data.shape
        all_pixel_diff = []
        all_pixel_avg = []

        if n_samples*b < 128:
            repeated_data = replicate_tensor(data, n_samples)
            repeated_data_list = [repeated_data] + data_list[1:] # concat extra useless input
            self.set_input(repeated_data_list)
            self.test(sampling=True)
            sample_x = self.fake_B.cpu()[:,:1,:,:] # bxn_samples
        else:
            # for larger samples, do batch forward
            print(f'> sampling {n_samples} times of {b} input ...')
            repeated_data = replicate_tensor(data, n_samples, expand=True) # five dimension
            sample_x = []
            for rdata in repeated_data.transpose(0,1):
                repeated_data_list = [rdata] + data_list[1:] # concat extra useless input
                self.set_input(repeated_data_list)
                self.test(sampling=True)
                sample_x.append(self.fake_B.cpu()[:,:1,:,:])
            sample_x = torch.stack(sample_x, 1)

        input_imgs = repeated_data.cpu()
        
        # compute sample mean std
        all_pixel_diff = (input_imgs.view(b,n_samples,c,h,w) - sample_x.view(b,n_samples,c,h,w)).abs()
        pixel_diff_mean = torch.mean(all_pixel_diff, dim=1) # n_samples
        pixel_diff_std = torch.std(all_pixel_diff, dim=1) # n_samples

        if return_all:
            return sample_x.view(b,n_samples,c,h,w)
        else:    
            sample_x = sample_x.view(b,n_samples,c,h,w)[:,:max_display,:,:,:].contiguous()
            sample_x[:,0,:,:] = data[:,:1,:,:]
            sample_x[:,-1,:,:] = pixel_diff_mean.div(pixel_diff_mean.max()).mul_(2).add_(-1)
            
            sample_x = sample_x.view(b*max_display,c,h,w)
        

        return sample_x, pixel_diff_mean, pixel_diff_std