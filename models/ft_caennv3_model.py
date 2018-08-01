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
import warnings

class FTCAENNV3Model(BaseModel):
    def name(self):
        return 'FTCAENNV3Model'

    @staticmethod
    def modify_commandline_options(parser, is_train=True):

        def str2bool(v):
            if v.lower() in ('yes', 'true', 't', 'y', '1'):
                return True
            elif v.lower() in ('no', 'false', 'f', 'n', '0'):
                return False
            else:
                raise argparse.ArgumentTypeError('Boolean value expected.')
                
        parser.add_argument("--ctx_gen_det_skip", type=str2bool, nargs='?', const=True, default=False,
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
        parser.add_argument("--weight_norm", type=str2bool, nargs='?', default=True,
                    help="Use weight norm?")
        parser.add_argument('--beta', type=float, default=1.,
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

        parser.add_argument('--ctx_as_residual', action='store_true', help='use context as residual of cvae')
        parser.add_argument('--log_stdv', type=float, default=None, help='stdv for logistic loss.')
        parser.add_argument('--force_use_posterior', action='store_true', help='use posterior and use input of det net.')

        return parser

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.isTrain = opt.isTrain
        # specify the training losses you want to print out. The program will call base_model.get_current_losses
        self.loss_names = ['G', 'FFTVisiable', 'FFTInvisiable', 'KL', 'VAE', 'div', 'bits_per_dim']
        # specify the images you want to save/display. The program will call base_model.get_current_visuals
        self.visual_names = ['real_A', 'fake_B', 'real_B']
        # specify the models you want to save to the disk. The program will call base_model.save_networks and base_model.load_networks
        if self.isTrain:
            self.model_names = ['G', 'CVAE']
        else:  # during test time, only load Gs
            self.model_names = ['G', 'CVAE']
        # load/define networks
        self.netG = networks.define_G(opt.input_nc, 2, opt.ngf,
                        opt.which_model_netG, opt.norm, not opt.no_dropout, opt.init_type, self.gpu_ids, no_last_tanh=True)
        
        if self.isTrain and not opt.continue_train:
            assert(opt.preload_G or opt.train_G)
            if opt.preload_G:
                self.model_names = ['G']
                self.load_networks('0')
                self.model_names = ['G', 'CVAE']

        self.netCVAE = init_net(CVAE(opt, in_channels=3, ctx_in_channels=7, out_channels=opt.output_nc), opt.init_type, self.gpu_ids) 

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
            
            if self.opt.output_nc == 2:
                # the imagnary part of reconstrued data
                self.imag_gt = torch.cuda.FloatTensor(opt.batchSize, 1, opt.fineSize, opt.fineSize)


        assert not opt.ctx_gen_det_skip
        if 'random' in self.opt.checkpoints_dir:
            assert 'random' in self.opt.dynamic_mask_type
        
    def set_input1(self, input):
        # output from FT loader
        # BtoA is used to test if the network is identical
        img, _, _ = input
        img = img.to(self.device)

        self.mask = self.gen_random_mask(batchSize=img.shape[0])

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

    def set_input2(self, input):
        # for MRI data
        input, target, mask, metadata = input
        input = input.to(self.device)
        if len(input.shape) > 4:
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
            self.mask = mask[:,:1,:,:1,0] #(b,1,h,1)

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
    def test(self, sampling=False):
        with torch.no_grad():
            self.forward(sampling=sampling)

    def forward(self, sampling=False):
        ## CNN part
        h, b = self.mask.shape[2], self.real_A.shape[0]
        mask = Variable(self.mask)
        fake_B_Gs, mask_embed = self.netG(self.real_A, mask)

        fake_B_G3 = torch.cat([a[:,:1,:,:] for a in fake_B_Gs], 1)
        self.fake_B_G = fake_B_Gs[-1]

        if not self.opt.train_G:
            self.fake_B_G = self.fake_B_G.detach()
            fake_B_G3 = fake_B_G3.detach()
            mask_embed = mask_embed.detach()

        ## CVAE part    
        sample_from = 'prior' if sampling else 'posterior'

        if not self.isTrain and self.opt.force_use_posterior:
            real_B = self.fake_B_G 
        else:
            real_B = self.real_B

        '''Setup input'''
        X = torch.cat([real_B[:,:1,:,:], mask_embed], 1) # inference network input
        ctx = torch.cat([fake_B_G3, mask_embed, self.real_A], 1)  # context network input

        # added at output
        if self.opt.ctx_as_residual:
            residual = self.real_A
        else:
            residual = self.fake_B_G # take the determnistic output

        self.fake_B, self.div_obj, self.aux_loss, self.dec_log_stdv = self.netCVAE(X, ctx, sample_from)
        # residual training
        ft_x = self.FFT(self.fake_B)
        self.fake_B = (self.IFFT((1 - self.mask) * ft_x) + residual) # just drop imaginary part

        if len(self.opt.gpu_ids) > 1:
            self.dec_log_stdv = self.dec_log_stdv.mean()

        # no need to learn it
        if self.opt.log_stdv is not None:
            self.dec_log_stdv = torch.cuda.FloatTensor(1).fill_(self.opt.log_stdv).detach()

        self.loss_div = self.div_obj.mean()

    def compute_KL(self):
        return float(self.loss_div)

    def compute_logistic(self):
        avg_batch = True
        c = 1 # the dimension to compute loss

        # TODO: we current only compute logistic on the real part
        self.loss_VAE, _loss = criteria(self.fake_B[:,:c,:,:].div(2), self.real_B[:,:c,:,:].div(2), self.div_obj, 
                                    self.aux_loss, self.dec_log_stdv, self.opt, 
                                    ctx=None, ids=[], mask=None, ft_stdv=None, avg_batch=avg_batch)
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

    def sampling(self, data_list, n_samples=8, max_display=8, return_all=False, sampling=True):
        def replicate_tensor(data, times, expand=False):
            ks = list(data.shape)
            data = data.view(ks[0], 1, ks[1], ks[2], ks[3])
            data = data.repeat(1, times, 1, 1, 1) # repeat will copy memories which we do not need here
            if not expand:
                data = data.view(ks[0]*times, ks[1], ks[2], ks[3])
            return data
        
        if not sampling:
            warnings.warn('sampling is set to False', UserWarning)
        max_display = min(max_display, n_samples)
        # data points [N,2,H,W] for multiple sampling and observe sample difference
        data = data_list[0]
        if len(data.shape) > 4:
            data = data.squeeze(1).permute(0,3,1,2)
        assert(data.shape[0] >= max_display)
        data = data[:max_display]
        b,c,h,w = data.shape
        all_pixel_diff = []
        all_pixel_avg = []

        if n_samples*b < 128:
            repeated_data = replicate_tensor(data, n_samples)
            repeated_data_list = [repeated_data] + data_list[1:] # concat extra useless input
            self.set_input(repeated_data_list)
            self.test(sampling=sampling)
            sample_x = self.fake_B.cpu()[:,:1,:,:] # bxn_samples
        else:
            # for larger samples, do batch forward
            print(f'> sampling {n_samples} times of {b} input ...')
            repeated_data = replicate_tensor(data, n_samples, expand=True) # five dimension
            sample_x = []
            for rdata in repeated_data.transpose(0,1):
                repeated_data_list = [rdata] + data_list[1:] # concat extra useless input
                self.set_input(repeated_data_list)
                self.test(sampling=sampling)
                sample_x.append(self.fake_B.cpu()[:,:1,:,:])
            sample_x = torch.stack(sample_x, 1)

        input_imgs = repeated_data.cpu()[:,:1,:,:]
        c = 1
        # compute sample mean std
        all_pixel_diff = (input_imgs.view(b,n_samples,c,h,w) - sample_x.view(b,n_samples,c,h,w)).abs()
        pixel_diff_mean = torch.mean(all_pixel_diff, dim=1) # n_samples
        pixel_diff_std = torch.std(all_pixel_diff, dim=1) # n_samples

        if return_all:# return all samples
            return sample_x.view(b,n_samples,c,h,w)
        else:    
            # return 
            mean_samples_x = sample_x.view(b,n_samples,c,h,w).mean(dim=1)
            sample_x = sample_x.view(b,n_samples,c,h,w)[:,:max_display,:,:,:].contiguous()
            sample_x[:,0,:,:] = data[:,:1,:,:] # GT
            # sample_x[:,-1,:,:] = pixel_diff_mean.div(pixel_diff_mean.max()).mul_(2).add_(-1) # MEAN
            sample_x[:,-1,:,:] = pixel_diff_std.div(pixel_diff_std.max()).mul_(2).add_(-1) # STD
            sample_x[:,1,:,:] = mean_samples_x # E[x]

            sample_x = sample_x.view(b*max_display,c,h,w)
        
        return sample_x, pixel_diff_mean, pixel_diff_std

def discretized_logistic(mean, logscale, binsize = 1/256.0, sample=None):
    # from PixelCNN++
    scale = torch.exp(logscale)
    sample = (torch.floor(sample / binsize) * binsize - mean) / scale
    logp = torch.log(F.sigmoid(sample + binsize / scale) - F.sigmoid(sample) + 1e-7)
    return logp
def logsumexp(x):
    x_max, _ = torch.max(x, dim=1, keepdim=True)
    return x_max.view(-1) + torch.log(torch.sum(torch.exp(x - x_max), 1))
def compute_lowerbound(log_pxz, sum_kl_costs, k=1):
    if k == 1:
        return sum_kl_costs - log_pxz
    # log 1/k \sum p(x | z) * p(z) / q(z | x) = -log(k) + logsumexp(log p(x|z) + log p(z) - log q(z|x))
    log_pxz = log_pxz.view([-1, k])
    sum_kl_costs = sum_kl_costs.view([-1, k])
    return - (- torch.log(torch.Tensor(np.copy(k)).cuda()) + logsumexp(log_pxz - sum_kl_costs))
def criteria(x, orig_x, div_loss, aux_loss, dec_log_stdv, hps, ctx=None, ids=[], mask=None, ft_stdv=None, avg_batch=True):
    if hps.loss_rec == 'MSE': # gray scale images
        if hps.model == 'WAE':
            log_pxz = F.mse_loss(x, orig_x, reduce=False)
        else:
            log_pxz = - F.mse_loss(x, orig_x, reduce=False)
            one_over_var = torch.exp(-dec_log_stdv)
            log_pxz = - (0.5 * (one_over_var * -log_pxz + dec_log_stdv))

    elif hps.loss_rec == 'NLL':
        # be careful about the - 0.5
        x = torch.clamp(x, -0.5 + 1 / 512., 0.5 - 1 / 512.)
        orig_x.clamp_(-0.5, 0.5)
        log_pxz = discretized_logistic(x, dec_log_stdv, sample=orig_x)

    # mask loss for the model prior samples (context to x path)
    # TODO: should we consider reweighting for masked datapoints?
    if (hps.alpha > 0 and hps.ctx_mask_loss and ctx is not None):
        if hps.dataset == 'cifar10':
            #TODO: should we compute the loss on the imaginary part too?
            x_ft = torch.fft(x.permute(0, 2, 3, 1), 2).permute(0,3,1,2)
            orig_x_ft = torch.rfft(orig_x[:,:1,:,:], 2, onesided=False).squeeze().permute(0,3,1,2)
            log_pxz2 = discretized_logistic(x_ft, ft_stdv, sample=orig_x_ft)
            log_pxz2 = util.sum_axes(log_pxz2, axes=[1]).unsqueeze(1)
            log_pxz[ids, :, :, :] = (mask[ids, :, :, :] * log_pxz2[ids, :, :, :])
        else:
            mask = ctx[:, 1, :, :].unsqueeze(1)
            log_pxz[ids, :, :, :] = mask[ids, :, :, :] * log_pxz[ids, :, :, :]

    log_pxz = util.sum_axes(log_pxz, axes=[1, 2, 3])

    if hps.model == 'WAE':
        obj = torch.mean(log_pxz) + div_loss
        loss = obj
    else:
        if avg_batch:
            obj = torch.mean(div_loss - log_pxz)
            loss = torch.mean(compute_lowerbound(log_pxz, aux_loss, hps.k))
        else:
            obj = torch.sum(div_loss - log_pxz)
            loss = torch.sum(compute_lowerbound(log_pxz, aux_loss, hps.k))

    return obj, loss

# def crop(layer, target_size):
#     dif = [(layer.shape[2] - target_size[0]) // 2, (layer.shape[3] - target_size[1]) // 2]
#     cs=target_size

#     return layer[:, :, dif[0]:dif[0]+cs[0], dif[1]:dif[1]+cs[1]]

def conv_same(n_in, n_out, stride=1):
    ks = 4 if stride == 2 else 3
    return nn.Conv2d(n_in, n_out, kernel_size=ks, stride=stride, padding=1, bias=True)

class IAFLayer(nn.Module):
    def __init__(self, hps, downsample):
        super(IAFLayer, self).__init__()
        self.hps = hps
        self.downsample = downsample
        self.h_size = hps.h_size
        self.z_size = hps.z_size
        stride = [2, 2] if self.downsample else [1, 1]

        if hps.sum_m_s_from_gen:
            z_dim_mult = 4
        else:
            z_dim_mult = 0

        if hps.model_vae == 'IAF':
            self.ar_multiconv2d = ArMulticonv2d(hps, [self.h_size, self.h_size], [self.z_size, self.z_size])
            self.conv1down = conv_same(self.h_size, z_dim_mult * self.z_size + 2 * self.h_size)
            self.conv1up = conv_same(self.h_size, 2 * self.z_size + 2 * self.h_size, stride=stride)
        else:
            self.conv1down = conv_same(self.h_size, z_dim_mult * self.z_size + self.h_size)
            self.conv1up = conv_same(self.h_size, 2 * self.z_size + self.h_size, stride=stride)

        self.conv2up = conv_same(self.h_size, self.h_size)
        self.conv1context = conv_same(self.h_size, self.h_size + 2 * self.z_size, stride=stride)
        self.conv2context = conv_same(self.h_size, self.h_size)
        self.skip_ctx = 0
        self.qz_mean_ctx = 0
        self.qz_logsd_ctx = 0

        if downsample:
            self.pool = nn.AvgPool2d(2, ceil_mode=True)
            self.pool_context = nn.AvgPool2d(2, ceil_mode=True)
            self.upsample1 = nn.Upsample(scale_factor = 2, mode = 'nearest')
            self.conv2down = nn.ConvTranspose2d(self.h_size + self.z_size, self.h_size, 4, stride = 2, padding =1)
        else:
            self.conv2down = conv_same(self.h_size + self.z_size, self.h_size)

        if hps.weight_norm:
            self.conv1up = weightNorm(self.conv1up)
            self.conv2up = weightNorm(self.conv2up)
            self.conv1down = weightNorm(self.conv1down)
            self.conv2down = weightNorm(self.conv2down)
            self.conv1context = weightNorm(self.conv1context)
            self.conv2context = weightNorm(self.conv2context)

    def context(self, x):
        input = x
        x = F.elu(x)
        x = self.conv1context(x)
        self.qz_mean_ctx = x[:,:self.z_size,:,:]
        self.qz_logsd_ctx = x[:,self.z_size:2*self.z_size,:,:]
        h = x[:,2*self.z_size:2*self.z_size+self.h_size,:,:]
        h = F.elu(h)
        h = self.conv2context(h)
        if self.downsample:
            if self.hps.ctx_gen_det_skip:
                self.skip_ctx = input
            input = self.pool_context(input)

        return input + 0.1 * h

    def up(self, x):
        input = x
        x = F.elu(x)
        x = self.conv1up(x)
        self.qz_mean = x[:,:self.z_size,:,:]
        self.qz_logsd = x[:,self.z_size:2*self.z_size,:,:]
        h = x[:,2*self.z_size:2*self.z_size+self.h_size,:,:]
        if self.hps.model_vae == 'IAF':
            self.up_context = x[:,2*self.z_size+self.h_size:,:,:]
        h = F.elu(h)
        h = self.conv2up(h)
        if self.downsample:
            input = self.pool(input)

        return input + 0.1 * h

    def down(self, x, ids, target_size, sample_from):

        hps = self.hps
        input = x
        x = F.elu(x)
        x = self.conv1down(x)
        if hps.sum_m_s_from_gen:
            pz_mean = x[:,:self.z_size,:,:]
            pz_logsd = x[:,self.z_size:2*self.z_size,:,:]
            rz_mean = x[:,2*self.z_size:3*self.z_size,:,:]
            rz_logsd = x[:,3*self.z_size:4*self.z_size,:,:]
            z_dim_mult = 4
        else:
            pz_mean = 0.
            pz_logsd = 0.
            rz_mean = 0.
            rz_logsd = 0.
            z_dim_mult = 0

        h_det = x[:,z_dim_mult*self.z_size:z_dim_mult*self.z_size + self.h_size,:,:]
        prior = torch.distributions.Normal(pz_mean + self.qz_mean_ctx, 
                                               torch.exp(pz_logsd + self.qz_logsd_ctx))
        
        if hps.model_vae == 'IAF':
            down_context = x[:,z_dim_mult*self.z_size + self.h_size:z_dim_mult*self.z_size + 2*self.h_size,:,:]
            context = self.up_context + down_context
        if hps.loss_div == 'KL':
            if sample_from in ["prior"]:
                # print("Sampling from prior!")
                z = prior.rsample()
                ## for testing 
                # z = pz_mean + self.qz_mean_ctx
            else:
                # print("Sampling from posterior!")
                posterior = torch.distributions.Normal(rz_mean + self.qz_mean,
                                               torch.exp(rz_logsd + self.qz_logsd))
                z = posterior.rsample()

            if sample_from == "prior":
                kl_cost = kl_obj = torch.zeros([input.shape[0]]).cuda()
            else:
                logqs = posterior.log_prob(z)
                if hps.model_vae == 'IAF':
                    x = self.ar_multiconv2d(z, context)
                    arw_mean, arw_logsd = x[0] * 0.1, x[1] * 0.1
                    z = (z - arw_mean) / torch.exp(arw_logsd)
                    logqs += arw_logsd
                logps = prior.log_prob(z)

                kl_cost = logqs - logps

                # alternate samples from prior and postirior, if sample is on ids list
                # we will take sample from prior
                if len(ids) > 0:
                    kl_cost[ids, :, :, :] = 0 * kl_cost[ids, :, :, :]
                    z[ids, :, :, :] = prior.rsample()[ids, :, :, :]

                if hps.kl_min > 0:
                    # [0, 1, 2, 3] -> [0, 1] -> [1] / (b * k)
                    # we use sum_axes insetad of torch.sum to be able to pass axes
                    kl_ave = torch.mean(util.sum_axes(kl_cost, axes=[2, 3]), 0, keepdim=True)
                    kl_ave = torch.clamp(kl_ave, min=hps.kl_min)
                    kl_ave = kl_ave.repeat([input.shape[0], 1])
                    kl_obj = torch.sum(kl_ave, 1)
                else:
                    kl_obj = torch.sum(kl_cost, [1, 2, 3])

                kl_cost = util.sum_axes(kl_cost, [1, 2, 3])
        else:
            raise ValueError('Unknown divergence.')

        h = torch.cat((z, h_det), dim=1)

        h = F.elu(h)
        h = self.conv2down(h)

        if self.downsample:
            input = self.upsample1(input)
            if hps.ctx_gen_det_skip:
                input = input + self.skip_ctx

        output = input + 0.1 * h

        return output, kl_obj, kl_cost

    def forward(self, x, ctx=None, up=True, sample_from='None', ids=[], target_size=None):
        assert sample_from in ('prior', 'posterior')
        if up == True:
            # x = self.up(x)
            if sample_from == 'prior' and ctx is not None:
                x = None
            else:
                x = self.up(x)

            if ctx is not None:
                ctx = self.context(ctx)
            return x, ctx
        else:
            if target_size is None:
                target_size = x.shape[-2:]
            x, kl_obj, kl_cost = self.down(x, ids, target_size, sample_from)
            return x, kl_obj, kl_cost
            
class CVAE(nn.Module):
    def __init__(self, hps, in_channels, ctx_in_channels, out_channels):
        super(CVAE, self).__init__()
        self.hps = hps
        self.m_trunc = []
        self.convinit = conv_same(in_channels, hps.h_size, stride=2)
        if hps.std_in_nll_dep_on_z:
            self.deconvfin = nn.ConvTranspose2d(hps.h_size, 2 * out_channels, 4, stride=2, padding=1)
            self.out_channels = 2 * out_channels
        else:
            self.deconvfin = nn.ConvTranspose2d(hps.h_size, out_channels, 4, stride=2, padding=1)
            self.dec_log_stdv =  nn.Parameter(torch.zeros(1,1))

        if hps.n_ctx > 0 and not hps.use_ctx_as_input:
            self.convinit_ctx = conv_same(ctx_in_channels, hps.h_size, stride=2)
        else:
            self.h_top = nn.Parameter(torch.zeros(hps.h_size))

        if not hps.ctx_gen_det_conn:
            self.h_top = nn.Parameter(torch.zeros(hps.h_size))

        if hps.weight_norm:
            self.convinit = weightNorm(self.convinit)
            self.deconvfin = weightNorm(self.deconvfin)
            if hps.n_ctx > 0 and not hps.use_ctx_as_input:
                self.convinit_ctx = weightNorm(self.convinit_ctx)

        self.layers = nn.ModuleList([])
        for i in range(hps.depth):
            self.layers.append(nn.ModuleList([]))
            for j in range(hps.num_blocks):
                downsample = (i > 0) and (j == 0)
                self.layers[-1].append(IAFLayer(hps, downsample))

        self.IFFT = IFFT()
        self.FFT = FFT()

    # def forward(self, x, ctx, true_ctx, sample_from, att_w, ids=[]): # this one will cause in parallel_apply bugs
    def forward(self, x, ctx, sample_from='None', ids=[]):
        assert sample_from in ('prior', 'posterior')
        input_size = x.shape[-2:]
        hps = self.hps
        # Input images are repeated k times on the input.
        # This is used for Importance Sampling loss (k is number of samples).
        batch_size = x.shape[0] * hps.k
        if hps.k > 1:
            x = x.repeat(hps.k, 1, 1, 1)
            ctx = ctx.repeat(hps.k, 1, 1, 1) if ctx is not None else None
        
        h = self.convinit(x)
        if ctx is not None:
            h_ctx = self.convinit_ctx(ctx)
        else:
            h_ctx = None
        
        shapes = []
        for i, layer in enumerate(self.layers):
            for j, sub_layer in enumerate(layer):
                h, h_ctx = sub_layer(h, h_ctx, up=True, sample_from=sample_from)
                shapes.append(h_ctx.shape[-2:])
        shapes = shapes[:-1]

        if h_ctx is not None and hps.ctx_gen_det_conn:
            h = h_ctx
        else:
            h_top = self.h_top
            h_top = h_top.view([1, -1, 1, 1])
            h = h_top.repeat([batch_size, 1, shapes[-1][0], shapes[-1][1]])
    
        kl_cost = Variable(torch.zeros([batch_size])).cuda()
        kl_obj = Variable(torch.zeros([batch_size])).cuda()
        for i, layer in reversed(list(enumerate(self.layers))):
            for j, sub_layer in reversed(list(enumerate(layer))):
                h, cur_obj, cur_cost = sub_layer(h, up=False, ids=ids,
                                                 target_size=shapes[i*len(layer)+j-1] if i*len(layer)+j-1 > 0 else None,
                                                 sample_from=sample_from)
                kl_obj = cur_obj + kl_obj
                kl_cost = cur_cost + kl_cost

        x = F.elu(h)
        x = self.deconvfin(x)
        if hps.std_in_nll_dep_on_z:
            dec_log_stdv = x[:,self.out_channels:,:,:]
            dec_log_stdv = torch.log(F.softplus(dec_log_stdv) + 0.00001)
            x = x[:,:self.out_channels,:,:]
        else:
            dec_log_stdv = self.dec_log_stdv

        if hps.loss_rec == 'NLL':
            pass # process outside
        else:
            x = F.sigmoid(x) - 0.5

        return x, kl_obj, kl_cost, dec_log_stdv

class ArMulticonv2d(nn.Module):
    
    def __init__(self, hps, n_h, n_out, nl=F.elu):
        super(ArMulticonv2d, self).__init__()
    # name, x, context, n_h, n_out, nl=nn.elu, ):
        self.hps = hps
        self.n_h = n_h
        self.n_out = n_out
        z_size = hps.z_size
        self.nl = nl
        self.layersA = nn.ModuleList([])
        self.layersB = nn.ModuleList([])
        n_in = z_size
        for i, n_filter in enumerate(n_h):
            if hps.weight_norm:
                # print('adding wn')
                self.layersA.append(weightNorm(MaskedConv2d_tf(False, n_in, n_filter, 3, padding=(3 - 1) // 2, bias=True)))
            else:
                self.layersA.append(MaskedConv2d_tf(False, n_in, n_filter, 3, padding=(3 - 1) // 2, bias=True))
            n_in = n_filter

        for i, n_filter in enumerate(n_out):
            if hps.weight_norm:
                self.layersB.append(weightNorm(MaskedConv2d_tf(True, n_in, n_filter, 3, padding=(3 - 1) // 2, bias=True)))
            else:
                self.layersB.append(MaskedConv2d_tf(True, n_in, n_filter, 3, padding=(3 - 1) // 2, bias=True))


    def forward(self, x, context):
        for i, layer in enumerate(self.layersA):
            x=layer(x)
            if i == 0:
                x += context
            x=self.nl(x)
        out = []
        for i, layer in enumerate(self.layersB):
            out.append(layer(x))
        return out
class MaskedConv2d_tf(nn.Conv2d):

    def __init__(self, zerodiagonal, *args, **kwargs):
        super(MaskedConv2d_tf, self).__init__(*args, **kwargs)
        _, _, h, w = self.weight.size()
        n_in = self.in_channels
        n_out = self.out_channels
        # TODO: Check if the masks should be flipped or not...
        mask = torch.Tensor(np.copy(get_conv_ar_mask(h, w, n_in, n_out, zerodiagonal, flip_mask=False))).cuda()
        self.register_buffer('mask', mask)


    def forward(self, x):
        self.weight.data *= self.mask
        return super(MaskedConv2d_tf, self).forward(x)