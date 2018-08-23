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
from torch import nn
import warnings
from torch.nn.parameter import Parameter

from .ft_recurnnv2_model import AUTOPARAM
import math
from .ft_pasgan_model import kspaceMap

class FTPASGANABLATIONModel(BaseModel):
    def name(self):
        return 'FTPASGANABLATIONModel'

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        
        if is_train:
            parser.add_argument('--lambda_gan', type=float, default=0.01, help='weight for rec loss')
            parser.add_argument('--use_allgen_for_disc', action='store_true', help='use all intermediate outputs from generators to train D')

        parser.add_argument('--loss_type', type=str, default='MSE', choices=['MSE','L1'], help=' loss type')
        parser.add_argument('--betas', type=str, default='0,0,0', help=' beta of sparsity loss')
        parser.add_argument('--sparsity_norm', type=float, default=0, help='norm of sparsity loss. It should be smaller than 1')
        parser.add_argument('--where_loss_weight', type=str, choices=['full','logvar'], default='full', help='the type of learnable W to apply to')
        parser.add_argument('--scale_logvar_each', action='store_true', help='scale logvar map each')
        parser.add_argument('--clip_weight', type=float, default=0, help='clip loss weight to prevent it to too small value')
        
        parser.add_argument('--no_init_kspacemap_embed', action='store_true', help='do *not* init the weight of kspacemap module')
        parser.add_argument('--no_kspacemap_embed', action='store_true', help='do *not* use embed of kspace embedding')
        parser.add_argument('--use_fixed_weight', type=str, default='1,1,1', help='use fixed weight for all loss')
        parser.add_argument('--mask_cond', action='store_true', help='condition on mask')
        parser.add_argument('--use_mse_as_disc_energy', action='store_true', help='use mse as disc energy')

        if not is_train:
            parser.add_argument('--no_lsgan', action='store_true', help='do *not* use least square GAN, if false, use vanilla GAN')
        
        parser.add_argument('--set_sampling_at_stage', type=int, default=None, help='sampling from the model')
        parser.add_argument('--grad_ctx', action='store_true', help='gan criterion has loss signal at provided')
        parser.add_argument('--no_zscore_clamp', action='store_true', help='clamp data using z_score')
        parser.add_argument('--no_uncertanity', action='store_true', help='no uncertainty analysis')
        parser.add_argument('--pixelwise_loss_merge', action='store_true', help='no uncertainty analysis')

        parser.set_defaults(pool_size=0)
        parser.set_defaults(which_model_netG='pasnet')
        parser.set_defaults(input_nc=2)
        parser.set_defaults(output_nc=3)
        parser.set_defaults(niter=50)
        parser.set_defaults(niter_decay=50)
        parser.set_defaults(print_freq=100)
        parser.set_defaults(norm='instance')
        parser.set_defaults(dynamic_mask_type='random')
        parser.set_defaults(no_dropout=True)
        parser.set_defaults(loadSize=144)
        parser.set_defaults(fineSize=128)
        parser.set_defaults(which_model_netD='n_layers_channel')
        parser.set_defaults(n_layers_D=4)
        parser.set_defaults(ndf=256)
        parser.set_defaults(no_lsgan=False)
        parser.set_defaults(no_kspacemap_embed=True)
        parser.set_defaults(mask_cond=True)

        return parser

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.isTrain = opt.isTrain
        # specify the training losses you want to print out. The program will call base_model.get_current_losses
        self.loss_names = ['G', 'FFTVisiable', 'FFTInvisiable', 'l2', 'uncertainty', 'sparsity', 
                            'GW_0', 'GW_1', 'GW_2', 'G_GAN', 'G_all', 'D', 'visible_disc', 'invisiable_disc', 'gradOutputD']
        # specify the images you want to save/display. The program will call base_model.get_current_visuals
        self.visual_names = ['real_A', 'fake_B', 'real_B']
        # specify the models you want to save to the disk. The program will call base_model.save_networks and base_model.load_networks
        if self.isTrain:
            self.model_names = ['G','D']
        else:  # during test time, only load Gs
            self.model_names = ['G','D']

        if opt.use_fixed_weight != 'None': 
            assert opt.clip_weight == 0
            opt.use_fixed_weight = [float(a) for a in opt.use_fixed_weight.split(',')]
        else:
            opt.use_fixed_weight = None
        self.num_stage = 3 # number fo stage in pasnet
        assert(opt.output_nc == 3)
        # load/define networks
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf,
                                    opt.which_model_netG, opt.norm, not opt.no_dropout, opt.init_type, self.gpu_ids, no_last_tanh=True)
        self.netP = AUTOPARAM(length=self.num_stage, fixed_ws=opt.use_fixed_weight).to(self.device)
        self.mask = create_mask(opt.fineSize, mask_fraction=self.opt.kspace_keep_ratio).to(self.device)
        self.RFFT = RFFT().to(self.device)
        self.IFFT = IFFT().to(self.device)
        # for evaluation
        self.FFT = FFT().to(self.device)

        if self.isTrain or 'D' in self.model_names:
            use_sigmoid = opt.no_lsgan
            self.cond_input_D = False
            if opt.which_model_netD in ('n_layers_channel', 'n_layers_channel_group'):
                if opt.which_model_netD == "n_layers_channel_group":
                    self.opt.mask_cond = False
                pre_process = kspaceMap(imSize=opt.fineSize, no_embed=opt.no_kspacemap_embed).to(self.device)
                d_in_nc = opt.fineSize + (6 if self.opt.mask_cond else 0)
                self.netD = networks.define_D(d_in_nc, opt.ndf,
                                            opt.which_model_netD, opt.n_layers_D, opt.norm, use_sigmoid, 
                                            opt.init_type, self.gpu_ids, preprocess_module=pre_process)
            else:
                d_in_nc = 2 + (6 if self.opt.mask_cond else 0)
                self.netD = networks.define_D(d_in_nc, opt.ndf,
                                            opt.which_model_netD, opt.n_layers_D, opt.norm, use_sigmoid, 
                                            opt.init_type, self.gpu_ids)
                self.cond_input_D = True # conditional GAN
            # initlize the netD first layer parameter to be the sum of all channels
            if not opt.no_kspacemap_embed and not opt.no_init_kspacemap_embed:
                pre_process.init_weight()
            
        if self.isTrain:
            # self.fake_AB_pool = ImagePool(opt.pool_size)
            # define loss functions
            if 'aux' in opt.which_model_netD:
                self.criterionGAN = networks.GANLossKspaceAux(use_lsgan=not opt.no_lsgan, use_mse_as_energy=opt.use_mse_as_disc_energy).to(self.device)
            else:
                if opt.which_model_netD == 'n_layers_channel':
                    self.criterionGAN = networks.GANLossKspace(use_lsgan=not opt.no_lsgan, use_mse_as_energy=opt.use_mse_as_disc_energy, grad_ctx=self.opt.grad_ctx).to(self.device)
                else:
                    self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan).to(self.device)
            # define loss functions
            if opt.loss_type == 'MSE':
                self.criterion = torch.nn.MSELoss(reduce=False) 
            elif opt.loss_type == 'L1':
                self.criterion = torch.nn.L1Loss(reduce=False) 
            # initialize optimizers
            self.optimizers = []
            self.optimizer_G = torch.optim.Adam(list(self.netG.parameters()) + list(self.netP.parameters()),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            # self.optimizer_P = torch.optim.SGD(self.netP.parameters(), momentum=0.9)
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

        if self.opt.output_nc >= 2:
            # the imagnary part of reconstrued data
            self.imag_gt = torch.cuda.FloatTensor(opt.batchSize, 1, opt.fineSize, opt.fineSize)

        self.zscore = 3 if not opt.no_zscore_clamp else 0

        self.betas = [float(a) for a in self.opt.betas.split(',')]
        assert len(self.betas) == self.num_stage, 'beta length is euqal to the module #'
        
    def certainty_loss(self, fake_B, real_B, logvar, beta, stage, weight_logvar, weight_all):
        
        o = int(np.floor(self.opt.output_nc/2))
        # gaussian nll loss
        l2 = self.criterion(fake_B[:,:o,:,:], real_B[:,:o,:,:]) 
        if self.opt.no_uncertanity:
            loss = l2
        else:
            # to be numercial stable we clip logvar to make variance in [0.01, 5]
            logvar = logvar.clamp(-4.605, 1.609)
            one_over_var = torch.exp(-logvar)
            # uncertainty loss
            assert len(l2) == len(logvar)
            loss = 0.5 * (one_over_var * l2 + logvar * weight_logvar)
        if not self.opt.pixelwise_loss_merge:
            loss = loss.mean()
        
        # l0 sparsity distance
        if beta != 0:
            b,c,h,w = logvar.shape
            k = b*c*h*w
            var = logvar.exp()
            sparsity = var.norm(self.opt.sparsity_norm).div(k) * beta 
        else:
            sparsity = 0
        
        full_loss = loss + sparsity
        # record for plot
        if stage == 0: 
            self.loss_l2 = self.loss_uncertainty = self.loss_sparsity = 0

        self.loss_l2 += l2.mean()
        self.loss_uncertainty += logvar.exp().mean()
        self.loss_sparsity += sparsity

        full_loss = full_loss * weight_all

        return full_loss

    def get_current_histograms(self):
        histogarms = {}
        for i, logvar in enumerate(self.logvars):
            histogarms[f'uncertainty_stage{i}'] = logvar.exp()
        return histogarms

    def set_input(self, input):
        if self.mri_data:
            if len(input) == 4:
                input = input[1:]
            self.set_input2(input, zscore=self.zscore)
        else:
            self.set_input1(input)

    def forward(self, sampling=False):
        # conditioned on mask
        h, b = self.mask.shape[2], self.real_A.shape[0]
        mask = Variable(self.mask.view(self.mask.shape[0],1,h,1).expand(b,1,h,1))
        if sampling:
            # may not useful
            assert not sampling 
            self.fake_Bs, self.logvars, self.mask_cond = self.netG(self.real_A, mask, self.metadata, self.opt.set_sampling_at_stage)
        else:    
            self.fake_Bs, self.logvars, self.mask_cond = self.netG(self.real_A, mask, self.metadata)

        self.fake_B = self.fake_Bs[-1]

    def test(self, sampling=False):
        with torch.no_grad():
            self.forward(sampling)

    def create_D_input(self, fake_B):
        fake_B = fake_B[:,:1,:,:] # discard imaginary part
        fake_B = self._clamp(fake_B)
        if self.opt.mask_cond:
            fake = torch.cat([fake_B, self.mask_cond.detach()], dim=1)
        else:
            fake = fake_B
        if self.cond_input_D:
            fake = torch.cat([fake, self.real_A[:,:1,:,:]], dim=1)
        return fake

    def backward_G(self):
        # First, G(A) should fake the discriminator
        weights = self.netP()
        if self.opt.clip_weight > 0:
            weights.clamp_(self.opt.clip_weight, self.num_stage-self.opt.clip_weight*(self.num_stage-1))

        self.loss_G = 0
        for stage, (fake_B, logvar, beta) in enumerate(zip(self.fake_Bs, self.logvars, self.betas)):
            if self.opt.where_loss_weight == 'full':
                w_full = weights[stage]
                w_lv = 1
            elif self.opt.where_loss_weight == 'logvar':
                w_lv = weights[stage]
                w_full = 1
            self.loss_G += self.certainty_loss(fake_B, self.real_B, logvar, beta, stage, 
                                                weight_logvar=w_lv, weight_all=w_full) 
        if self.opt.pixelwise_loss_merge:
            self.loss_G = self.loss_G.mean()
            
        # track norm of gradient
        fake_B = self.fake_B.mul(1)
        gradnorm_fakeAB = torch.cuda.FloatTensor(1)
        def extract(grad):
            global set_input2
            gradnorm_fakeAB = grad.norm(2)
        fake_B.register_hook(extract)

        fake = self.create_D_input(fake_B)
        pred_fake = self.netD(fake, self.mask)
        self.loss_G_GAN = self.criterionGAN(pred_fake, True, self.mask, degree=1, updateG=True, pred_gt=(fake[:,:1,:,:],self.real_B)) * self.opt.lambda_gan

        self.loss_G_all = self.loss_G_GAN + self.loss_G
        self.loss_G_all.backward()

        # compute multiple loss for visualization
        self.compute_special_losses()
        self.loss_gradOutputD = float(gradnorm_fakeAB.item())
        for i in range(weights.shape[0]):
            setattr(self, f'loss_GW_{i}', float(weights[i].item()))

    def _clamp(self, data):
        # process data for D input
        # make consistent range with real_B for inputs of D
        if self.mri_data:
            if self.zscore != 0:
                data = data.clamp(-self.zscore, self.zscore) 
        else:
            data = data.clamp(-1, 1) 
        return data
    
    def forward_D(self):
        # to get the disc score of visiable and invisiable parts during testing
        with torch.no_grad():
            fake = self.create_D_input(self.fake_B)
            pred_fake = self.netD(fake, self.mask) 
            if isinstance(pred_fake, tuple):
                pred_fake = pred_fake[1]
        return self.mask_disc_score(pred_fake, self.mask.squeeze()), pred_fake

    def mask_disc_score(self, pred, mask):
        if isinstance(pred, tuple):
            pred = pred[1] # for discriminator has two output takes the second one
        if len(pred.shape) > 2:
            return 0, 0
        mask = mask.mul(1) # copy mask
        if isinstance(pred, tuple): 
            pred = pred[1]
        # pred, mask [B,H]
        self.loss_invisiable_disc = (pred * (1-mask)).sum().div_((1-mask).sum())
        # the lowfreq part is always zero 
        # we do not want to monitor it and affact the average
        mask[:,:3] = 0
        mask[:,-3:] = 0
        self.loss_visible_disc = (pred * mask).sum().div_(mask.sum())
        return float(self.loss_visible_disc.item()), float(self.loss_invisiable_disc.item())

    def backward_D(self):
        # Fake
        # stop backprop to the generator by detaching fake_B
        fake = self.create_D_input(self.fake_B.detach())
        pred_fake = self.netD(fake, self.mask) 

        degree = 0 if not self.opt.use_allgen_for_disc else 0.2
        self.loss_D_fake = self.criterionGAN(pred_fake, False, self.mask, degree=degree, pred_gt=(fake[:,:1,:,:],self.real_B))
        self.mask_disc_score(pred_fake, self.mask.squeeze())

        if self.opt.use_allgen_for_disc:
            for p, deg in zip([-2, -3], [0.1, 0.0]):
                pred_fake = self.create_D_input(self.fake_Bs[p].detach())
                pred_fake = self.netD(pred_fake, self.mask) 
                self.loss_D_fake += self.criterionGAN(pred_fake, False, self.mask, degree=deg, pred_gt=(fake[:,:1,:,:],self.real_B))
            
            self.loss_D_fake = self.loss_D_fake / self.num_stage

        # Real
        real = self.create_D_input(self.real_B)
        pred_real = self.netD(real, self.mask)
        self.loss_D_real = self.criterionGAN(pred_real, True, self.mask, degree=1, pred_gt=(fake[:,:1,:,:],self.real_B))

        # Combined loss
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5

        self.loss_D.backward()

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

    # change to sampling() when want to use it
    def _sampling(self, data_list, n_samples=8, max_display=8, return_all=False, sampling=True):
        
        
        def replicate_tensor(data, times, expand=False):
            ks = list(data.shape)
            data = data.view(ks[0], 1, *ks[1:])
            data = data.repeat(1, times, *[1 for _ in range(len(ks[1:]))]) # repeat will copy memories which we do not need here
            if not expand:
                data = data.view(ks[0]*times, *ks[1:])
            return data
        
        if not sampling:
            warnings.warn('sampling is set to False', UserWarning)
        assert self.mri_data, 'not working for non mri data loader now'
        max_display = min(max_display, n_samples)
        # data points [N,2,H,W] for multiple sampling and observe sample difference
        data = data_list[0] # target
        mask = data_list[1]
        metadata = data_list[2]

        assert(data.shape[0] >= max_display)
        data = data[:max_display]
        mask = mask[:max_display]

        # if n_samples*b < 128:
        repeated_data = replicate_tensor(data, n_samples)
        repeated_mask = replicate_tensor(mask, n_samples)
        scan_type = metadata['scan_type'][:max_display]
        scan_type = np.tile(np.array(scan_type)[:,np.newaxis],(1,n_samples))
        metadata['scan_type'] = list(scan_type.reshape(-1))
        repeated_metadata = metadata

        b,c,h,w = data.shape
        # all_pixel_diff = []
        # all_pixel_avg = []

        repeated_data_list = [repeated_data, repeated_mask, repeated_metadata] # concat extra useless input
        self.set_input(repeated_data_list)
        self.test(sampling=True)
        sample_x = self.fake_B.cpu()[:,:1,:,:] # bxn_samples
        input_x = self.real_A.cpu()[:,:1,:,:] # bxn_samples
        # else:
        #     # for larger samples, do batch forward
        #     print(f'> sampling {n_samples} times of {b} input ...')
        #     repeated_data = replicate_tensor(data, n_samples, expand=True) # five dimension
        #     sample_x = []
        #     input_x = []
        #     for rdata in repeated_data.transpose(0,1):
        #         repeated_data_list = [rdata] + data_list[1:] # concat extra useless input
        #         self.set_input(repeated_data_list)
        #         self.test(sampling=sampling)
        #         sample_x.append(self.fake_B.cpu()[:,:1,:,:])
        #         input_x.append(self.real_A.cpu()[:,:1,:,:])
        #     sample_x = torch.stack(sample_x, 1)
        #     input_x = torch.stack(input_x, 1)

        input_imgs = repeated_data.cpu()
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
            input_x = input_x.view(b,n_samples,c,h,w)
            sample_x[:,0,:,:] = input_x[:,0,:1,:,:] # Input
            sample_x[:,1,:,:] = data[:,:1,:,:] # GT
            sample_x[:,2,:,:] = mean_samples_x # E[x]
            # sample_x[:,-1,:,:] = pixel_diff_mean.div(pixel_diff_mean.max()).mul_(2).add_(-1) # MEAN
            sample_x[:,-1,:,:] = pixel_diff_std.div(pixel_diff_std.max()).mul_(2).add_(-1) # STD
            
            sample_x = sample_x.view(b*max_display,c,h,w)
        
        # self.netG.module.set_sampling_at_stage(None)

        return sample_x, pixel_diff_mean, pixel_diff_std

    
    def set_input_exp(self, input, mask, zscore=3):
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