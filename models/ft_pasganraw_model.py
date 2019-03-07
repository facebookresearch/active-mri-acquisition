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
import functools
from .ft_recurnnv2_model import AUTOPARAM
import math, sys
from util import visualizer

class GANLossKspace(nn.Module):
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0, 
                    use_mse_as_energy=False, grad_ctx=False, gamma=100):
        super(GANLossKspace, self).__init__()
        # self.register_buffer('real_label', torch.ones(imSize, imSize))
        # self.register_buffer('fake_label', torch.zeros(imSize, imSize))
        self.grad_ctx = grad_ctx
        if use_lsgan:
            self.loss = nn.MSELoss(size_average=False)
        else:
            self.loss = nn.BCELoss(size_average=False)
        self.use_mse_as_energy = use_mse_as_energy
        if use_mse_as_energy:
            self.FFT = FFT()
            self.gamma = gamma
            self.bin = 5

    def get_target_tensor(self, input, target_is_real, degree, mask, pred_gt=None):
        
        if target_is_real:
            target_tensor = torch.ones_like(input)
            target_tensor[:] = degree

        else:
            target_tensor = torch.zeros_like(input)
            if not self.use_mse_as_energy:
                if degree != 1:
                    target_tensor[:] = degree
            else:
                pred, gt = pred_gt 
                h = gt.shape[3]
                ks_gt = self.FFT(gt, normalized=True) 
                ks_input = self.FFT(pred[:,:2,:,:], normalized=True) 
                ks_row_mse = F.mse_loss(ks_input, ks_gt, reduce=False).sum(1,keepdim=True).sum(3,keepdim=True).squeeze() / (2*h)
                energy = torch.exp(-ks_row_mse * self.gamma)

                ## do some bin process
                # import pdb; pdb.set_trace()
                # energy = torch.floor(energy * 10 / self.bin) * self.bin / 10
                
                target_tensor[:] = energy
            # force observed part to always
            for i in range(mask.shape[0]):
                idx = torch.nonzero(mask[i,0,:,0])
                target_tensor[i,idx] = 1 
        return target_tensor

    def __call__(self, input, target_is_real, mask, degree=1, updateG=False, pred_gt=None):
        # input [B, imSize]
        # degree is the realistic degree of output
        # set updateG to True when training G.
        target_tensor = self.get_target_tensor(input, target_is_real, degree, mask, pred_gt)
        b,h = target_tensor.shape
        if updateG and not self.grad_ctx:
            mask_ = mask.squeeze()
            # maskout the observed part loss
            return self.loss(input * (1-mask_), target_tensor * (1-mask_)) / (1-mask_).sum()
        else:
            return self.loss(input, target_tensor) / (b*h)

class kspaceMap(nn.Module):
    def __init__(self, imSize=128, no_embed=False):
        super(kspaceMap, self).__init__()
        
        self.FFT = FFT()
        self.IFFT = IFFT()
        self.imSize = imSize
        # seperate image to spectral maps
        self.register_buffer('seperate_mask', torch.FloatTensor(1,imSize,1,imSize,1))
        self.seperate_mask.fill_(0)
        for i in range(imSize):
            self.seperate_mask[0,i,0,i,0] = 1

        self.no_embed = no_embed
        if not no_embed:
            self.embed = nn.Sequential(
                            nn.Conv2d(imSize, imSize, 1, 1, padding=0, bias=False),
                            nn.LeakyReLU(0.2,True)
            )
        else:
            print(f'[kspaceMap] -> do not use kspace embedding')

    def init_weight(self, p=0.2):
        # init the weight of embed to be Bernoulli distribuion of all 1.
        # so just like a dropout, so the each filter output approximate the sum of all spectral maps
        print(f'[kspaceMap] -> init weight as Bernoulli distribuion ({p})')
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data[:] = 1
                seed = torch.zeros_like(m.weight.data).normal_()
                m.weight.data[seed < p] = 0

    def forward(self, input, mask):
        # we assume the input only has real part of images (if they are obtained from IFFT)
        bz = input.shape[0]
        x = input[:,:2,:,:]
        if input.shape[1] > 2:
            others = input[:,2:,:,:]
        kspace = self.FFT(x)
        h,w = input.shape[2:]
        
        kspace = kspace.unsqueeze(1).repeat(1,self.imSize,1,1,1) # [B,imsize,2,imsize,imsize]
        # kspace = kspace.unsqueeze(1).expand(bz,self.imSize,2,h,w) # [B,imsize,2,imsize,imsize]
        masked_kspace = self.seperate_mask * kspace
        masked_kspace = masked_kspace.view(bz*self.imSize, 2, self.imSize, self.imSize)
        seperate_imgs = self.IFFT(masked_kspace).norm(dim=1).view(bz, self.imSize, self.imSize, self.imSize) # discard the imaginary part [B,imsize,imsize, imsize]

        if not self.no_embed:
            seperate_imgs = self.embed(seperate_imgs)

        if input.shape[1] > 2:
            output = torch.cat([seperate_imgs, others], 1)
        else:
            output = seperate_imgs

        return output

class FTPASGANRAWModel(BaseModel):
    def name(self):
        return 'FTPASGANRAWModel'

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
        parser.add_argument('--pixelwise_loss_merge', action='store_true', help='no uncertainty analysis')
        parser.add_argument('--gamma', type=int, default=20, help='energy function gamma')

        parser.set_defaults(pool_size=0)
        parser.set_defaults(which_model_netG='pasnetplus_nomaskcond_320')
        parser.set_defaults(input_nc=2)
        parser.set_defaults(output_nc=3)
        parser.set_defaults(niter=50)
        parser.set_defaults(niter_decay=50)
        parser.set_defaults(print_freq=100)
        parser.set_defaults(norm='instance')
        parser.set_defaults(dynamic_mask_type='random')
        parser.set_defaults(no_dropout=True)
        parser.set_defaults(loadSize=360)
        parser.set_defaults(fineSize=320)
        parser.set_defaults(which_model_netD='n_layers_channel_320')
        parser.set_defaults(n_layers_D=4)
        parser.set_defaults(ndf=256)
        parser.set_defaults(no_lsgan=False)
        parser.set_defaults(no_kspacemap_embed=True)
        parser.set_defaults(mask_cond=True)
        parser.set_defaults(grad_ctx=True)
        parser.set_defaults(pixelwise_loss_merge=True)
        parser.set_defaults(use_mse_as_disc_energy=True)
        parser.set_defaults(eval_full_valid=True)
        # # 0.16 will around have 
        # parser.set_defaults(kspace_keep_ratio=0.16)

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
            if opt.which_model_netD in ('n_layers_channel', 'n_layers_channel_320'):
                pre_process = kspaceMap(imSize=opt.fineSize, no_embed=opt.no_kspacemap_embed).to(self.device)
                d_in_nc = opt.fineSize + (6 if self.opt.mask_cond else 0)
                self.netD = networks.define_D(d_in_nc, opt.ndf,
                                            opt.which_model_netD, opt.n_layers_D, opt.norm, use_sigmoid, 
                                            opt.init_type, self.gpu_ids, preprocess_module=pre_process)
            else:
                d_in_nc = 4 + (6 if self.opt.mask_cond else 0)
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
                raise NotImplementedError
                self.criterionGAN = networks.GANLossKspaceAux(use_lsgan=not opt.no_lsgan).to(self.device)
            else:
                if opt.which_model_netD in ('n_layers_channel', 'n_layers_channel_320'):
                    self.criterionGAN = GANLossKspace(use_lsgan=not opt.no_lsgan, 
                                        use_mse_as_energy=opt.use_mse_as_disc_energy, grad_ctx=self.opt.grad_ctx, 
                                        gamma=self.opt.gamma).to(self.device)
                else:
                    raise NotImplementedError
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
            if not opt.debug:
                assert opt.eval_full_valid 

        if self.opt.output_nc >= 2:
            # the imagnary part of reconstrued data
            self.imag_gt = torch.cuda.FloatTensor(opt.batchSize, 1, opt.fineSize, opt.fineSize)

        self.zscore = 10

        self.betas = [float(a) for a in self.opt.betas.split(',')]
        assert len(self.betas) == self.num_stage, 'beta length is euqal to the module #'
        
        
    def certainty_loss(self, fake_B, real_B, logvar, beta, stage, weight_logvar, weight_all):
        
        o = 2
        # gaussian nll loss
        l2 = self.criterion(fake_B[:,:o,:,:], real_B[:,:o,:,:]) 
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

    # def set_input(self, input):
    #     assert self.mri_data
    #
    #     # for MRI data Slice loader
    #     input, target, mask, metadata = input
    #     target = target.to(self.device)
    #     input = input.to(self.device)
    #     self.mask = mask.to(self.device)
    #     self.metadata = None
    #
    #     # do clip internally in loader
    #     # target = self._clamp(target).detach()
    #     # input = self._clamp(input)
    #
    #     self.real_A = input
    #     self.real_B = target

    def set_input(self, input):
        assert self.mri_data

        # for MRI data Slice loader
        mask, target, input = input
        target = target.to(self.device)
        input = input.to(self.device)
        self.mask = mask.to(self.device)
        # self.metadata = None

        self.real_A = input
        self.real_B = target

    def forward(self, sampling=False):
        # conditioned on mask
        h, b = self.mask.shape[2], self.real_A.shape[0]
        import pdb; pdb.set_trace()
        # mask = Variable(self.mask.view(self.mask.shape[0],1,h,1).expand(b,1,h,1))
        mask = Variable(self.mask)
        self.fake_Bs, self.logvars, self.mask_cond = self.netG(self.real_A, mask, False)

        self.fake_B = self.fake_Bs[-1]
        
    def test(self, sampling=False):
        with torch.no_grad():
            self.forward(sampling)

    def create_D_input(self, fake_B):

        fake_B = self._clamp(fake_B)
        if self.opt.mask_cond:
            fake = torch.cat([fake_B, self.mask_cond.detach()], dim=1)
        else:
            fake = fake_B
        if self.cond_input_D:
            fake = torch.cat([fake, self.real_A], dim=1)
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
        self.loss_G_GAN = self.criterionGAN(pred_fake, True, self.mask, degree=1, updateG=True, pred_gt=(fake, self.real_B)) * self.opt.lambda_gan

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
        return self.mask_disc_score(pred_fake, self.mask.squeeze()), pred_fake

    def mask_disc_score(self, pred, mask):
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
        pred_fake = None

        pred_fake = self.netD(fake, self.mask) 
        degree = 0 
        self.loss_D_fake = self.criterionGAN(pred_fake, False, self.mask, degree=degree, pred_gt=(fake,self.real_B))
        self.mask_disc_score(pred_fake, self.mask.squeeze())

        # Real
        real = self.create_D_input(self.real_B)
        pred_real = self.netD(real, self.mask)
        self.loss_D_real = self.criterionGAN(pred_real, True, self.mask, degree=1, pred_gt=(fake,self.real_B))

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
    
    def set_input_exp(self, input, mask, zscore=3, add_kspace_noise=False):
        # used for test kspace scanning line recommentation
        target, _, metadata = input
        target = target.to(self.device)
        # self.metadata = self.metadata2onehot(metadata, dtype=type(target)).to(self.device)
        self.metadata = None
        target = self._clamp(target).detach()

        if len(mask.shape) == 5:
            mask = mask[:1,:1,:,:1,0].to(self.device).repeat(target.shape[0],1,1,1)
        self.mask = mask
            
        fft_kspace = self.FFT(target)
        if add_kspace_noise:
            noises = torch.zeros_like(fft_kspace).normal_()
            fft_kspace = fft_kspace + noises

        ifft_img = self.IFFT(fft_kspace * self.mask)

        self.real_A = ifft_img
        self.real_B = target


    def validation(self, val_data_loader, how_many_to_display=36, how_many_to_valid=4096*4, n_samples=8, metasavepath=None):

        if self.mri_data:
            tensor2im = functools.partial(util.tensor2im, renormalize=False)
        else:
            pass

        val_data = []
        val_count = 0
        need_sampling = hasattr(self, 'sampling') 
        losses = {
            'reconst_loss': [],
            'reconst_ssim': [],
            'FFTVisiable_loss': [],
            'FFTInvisiable_loss': []
        }
        if need_sampling:
            losses['sampling_loss'] = []
            losses['sampling_ssim'] = []

        if hasattr(self,'forward_D'):
            losses['visible_disc'] = []
            losses['invisible_disc'] = []

        netG = getattr(self, 'netG')
        
        # turn on the validation sign
        self.validation_phase = True

        # visualization for a fixed number of data
        if not hasattr(self, 'display_data') or self.display_data is None:
            self.display_data = []
            for data in val_data_loader:
                if self.mri_data:
                    assert len(data) == 4
                self.display_data.append(data)
                val_count += data[0].shape[0]
                if val_count >= how_many_to_display: break
                   
        # prepare display data
        for it, data in enumerate(self.display_data):
            self.set_input(data)
            self.test() # Weird. using forward will cause a mem leak
            c = min(self.fake_B.shape[0], how_many_to_display)
            ## visualize input, output, gt
            real_A, fake_B, real_B, = self.real_A[:c,...].cpu(), self.fake_B[:c,...].cpu(), self.real_B[:c,...].cpu() # save mem
            if not hasattr(self, 'logvars'):
                val_data.append([real_A[:c,:2,...], fake_B[:c,:2,...], real_B[:c,:2,...]])            
            else:
                ## visualize uncertainty map
                if type(self.logvars) is not list:
                    self.logvars = [self.logvars]

                ## apply the same normalization for the (3) logvars maps for each image 
                b = self.logvars[0].shape[0]
                self.logvars = torch.stack(self.logvars,1).exp_()

                if self.opt.scale_logvar_each:
                    maxv = self.logvars.max(2)[0].max(2)[0].max(2)[0]
                    maxv = maxv.view(maxv.shape[0],maxv.shape[1],1,1,1)
                else:
                    maxv = self.logvars.max(1)[0].max(1)[0].max(1)[0].max(1)[0]
                    maxv = maxv.view(maxv.shape[0],1,1,1,1)
                self.logvars = self.logvars / maxv
                ## organize it to visable format, each line shows [vnd] u.map and 
                ## the following len(self.logvars) lines show u.maps of the same image at different stages
                vnd = int(np.sqrt(how_many_to_display)) 
                log_var_vis = []
                self.logvars = self.logvars.transpose(0,1)
                for s in range(2):
                    star, end = s*vnd, (s+1)*vnd
                    logvars = [logvar[star:end,...].cpu() for logvar in self.logvars]
                    logvars = torch.cat(logvars, 0)
                    log_var_vis.append(logvars)
                log_var_vis = torch.cat(log_var_vis, 0)
                val_data.append([real_A[:c,:2,...], fake_B[:c,:2,...], real_B[:c,:2,...], log_var_vis])

        # if self.mri_data:
        #     for i in range(3): # do not normalize variance only the first three
        #         for a in val_data: 
        #             # print(a[i].min().item(), a[i].max().item())
        #             util.mri_denormalize(a[i], zscore=self.zscore)

        if metasavepath is not None:
            ## save a pickle for inspection
            import pickle
            pickle_file = {}
            pickle_file['var'] = val_data[0][3].cpu().numpy()
            pickle_file['gt'] = val_data[0][2].cpu().numpy()
            pickle_file['rec'] = val_data[0][1].cpu().numpy()
            pickle_file['input'] = val_data[0][0].cpu().numpy()
            pickle_file['residual'] = np.abs(pickle_file['gt'] - pickle_file['rec'])
            pickle_file['mask'] = self.mask.repeat(1,1,1,128).cpu().numpy()
            pickle.dump(pickle_file, open(metasavepath,'wb'))

        visuals = {}
        input_tensor = torch.cat([a[0] for a in val_data], dim=0)[:how_many_to_display].norm(dim=1, keepdim=True)
        recon_tensor = torch.cat([a[1] for a in val_data], dim=0)[:how_many_to_display].norm(dim=1, keepdim=True)
        gt_tensor = torch.cat([a[2] for a in val_data], dim=0)[:how_many_to_display].norm(dim=1, keepdim=True)
        visuals['inputs'] = tensor2im(tvutil.make_grid(input_tensor, nrow=int(np.sqrt(how_many_to_display)), normalize=True, scale_each=True ))
        visuals['reconstructions'] = tensor2im(tvutil.make_grid(recon_tensor, nrow=int(np.sqrt(how_many_to_display)), normalize=True, scale_each=True ))
        visuals['groundtruths'] = tensor2im(tvutil.make_grid(gt_tensor, nrow=int(np.sqrt(how_many_to_display)) , normalize=True, scale_each=True))  
        # residual map comparison
        diff_rec = np.abs(gt_tensor - recon_tensor)
        diff_input = np.abs(gt_tensor - input_tensor)
        imsize = val_data[0][0].shape[2]
        n = diff_rec.shape[0] # how many samples
        diff = torch.zeros_like(diff_rec)
        diff[range(1,n,2),...] = diff_rec[:n//2,...] # show side by side
        diff[range(0,n,2),...] = diff_input[:n//2,...] # show side by side
        visuals['differences']= tensor2im(tvutil.make_grid(diff, nrow=int(np.sqrt(how_many_to_display))), renormalize=False) 
        visuals['differences'] = visualizer.gray2heatmap(visuals['differences'][:,:,0], cmap='gray') # view as heat map

        # show uncertainty_map images
        if hasattr(self, 'logvars'):
            _tmp = tvutil.make_grid(torch.cat([a[3] for a in val_data], dim=0)[:how_many_to_display], normalize=False, scale_each=False, nrow=vnd)
            # conver to rgb heat map
            _tmp = util.tensor2im(_tmp, renormalize=False)
            _tmp = visualizer.gray2heatmap(_tmp[:,:,0]) # view as heat map
            # _tmp = np.tile(_tmp[:,:,:1], (1,1,3)) # view as gray map
            visuals['certainty_map'] = _tmp

        ## show ssim map
        _, ssim_map = util.ssim_metric(recon_tensor[:,:2,:,:].norm(dim=1,keepdim=True), gt_tensor[:,:2,:,:].norm(dim=1,keepdim=True), full=True)
        visuals['ssim_map'] = tensor2im(tvutil.make_grid(ssim_map[:how_many_to_display], nrow=int(np.sqrt(how_many_to_display))), renormalize=False) 

        if need_sampling:
            # we need to do sampling
            sample_x, pixel_diff_mean, pixel_diff_std = self.sampling(self.display_data[0], n_samples=n_samples)
            
            if self.mri_data: util.mri_denormalize(sample_x)

            visuals['sample'] = tensor2im(tvutil.make_grid(sample_x, nrow=int(np.sqrt(sample_x.shape[0]))))
            nrow = int(np.ceil(np.sqrt(pixel_diff_mean.shape[0])))
            visuals['pixel_diff_mean'] = tensor2im(tvutil.make_grid(pixel_diff_mean, nrow=nrow, normalize=True, scale_each=True), renormalize=False)
            visuals['pixel_diff_std'] = tensor2im(tvutil.make_grid(pixel_diff_std, nrow=nrow, normalize=True, scale_each=True), renormalize=False)
            
            losses['pixel_diff_std'] = np.array(torch.mean(pixel_diff_mean).item())
            losses['pixel_diff_mean'] = np.array(torch.mean(pixel_diff_std).item())

        # evaluation the full set
        val_count = 0
        
        for it, data in enumerate(val_data_loader):
            if how_many_to_valid == 0: break
            ## posterior
            self.set_input(data)
            self.test() # Weird. using forward will cause a mem leak
            # only evaluate the real part if has two channels
            losses['reconst_loss'].append(float(F.mse_loss(self.fake_B[:,:2,...].norm(dim=1,keepdim=True), self.real_B[:,:2,...].norm(dim=1,keepdim=True), size_average=True)))
            losses['reconst_ssim'].append(float(util.ssim_metric(self.fake_B[:,:2,...].norm(dim=1,keepdim=True), self.real_B[:,:2,...].norm(dim=1,keepdim=True))))

            if need_sampling:
                if hasattr(self, 'compute_logistic'):
                    # compute at posterior
                    bits_per_dim = self.compute_logistic()
                    losses['bits_per_dim'] = bits_per_dim
                    losses['div'] = self.compute_KL()
            # prior
            if need_sampling:
                self.set_input(data)
                self.test(sampling=True)
                losses['sampling_loss'].append(float(F.mse_loss(self.fake_B[:,:2,...].norm(dim=1,keepdim=True), self.real_B[:,:2,...].norm(dim=1,keepdim=True), size_average=True)))
                losses['sampling_ssim'].append(float(util.ssim_metric(self.fake_B[:,:2,...].norm(dim=1,keepdim=True), self.real_B[:,:2,...].norm(dim=1,keepdim=True))))
            else:
                losses['sampling_loss'] = losses['reconst_loss']

            # compute at prior
            fft_vis, fft_inv = self.compute_special_losses()
            losses['FFTVisiable_loss'].append(fft_vis)
            losses['FFTInvisiable_loss'].append(fft_inv)

            # for GAN based method we track disc score of visible and invisible part
            if hasattr(self,'forward_D'):
                (vs, ivs), _ = self.forward_D()
                losses['visible_disc'].append(vs)
                losses['invisible_disc'].append(ivs)
            
            sys.stdout.write('\r validation (%d/%d) [rec loss: %.5f smp loss: %.5f]' % (val_count, how_many_to_valid, np.mean(losses['reconst_loss']), np.mean(losses['sampling_loss'])))
            sys.stdout.flush()

            val_count += self.fake_B.shape[0]
            if val_count >= how_many_to_valid: break

        if how_many_to_valid > 0:
            print(' ')
            for k, v in losses.items():
                if k in ('sampling_loss', 'reconst_loss'):
                    nan_n = len([a for a in v if str(a) == 'nan'])
                    if nan_n < (0.1 * len(v)):
                        v = [a for a in v if str(a) != 'nan']
                losses[k] = np.mean(v)
                print('\t {} (E[error]): {} '.format(k, losses[k]))

        self.validation_phase = False

        return visuals, losses
