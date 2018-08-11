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
from torch.nn.parameter import Parameter

from .ft_recurnnv2_model import AUTOPARAM
import math

class kspaceMap(nn.Module):
    def __init__(self, imSize=128, no_embed=False):
        super(kspaceMap, self).__init__()
        
        self.RFFT = RFFT()
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
        x = input[:,:1,:,:]
        
        if x.shape[1] > 1:
            others = input[:,1:,:,:]
        kspace = self.RFFT(x)
        
        kspace = kspace.unsqueeze(1).repeat(1,self.imSize,1,1,1) # [B,imsize,2,imsize,imsize]
        masked_kspace = self.seperate_mask * kspace
        masked_kspace = masked_kspace.view(bz*self.imSize, 2, self.imSize, self.imSize)
        seperate_imgs = self.IFFT(masked_kspace)[:,0,:,:].view(bz, self.imSize, self.imSize, self.imSize) # discard the imaginary part [B,imsize,imsize, imsize]

        if not self.no_embed:
            seperate_imgs = self.embed(seperate_imgs)

        if x.shape[1] > 1:
            output = torch.cat([seperate_imgs, others], 1)
        else:
            output = seperate_imgs

        return output

    def _deprecated_forward(self, input, mask):
        # mask [B,1,imsize,1]
        x = input[:,:1,:,:]
        
        if x.shape[1] > 1:
            others = input[:,1:,:,:]
        kspace = self.RFFT(x)
        
        bz = mask.shape[0]

        if not hasattr(self, 'seperate_mask') or self.seperate_mask.shape[0] != bz:
            self.seperate_mask = torch.zeros_like(mask).unsqueeze(1).expand(bz,self.imSize,1,self.imSize,1)
            
        ## seperate an image to N images N, each for a kspace line
        if mask is None:
            self.seperate_mask.fill_(1)
        else:
            self.seperate_mask.fill_(0)
            for i in range(bz):
                idx = torch.nonzero(mask[i,0,:,0])
                self.seperate_mask[i,idx,:,idx,:] = 1 # [1,imsize,1,imsize,1]

        kspace = kspace.view(bz,1,2,self.imSize,self.imSize).repeat(1,self.imSize,1,1,1) # [B,imsize,2,imsize,imsize]
        masked_kspace = self.seperate_mask * kspace
        masked_kspace = masked_kspace.view(bz*self.imSize, 2, self.imSize, self.imSize)
        seperate_imgs = self.IFFT(masked_kspace)[:,0,:,:].view(bz, self.imSize, self.imSize, self.imSize) #[B,imsize,imsize, imsize]

        if not self.no_embed:
            seperate_imgs = self.embed(seperate_imgs)

        if x.shape[1] > 1:
            output = torch.cat([seperate_imgs, others], 1)
        else:
            output = seperate_imgs

        return output


class FTRECURGANModel(BaseModel):
    def name(self):
        return 'FTRECURGANModel'

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
        parser.add_argument('--use_fixed_weight', type=str, default='1,1,1', help='do *not* use embed of kspace embedding')

        if not is_train:
            parser.add_argument('--no_lsgan', action='store_true', help='do *not* use least square GAN, if false, use vanilla GAN')
            
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
            pre_process = kspaceMap(imSize=opt.fineSize, no_embed=opt.no_kspacemap_embed).to(self.device)
            self.netD = networks.define_D(opt.fineSize, opt.ndf,
                                          opt.which_model_netD, opt.n_layers_D, opt.norm, use_sigmoid, 
                                          opt.init_type, self.gpu_ids, preprocess_module=pre_process)
            # initlize the netD first layer parameter to be the sum of all channels
            if not opt.no_kspacemap_embed and not opt.no_init_kspacemap_embed:
                pre_process.init_weight()

        if self.isTrain:
            self.fake_AB_pool = ImagePool(opt.pool_size)
            # define loss functions
            if 'aux' in opt.which_model_netD:
                self.criterionGAN = networks.GANLossKspaceAux(use_lsgan=not opt.no_lsgan).to(self.device)
            else:
                self.criterionGAN = networks.GANLossKspace(use_lsgan=not opt.no_lsgan).to(self.device)

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
        
        self.betas = [float(a) for a in self.opt.betas.split(',')]
        assert len(self.betas) == self.num_stage, 'beta length is euqal to the module #'
        
    def certainty_loss(self, fake_B, real_B, logvar, beta, stage, weight_logvar, weight_all):
        
        o = int(np.floor(self.opt.output_nc/2))
        # l2 reconstruction loss
        l2 = self.criterion(fake_B[:,:o,:,:], real_B[:,:o,:,:]) 
        # uncertainty simplified gaussian nll
        one_over_var = torch.exp(-logvar)
        assert len(l2) == len(logvar)
        loss = 0.5 * (one_over_var * l2 + logvar * weight_logvar)
        loss = loss.mean()
        
        # l0 sparsity distance
        if beta != 0:
            b,c,h,w = logvar.shape
            k = b*c*h*w
            var = logvar.exp()
            # var[var < 0.1] = 0
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
            self.set_input2(input)
        else:
            self.set_input1(input)

    def forward(self):
        # conditioned on mask
        h, b = self.mask.shape[2], self.real_A.shape[0]
        mask = Variable(self.mask.view(self.mask.shape[0],1,h,1).expand(b,1,h,1))
        self.fake_Bs, self.logvars, mask_cond = self.netG(self.real_A, mask)

        self.fake_B = self.fake_Bs[-1]

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
        # gan loss
        fake_AB = self._clamp(self.fake_B) 
        # track norm of gradient
        gradnorm_fakeAB = torch.cuda.FloatTensor(1)
        def extract(grad):
            global gradnorm_fakeAB
            gradnorm_fakeAB = torch.abs(grad).norm(2)
        fake_AB.register_hook(extract)
        
        pred_fake = self.netD(fake_AB, self.mask)
        self.loss_G_GAN = self.criterionGAN(pred_fake, True, self.mask, degree=1, updateG=True) * self.opt.lambda_gan

        self.loss_G_all = self.loss_G_GAN + self.loss_G
        self.loss_G_all.backward()

        # compute multiple loss
        self.compute_special_losses()
        self.loss_gradOutputD = float(gradnorm_fakeAB.item())
        for i in range(weights.shape[0]):
            setattr(self, f'loss_GW_{i}', float(weights[i].item()))

    def _clamp(self, data):
        # process data for D input
        data = data[:,:1,:,:] # discard imaginary part
        # make consistent range with real_B for inputs of D
        if self.mri_data:
            zscore = 3
            data = data.clamp(-zscore, zscore) 
        else:
            data = data.clamp(-1, 1) 
        return data
    
    def forward_D(self):
        # to get the disc score of visiable and invisiable parts during testing
        with torch.no_grad():
            fake_AB = self._clamp(self.fake_B)
            pred_fake = self.netD(fake_AB, self.mask) 
        return self.mask_disc_score(pred_fake, self.mask.squeeze())

    def mask_disc_score(self, pred, mask):
        mask = mask.mul(1) # copy mask
        if isinstance(pred, tuple): 
            pred = pred[1]
        # pred, mask [B,H]
        self.loss_invisiable_disc = (pred * (1-mask)).sum().div_((1-mask).sum())
        # the lowfreq part is always zero we do not want to monitor it and affact the average
        mask[:,:3] = 0
        mask[:,-3:] = 0
        self.loss_visible_disc = (pred * mask).sum().div_(mask.sum())
        
        return float(self.loss_visible_disc.item()), float(self.loss_invisiable_disc.item())

    def backward_D(self):
        # Fake
        # stop backprop to the generator by detaching fake_B
        fake_AB = self.fake_AB_pool.query(self._clamp(self.fake_B))
        pred_fake = self.netD(fake_AB.detach(), self.mask) 
        degree = 0 if not self.opt.use_allgen_for_disc else 0.2
        self.loss_D_fake = self.criterionGAN(pred_fake, False, self.mask, degree=degree)
        self.mask_disc_score(pred_fake, self.mask.squeeze())

        if self.opt.use_allgen_for_disc:
            pred_fake = self.netD(self._clamp(self.fake_Bs[-2]).detach(), self.mask) 
            self.loss_D_fake += self.criterionGAN(pred_fake, False, self.mask, degree=0.1)
            pred_fake = self.netD(self._clamp(self.fake_Bs[-3]).detach(), self.mask) 
            self.loss_D_fake += self.criterionGAN(pred_fake, False, self.mask, degree=0.0)
            self.loss_D_fake = self.loss_D_fake / self.num_stage

        # Real
        real_AB = self._clamp(self.real_B)
        pred_real = self.netD(real_AB, self.mask)
        self.loss_D_real = self.criterionGAN(pred_real, True, self.mask, degree=1)

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

