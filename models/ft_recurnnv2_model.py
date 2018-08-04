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

class AUTOPARAM(nn.Module):
    def __init__(self, length=3):
        super(AUTOPARAM, self).__init__()
        self.weight = Parameter(torch.FloatTensor(length).fill_(1))

    def forward(self):
        weight = F.softmax(self.weight) * self.weight.shape[0] # sum to be one

        return weight

class FTRECURNNV2Model(BaseModel):
    def name(self):
        return 'FTRECURNNV2Model'

    @staticmethod
    def modify_commandline_options(parser, is_train=True):

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

        if is_train:
            parser.add_argument('--lambda', type=float, default=100.0, help='weight for rec loss')
        parser.add_argument('--loss_type', type=str, default='MSE', choices=['MSE','L1'], help=' loss type')
        parser.add_argument('--betas', type=str, default='0,0.5,1', help=' beta of sparsity loss')
        parser.add_argument('--sparsity_norm', type=float, default=0, help='norm of sparsity loss. It should be smaller than 1')
        parser.add_argument('--use_learnable_W', type=str, choices=['full','logvar'], default='full', help='the type of learnable W to apply to')
        parser.add_argument('--scale_logvar_each', action='store_true', help='scale logvar map each')

        return parser

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.isTrain = opt.isTrain
        # specify the training losses you want to print out. The program will call base_model.get_current_losses
        self.loss_names = ['G', 'FFTVisiable', 'FFTInvisiable', 'l2', 'uncertainty', 'sparsity', 'GW_0', 'GW_1', 'GW_2']
        # specify the images you want to save/display. The program will call base_model.get_current_visuals
        self.visual_names = ['real_A', 'fake_B', 'real_B']
        # specify the models you want to save to the disk. The program will call base_model.save_networks and base_model.load_networks
        if self.isTrain:
            self.model_names = ['G']
        else:  # during test time, only load Gs
            self.model_names = ['G']
        assert(opt.output_nc == 3)
        # load/define networks
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf,
                                    opt.which_model_netG, opt.norm, not opt.no_dropout, opt.init_type, self.gpu_ids, no_last_tanh=True)
        self.netP = AUTOPARAM(length=3).to(self.device)
        self.mask = create_mask(opt.fineSize, mask_fraction=self.opt.kspace_keep_ratio).to(self.device)
        self.RFFT = RFFT().to(self.device)
        self.IFFT = IFFT().to(self.device)
        # for evaluation
        self.FFT = FFT().to(self.device)
        if self.isTrain:
            # define loss functions
            if opt.loss_type == 'MSE':
                self.criterion = torch.nn.MSELoss(reduce=False) 
            elif opt.loss_type == 'L1':
                self.criterion = torch.nn.L1Loss(reduce=False) 
            # initialize optimizers
            self.optimizers = []
            self.optimizer_G = torch.optim.Adam(list(self.netG.parameters()) + list(self.netP.parameters()),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))

            self.optimizers.append(self.optimizer_G)

        if self.opt.output_nc >= 2:
            # the imagnary part of reconstrued data
            self.imag_gt = torch.cuda.FloatTensor(opt.batchSize, 1, opt.fineSize, opt.fineSize)

        self.betas = [float(a) for a in self.opt.betas.split(',')]
        assert len(self.betas) == 3, 'beta length is euqal to the module #'

    def certainty_loss(self, fake_B, real_B, logvar, beta, stage, weight_logvar, weight_all):
        
        o = int(np.floor(self.opt.output_nc/2))
        # l2 reconstruction loss
        l2 = self.criterion(fake_B[:,:o,:,:], real_B[:,:o,:,:]) 
        one_over_var = torch.exp(-logvar)
        # uncertainty loss
        assert len(l2) == len(logvar)
        loss = 0.5 * (one_over_var * l2 + logvar * weight_logvar)
        loss = loss.mean()

        # l0 sparsity distance
        if beta != 0:
            b,c,h,w = logvar.shape
            k = b*c*h*w
            sparsity = logvar.norm(self.opt.sparsity_norm).div(k) * beta 
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
        self.loss_G = 0
        for stage, (fake_B, logvar, beta) in enumerate(zip(self.fake_Bs, self.logvars, self.betas)):
            if self.opt.use_learnable_W == 'full':
                w_full = weights[stage]
                w_lv = 1
            elif self.opt.use_learnable_W == 'logvar':
                w_lv = weights[stage]
                w_full = 1
            else:
                w_full = w_lv = 1
            self.loss_G += self.certainty_loss(fake_B, self.real_B, logvar, beta, stage, 
                                                weight_logvar=w_lv, weight_all=w_full) 
                            
        self.loss_G.backward()

        self.compute_special_losses()

        for i in range(weights.shape[0]):
            setattr(self, f'loss_GW_{i}', float(weights[i].item()))

    def optimize_parameters(self):
        self.forward()

        # update G
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()

