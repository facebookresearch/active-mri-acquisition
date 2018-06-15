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
from .networks import init_net
from .fft_utils import *

class Push(nn.Module):
    vars = {}
    def __init__(self, name):
        super(Push, self).__init__()
        self.name = name

    def forward(self, x):
        Push.vars[self.name] = x
        return x

    def __repr__(self):
        return 'Push({})'.format(self.name)

class Pop(nn.Module):
    def __init__(self, name):
        super(Pop, self).__init__()
        self.name = name

    def forward(self, x):
        y = Push.vars.pop(self.name)
        return torch.cat((x, y), 1)

    def __repr__(self):
        return 'Pop({})'.format(self.name)

def unet_layers(fm_in):
    fm = 64
    conv = lambda fm_in, fm_out, stride=2: nn.Conv2d(fm_in, fm_out, 4, stride, 1)
    convT = lambda fm_in, fm_out: nn.ConvTranspose2d(fm_in, fm_out, 4, 2, 1)
    return [
        conv(fm_in, fm),                           Push(1),
        nn.LeakyReLU(0.2, True), conv(fm*1, fm*2), Push(2),
        nn.LeakyReLU(0.2, True), conv(fm*2, fm*4), Push(3),
        nn.LeakyReLU(0.2, True), conv(fm*4, fm*8), Push(4),
        nn.LeakyReLU(0.2, True), conv(fm*8, fm*8), Push(5),
        nn.LeakyReLU(0.2, True), conv(fm*8, fm*8), Push(6),
        nn.LeakyReLU(0.2, True), conv(fm*8, fm*8),
        nn.ReLU(True), convT(fm*8*1, fm*8), Pop(6),
        nn.ReLU(True), convT(fm*8*2, fm*8), Pop(5),
        nn.ReLU(True), convT(fm*8*2, fm*8), Pop(4),
        nn.ReLU(True), convT(fm*8*2, fm*4), Pop(3),
        nn.ReLU(True), convT(fm*4*2, fm*2), Pop(2),
        nn.ReLU(True), convT(fm*2*2, fm*1), Pop(1),
        nn.ReLU(True), convT(fm*2*1, 1)]

class FTCNNModel(BaseModel):
    def name(self):
        return 'FTCNNModel'

    @staticmethod
    def modify_commandline_options(parser, is_train=True):

        # changing the default values to match the pix2pix paper
        # (https://phillipi.github.io/pix2pix/)
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
        self.loss_names = ['G']
        # specify the images you want to save/display. The program will call base_model.get_current_visuals
        self.visual_names = ['real_A', 'fake_B', 'real_B']
        # specify the models you want to save to the disk. The program will call base_model.save_networks and base_model.load_networks
        if self.isTrain:
            self.model_names = ['G']
        else:  # during test time, only load Gs
            self.model_names = ['G']
        # load/define networks
        if opt.which_model_netG == 'jure_unet':
            model = nn.Sequential(*unet_layers(2))
        else:
            model = networks.define_G(2, opt.output_nc, opt.ngf,
                                      opt.which_model_netG, opt.norm, not opt.no_dropout, opt.init_type, self.gpu_ids)

        self.netG = init_net( nn.Sequential(*unet_layers(2)), opt.init_type, self.gpu_ids )
        self.FFT = RFFT().to(self.device)
        self.mask = create_mask(opt.fineSize).to(self.device)
        self.IFFT = IFFT().to(self.device)

        if self.isTrain:
            # define loss functions
            self.criterion = torch.nn.MSELoss()

            # initialize optimizers
            self.optimizers = []
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))

            self.optimizers.append(self.optimizer_G)

    def set_input(self, input):
        # output from FT loader
        AtoB = self.opt.which_direction == 'AtoB'
        img, _, _ = input
        img = img.to(self.device)
        # doing FFT
        fft_kspace = self.FFT(img)
        masked_fft_kspace = (fft_kspace * self.mask)
        
        context = self.IFFT(masked_fft_kspace)

        if AtoB:
            self.real_A = context
            self.real_B = img
        else:
            self.real_A = img
            self.real_B = context
        

    def forward(self):
        self.fake_B = self.netG(self.real_A)    

    def backward_G(self):
        # First, G(A) should fake the discriminator

        self.loss_G  = self.criterion(self.fake_B, self.real_B) 
        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()

        # update G
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()

