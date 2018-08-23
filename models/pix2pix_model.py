import torch
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
from util import util
import torchvision.utils as tvutil
import torch
import torch.nn.functional as F
from .fft_utils import *

class Pix2PixModel(BaseModel):
    def name(self):
        return 'Pix2PixModel'

    @staticmethod
    def modify_commandline_options(parser, is_train=True):

        # changing the default values to match the pix2pix paper
        # (https://phillipi.github.io/pix2pix/)
        parser.set_defaults(pool_size=0)
        parser.set_defaults(no_lsgan=True)
        parser.set_defaults(dataset_mode='aligned')
        parser.set_defaults(which_model_netG='unet_128')
        if is_train:
            parser.add_argument('--lambda_L1', type=float, default=100.0, help='weight for L1 loss')
        
        return parser

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.isTrain = opt.isTrain
        # specify the training losses you want to print out. The program will call base_model.get_current_losses
        self.loss_names = ['G_GAN', 'G_L1', 'D_real', 'D_fake', 'VGG']
        # specify the images you want to save/display. The program will call base_model.get_current_visuals
        self.visual_names = ['real_A', 'fake_B', 'real_B']
        # specify the models you want to save to the disk. The program will call base_model.save_networks and base_model.load_networks
        if self.isTrain:
            self.model_names = ['G', 'D']
        else:  # during test time, only load Gs
            self.model_names = ['G']
        # load/define networks
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf,
                                      opt.which_model_netG, opt.norm, not opt.no_dropout, opt.init_type, self.gpu_ids, no_last_tanh=True)
        if self.isTrain:
            use_sigmoid = opt.no_lsgan
            input_nc_D = 2
            self.netD = networks.define_D(input_nc_D, opt.ndf,
                                          opt.which_model_netD,
                                          opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, self.gpu_ids)
        self.RFFT = RFFT().to(self.device)
        self.mask = create_mask(opt.fineSize, mask_fraction=self.opt.kspace_keep_ratio).to(self.device)
        self.IFFT = IFFT().to(self.device)
        # for evaluation
        self.FFT = FFT().to(self.device)

        if self.isTrain:
            self.fake_AB_pool = ImagePool(opt.pool_size)
            # define loss functions
            self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan).to(self.device)
            self.criterionL1 = torch.nn.L1Loss()

            # initialize optimizers
            self.optimizers = []
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

            if opt.lambda_vgg > 0:
                #TODO VGG actually expect normalization (see torchvision) but we did not do here
                self.criterionPerceptualLoss = networks.VGGLoss(opt.gpu_ids, input_channel=opt.input_nc)

        self.zscore = 3
        if self.opt.output_nc == 2:
            # the imagnary part of reconstrued data
            self.imag_gt = torch.cuda.FloatTensor(opt.batchSize, 1, opt.fineSize, opt.fineSize)

    def set_input(self, input):
        if self.mri_data:
            if len(input) == 4:
                input = input[1:]
            self.set_input2(input)
        else:
            self.set_input1(input)
        
    def forward(self):
        h, b = self.mask.shape[2], self.real_A.shape[0]
        mask = self.mask.view(self.mask.shape[0],1,h,1).expand(b,1,h,1)
        
        if 'residual' in self.opt.which_model_netG:
            self.fake_B = self.netG(self.real_A, mask)
        else:
            self.fake_B = self.netG(self.real_A)

    def backward_D(self):
        # Fake
        # stop backprop to the generator by detaching fake_B
        fake_AB = self.create_D_input(self.fake_B)
        pred_fake = self.netD(fake_AB.detach()) # 14x14 for 128x128
        self.loss_D_fake = self.criterionGAN(pred_fake, False)

        # Real
        real_AB = self.create_D_input(self.real_B) 
        pred_real = self.netD(real_AB)
        self.loss_D_real = self.criterionGAN(pred_real, True)

        # Combined loss
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5

        self.loss_D.backward()

    def create_D_input(self, fake_B):
        fake_B = fake_B[:,:1,:,:] # discard imaginary part
        fake_B = self._clamp(fake_B)
        fake = torch.cat([fake_B, self.real_A[:,:1,:,:]], dim=1)
        return fake

    def _clamp(self, data):
        # process data for D input
        # make consistent range with real_B for inputs of D
        assert self.mri_data
        if self.mri_data:
            if self.zscore != 0:
                data = data.clamp(-self.zscore, self.zscore) 
        else:
            data = data.clamp(-1, 1) 
        return data

    def backward_G(self):
        # First, G(A) should fake the discriminator
        fake_AB = self.create_D_input(self.fake_B)
        pred_fake = self.netD(fake_AB)
        self.loss_G_GAN = self.criterionGAN(pred_fake, True)

        # Second, G(A) = B
        self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B) * self.opt.lambda_L1

        # third
        if hasattr(self, 'criterionPerceptualLoss'):
            self.loss_VGG = self.criterionPerceptualLoss(self.fake_B, self.real_B) * self.opt.lambda_vgg
            self.loss_G = self.loss_G_GAN + self.loss_G_L1 + self.loss_VGG 
        else:
            self.loss_G = self.loss_G_GAN + self.loss_G_L1

        self.loss_G.backward()

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
