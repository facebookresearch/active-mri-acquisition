import torch
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
from util import util
import torchvision.utils as tvutil
import torch

class SelfCyPix2PixModel(BaseModel):
    def name(self):
        return 'SelfCyPix2PixModel'

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
                                      opt.which_model_netG, opt.norm, not opt.no_dropout, opt.init_type, self.gpu_ids)

        if self.isTrain:
            use_sigmoid = opt.no_lsgan
            self.netD = networks.define_D(opt.output_nc, opt.ndf,
                                          opt.which_model_netD,
                                          opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, self.gpu_ids)

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

    def set_input(self, input):
        # output from FT loader
        AtoB = self.opt.which_direction == 'AtoB'
        img, _, context = input
        context = context[:, :1,:,:]
        if AtoB:
            self.real_A = context.to(self.device)
            self.real_B = img.to(self.device)
        else:
            self.real_A = img.to(self.device)
            self.real_B = context.to(self.device)
        

    def forward(self):
        self.fake_B = self.netG(self.real_A)
        if self.netG.training:
            # we expect the self consistency in model
            self.rect_B = self.netG(self.real_B)
    
    def backward_D(self):
        # Fake
        # stop backprop to the generator by detaching fake_B
        fake_B = self.fake_AB_pool.query(self.fake_B)
        pred_fake = self.netD(fake_B.detach()) # 14x14 for 128x128
        self.loss_D_fake = self.criterionGAN(pred_fake, False)

        # Real
        pred_real = self.netD(self.real_B)
        self.loss_D_real = self.criterionGAN(pred_real, True)

        # Combined loss
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5

        self.loss_D.backward()

    def backward_G(self):
        # First, G(A) should fake the discriminator
        pred_fake = self.netD(self.fake_B)
        self.loss_G_GAN = self.criterionGAN(pred_fake, True)

        # Second, G(A) = B
        self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B) * self.opt.lambda_L1 + \
                         self.criterionL1(self.rect_B, self.real_B) * self.opt.lambda_L1 #  self-consistency loss

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

    def validation(self, val_data_loader, how_many_to_display=64, how_many_to_valid=4096):
        val_data = []
        val_count = 0
        reconst_loss = 0
        self.netG.eval()
        for it, data in enumerate(val_data_loader):
            self.set_input(data)
            self.test() # Weird. using forward will cause a mem leak
            c = min(self.fake_B.shape[0], how_many_to_display)
            real_A, fake_B, real_B, = self.real_A[:c,...].cpu(), self.fake_B[:c,...].cpu(), self.real_B[:c,...].cpu()
            if val_count < how_many_to_display:
                val_data.append([real_A[:c,...], fake_B[:c,...], real_B[:c,...]])
            val_count += self.fake_B.shape[0]
            
            reconst_loss += float(torch.pow((fake_B - real_B), 2).sum())
            if val_count >= how_many_to_valid: break
        b,c,h,w = self.real_B.shape
        reconst_loss /= (val_count*h*w*c)

        visuals = {}
        visuals['inputs'] = util.tensor2im(tvutil.make_grid(torch.cat([a[0] for a in val_data], dim=0)))
        visuals['reconstructions'] = util.tensor2im(tvutil.make_grid(torch.cat([a[1] for a in val_data], dim=0)))
        visuals['groundtruths'] = util.tensor2im(tvutil.make_grid(torch.cat([a[2] for a in val_data], dim=0)))  

        print('\t Reconstruction loss: ', reconst_loss)

        self.netG.train()

        return visuals, {'reconst_loss': reconst_loss}