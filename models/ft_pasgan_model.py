from .base_model import BaseModel
from . import networks
from .fft_utils import *
from torch.autograd import Variable
from torch import nn

from .ft_recurnnv2_model import AUTOPARAM


class kspaceMap(nn.Module):
    def __init__(self, imSize=128, no_embed=False):
        super(kspaceMap, self).__init__()
        
        self.RFFT = RFFT()
        self.IFFT = IFFT()
        self.imSize = imSize
        # seperate image to spectral maps
        self.register_buffer('seperate_mask', torch.FloatTensor(1,imSize,1,1,imSize))
        self.seperate_mask.fill_(0)
        for i in range(imSize):
            self.seperate_mask[0,i,0,0,i] = 1

        self.no_embed = no_embed
        if not no_embed:
            self.embed = nn.Sequential(
                            nn.Conv2d(imSize, imSize, 1, 1, padding=0, bias=False),
                            nn.LeakyReLU(0.2,True)
            )
        else:
            print(f'[kspaceMap] -> do not use kspace embedding')

    def forward(self, input, mask):
        # we assume the input only has real part of images (if they are obtained from IFFT)
        bz = input.shape[0]
        x = input[:,:1,:,:]
        if input.shape[1] > 1:
            others = input[:,1:,:,:]
        kspace = self.RFFT(x)
        
        kspace = kspace.unsqueeze(1).repeat(1,self.imSize,1,1,1) # [B,imsize,2,imsize,imsize]
        masked_kspace = self.seperate_mask * kspace
        masked_kspace = masked_kspace.view(bz*self.imSize, 2, self.imSize, self.imSize)
        # discard the imaginary part [B,imsize,imsize, imsize]
        seperate_imgs = self.IFFT(masked_kspace)[:,0,:,:].view(bz, self.imSize, self.imSize, self.imSize)

        return torch.cat([seperate_imgs, others], 1)


class FTPASGANModel(BaseModel):
    def name(self):
        return 'FTPASGANModel'

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        
        if is_train:
            parser.add_argument('--lambda_gan', type=float, default=0.01, help='weight for rec loss')
            parser.add_argument('--use_allgen_for_disc', action='store_true',
                                help='use all intermediate outputs from generators to train D')

        parser.add_argument('--loss_type', type=str, default='MSE', choices=['MSE','L1'], help=' loss type')
        parser.add_argument('--betas', type=str, default='0,0,0', help=' beta of sparsity loss')
        parser.add_argument('--sparsity_norm', type=float, default=0,
                            help='norm of sparsity loss. It should be smaller than 1')
        parser.add_argument('--where_loss_weight', type=str, choices=['full','logvar'], default='full',
                            help='the type of learnable W to apply to')
        parser.add_argument('--scale_logvar_each', action='store_true', help='scale logvar map each')
        parser.add_argument('--clip_weight', type=float, default=0,
                            help='clip loss weight to prevent it to too small value')
        
        parser.add_argument('--no_init_kspacemap_embed', action='store_true',
                            help='do *not* init the weight of kspacemap module')
        parser.add_argument('--no_kspacemap_embed', action='store_true', help='do *not* use embed of kspace embedding')
        parser.add_argument('--use_fixed_weight', type=str, default='1,1,1', help='use fixed weight for all loss')
        parser.add_argument('--mask_cond', action='store_true', help='condition on mask')
        parser.add_argument('--use_mse_as_disc_energy', action='store_true', help='use mse as disc energy')

        if not is_train:
            parser.add_argument('--no_lsgan', action='store_true',
                                help='do *not* use least square GAN, if false, use vanilla GAN')
        
        parser.add_argument('--set_sampling_at_stage', type=int, default=None, help='sampling from the model')
        parser.add_argument('--grad_ctx', action='store_true',
                            help='gan criterion computes adversarial loss signal at provided kspace lines')
        parser.add_argument('--pixelwise_loss_merge', action='store_true', help='no uncertainty analysis')

        parser.set_defaults(pool_size=0)
        parser.set_defaults(which_model_netG='pasnet')
        parser.set_defaults(input_nc=2)
        parser.set_defaults(output_nc=3)
        parser.set_defaults(niter=50)
        parser.set_defaults(niter_decay=50)
        parser.set_defaults(print_freq=100)
        parser.set_defaults(norm='instance')
        parser.set_defaults(dynamic_mask_type='random_zz')
        parser.set_defaults(no_dropout=True)
        parser.set_defaults(loadSize=144)
        parser.set_defaults(fineSize=128)
        parser.set_defaults(which_model_netD='n_layers_channel')
        parser.set_defaults(n_layers_D=4)
        parser.set_defaults(ndf=256)
        parser.set_defaults(no_lsgan=False)
        parser.set_defaults(no_kspacemap_embed=True)
        parser.set_defaults(mask_cond=True)
        parser.set_defaults(grad_ctx=True)
        parser.set_defaults(pixelwise_loss_merge=True)

        return parser

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.isTrain = opt.isTrain
        # specify the training losses you want to print out. The program will call base_model.get_current_losses
        self.loss_names = ['G', 'FFTVisiable', 'FFTInvisiable', 'l2', 'uncertainty', 'sparsity', 
                            'GW_0', 'GW_1', 'GW_2', 'G_GAN', 'G_all', 'D', 'visible_disc', 'invisiable_disc',
                           'gradOutputD']
        # specify the images you want to save/display. The program will call base_model.get_current_visuals
        self.visual_names = ['real_A', 'fake_B', 'real_B']
        # specify the models you want to save to the disk. The program will call
        # base_model.save_networks and base_model.load_networks
        self.model_names = ['G','D']

        assert opt.clip_weight == 0
        opt.use_fixed_weight = [float(a) for a in opt.use_fixed_weight.split(',')]

        self.num_stage = 3 # number fo stage in pasnet
        assert(opt.output_nc == 3)
        # load/define networks
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf,
                                    opt.which_model_netG, opt.norm, not opt.no_dropout, opt.init_type, self.gpu_ids, no_last_tanh=True)
        self.netP = AUTOPARAM(length=self.num_stage, fixed_ws=opt.use_fixed_weight).to(self.device)
        # self.mask = create_mask(opt.fineSize, mask_fraction=self.opt.kspace_keep_ratio).to(self.device)
        self.RFFT = RFFT().to(self.device)
        self.IFFT = IFFT().to(self.device)
        # for evaluation
        self.FFT = FFT().to(self.device)

        if self.isTrain or 'D' in self.model_names:
            use_sigmoid = opt.no_lsgan
            self.cond_input_D = False
            if opt.which_model_netD == 'n_layers_channel':
                pre_process = kspaceMap(imSize=opt.fineSize, no_embed=opt.no_kspacemap_embed).to(self.device)
                d_in_nc = opt.fineSize + (6 if self.opt.mask_cond else 0)
                self.netD = networks.define_D(d_in_nc, opt.ndf,
                                            opt.which_model_netD, opt.n_layers_D, opt.norm, use_sigmoid, 
                                            opt.init_type, self.gpu_ids, preprocess_module=pre_process)

        if self.isTrain:
            # self.fake_AB_pool = ImagePool(opt.pool_size)
            # define loss functions
            self.criterionGAN = networks.GANLossKspace(use_lsgan=not opt.no_lsgan,
                                                       use_mse_as_energy=opt.use_mse_as_disc_energy,
                                                       grad_ctx=self.opt.grad_ctx).to(self.device)

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

        self.zscore = 3

        self.betas = [float(a) for a in self.opt.betas.split(',')]
        assert len(self.betas) == self.num_stage, 'beta length is euqal to the module #'
