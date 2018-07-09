import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.optim import lr_scheduler
import torch.nn.functional as F

###############################################################################
# Helper Functions
###############################################################################


def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=True)
    elif norm_type == 'none':
        norm_layer = None
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def get_scheduler(optimizer, opt):
    if opt.lr_policy == 'lambda':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + 1 + opt.epoch_count - opt.niter) / float(opt.niter_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler

def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)


def init_net(net, init_type='normal', gpu_ids=[]):
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)
    init_weights(net, init_type)
    return net

    
def define_G(input_nc, output_nc, ngf, which_model_netG, norm='batch', use_dropout=False, init_type='normal', gpu_ids=[], no_last_tanh=False):
    netG = None
    norm_layer = get_norm_layer(norm_type=norm)
    
    if which_model_netG == 'resnet_9blocks':
        netG = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=9)
    elif which_model_netG == 'resnet_9blocks_zz':
        netG = ResnetGenerator2(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=9, no_last_tanh=no_last_tanh)
    elif which_model_netG == 'resnet_6blocks':
        netG = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=6)
    elif which_model_netG == 'unet_128':
        netG = UnetGenerator(input_nc, output_nc, 7, ngf, norm_layer=norm_layer, use_dropout=use_dropout, no_last_tanh=no_last_tanh)
    elif which_model_netG == 'unet_256':
        netG = UnetGenerator(input_nc, output_nc, 8, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    elif which_model_netG == 'unet_128_residual':
        _netG = UnetGenerator(input_nc, output_nc, 7, ngf, norm_layer=norm_layer, use_dropout=use_dropout, no_last_tanh=True)
        netG = ResidualNetWrapper(_netG, no_last_tanh=no_last_tanh, output_nc=output_nc)
    elif which_model_netG == 'resnet_9blocks_residual':
        # _netG = ResnetGenerator2(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=9, no_last_tanh=True, n_downsampling=3)
        ## v3 performs better
        _netG = ResnetGenerator3(input_nc, output_nc, ngf, norm_layer=norm_layer, 
                        use_dropout=use_dropout, n_blocks=9, no_last_tanh=True, n_downsampling=3)
        netG = ResidualNetWrapper(_netG, no_last_tanh=no_last_tanh, output_nc=output_nc)
    elif which_model_netG == 'jure_unet':
        netG = nn.Sequential(*unet_layers(input_nc, output_nc))
    elif which_model_netG == 'jure_unet_residual':
        _netG = nn.Sequential(*unet_layers(input_nc, output_nc))
        netG = ResidualNetWrapper(_netG, no_last_tanh=True, output_nc=output_nc)
    elif which_model_netG == 'jure_unet_vae_residual':
        nz = 8
        nef = 64
        _netG = nn.Sequential(*unet_layers(input_nc, output_nc, to_output=False))
        netG = FuseModel(_netG, nef+nz, output_nc, no_last_tanh=True)
    elif which_model_netG == 'resnet_9blocks_vae_residual':
        nz = 8
        _netG = ResnetGenerator3(input_nc, output_nc, ngf, norm_layer=norm_layer, 
                        use_dropout=use_dropout, n_blocks=9, no_last_tanh=True, n_downsampling=3, to_output=False)
        netG = FuseModel(_netG, ngf+nz, output_nc, no_last_tanh=True)
    elif which_model_netG == 'resnet_9blocks_attention_residual':
        # 3 downsampling is enough
        netG = ResnetGeneratorAttResidual(input_nc, output_nc, ngf, norm_layer=norm_layer, 
                        use_dropout=use_dropout, n_blocks=9, no_last_tanh=no_last_tanh, n_downsampling=3, imgSize=128, mask_cond=True)
    elif which_model_netG == 'resnet_9blocks_attention_residual_psp':
        # 3 downsampling is enough
        netG = ResnetGeneratorAttResidual(input_nc, output_nc, ngf, norm_layer=norm_layer, 
                        use_dropout=use_dropout, n_blocks=9, no_last_tanh=no_last_tanh, n_downsampling=3, imgSize=128, use_psp=True)
    elif which_model_netG == 'resnet_9blocks_pixelattention_residual':
        netG = ResnetGeneratorPixelAttResidual(input_nc, output_nc, ngf, norm_layer=norm_layer, 
                        use_dropout=use_dropout, n_blocks=9, no_last_tanh=no_last_tanh, n_downsampling=3, imgSize=128)
    elif which_model_netG == 'resnet_9blocks_masking_residual':
        # used to compared to resnet_9blocks_attention_residual to see if softattention is useful
        netG = ResnetGeneratorMaskingResidual(input_nc, output_nc, ngf, norm_layer=norm_layer, 
                        use_dropout=use_dropout, n_blocks=9, no_last_tanh=no_last_tanh, n_downsampling=3, imgSize=128)
    elif which_model_netG == 'gcn_6layers':
        # graph convolutional network trial
        netG = FTGCN(nfeat=256, hidden=2048, out_dim=256)
    elif which_model_netG == 'kspace_unet_residual':
        # graph convolutional network trial
        # the input is [B,C,H,1]
        _netG = nn.Sequential(*kspace_unet_layers(input_nc, output_nc, to_output=True))
        netG = ResidualNetWrapper(_netG, no_last_tanh=True, output_nc=output_nc)

    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % which_model_netG)

    return init_net(netG, init_type, gpu_ids)


def define_D(input_nc, ndf, which_model_netD,
             n_layers_D=3, norm='batch', use_sigmoid=False, init_type='normal', gpu_ids=[]):
    netD = None
    norm_layer = get_norm_layer(norm_type=norm)

    if which_model_netD == 'basic':
        netD = NLayerDiscriminator(input_nc, ndf, n_layers=3, norm_layer=norm_layer, use_sigmoid=use_sigmoid)
    elif which_model_netD == 'n_layers':
        netD = NLayerDiscriminator(input_nc, ndf, n_layers_D, norm_layer=norm_layer, use_sigmoid=use_sigmoid)
    elif which_model_netD == 'pixel':
        netD = PixelDiscriminator(input_nc, ndf, norm_layer=norm_layer, use_sigmoid=use_sigmoid)
    else:
        raise NotImplementedError('Discriminator model name [%s] is not recognized' %
                                  which_model_netD)
    return init_net(netD, init_type, gpu_ids)

def get_non_linearity(layer_type='relu'):
    if layer_type == 'relu':
        nl_layer = functools.partial(nn.ReLU, inplace=True)
    elif layer_type == 'lrelu':
        nl_layer = functools.partial(
            nn.LeakyReLU, negative_slope=0.2, inplace=True)
    elif layer_type == 'elu':
        nl_layer = functools.partial(nn.ELU, inplace=True)
    else:
        raise NotImplementedError(
            'nonlinearity activitation [%s] is not found' % layer_type)
    return nl_layer

def define_E(input_nc, output_nc, ndf, which_model_netE,
             norm='batch', nl='lrelu',
             init_type='xavier', gpu_ids=[], vaeLike=True):
    netE = None
    norm_layer = get_norm_layer(norm)
    nl = 'lrelu'  # use leaky relu for E
    nl_layer = get_non_linearity(layer_type=nl)
    # if which_model_netE == 'resnet_128':
    #     netE = E_ResNet(input_nc, output_nc, ndf, n_blocks=4, norm_layer=norm_layer,
    #                     nl_layer=nl_layer, gpu_ids=gpu_ids, vaeLike=vaeLike)
    # elif which_model_netE == 'resnet_256':
    #     netE = E_ResNet(input_nc, output_nc, ndf, n_blocks=5, norm_layer=norm_layer,
    #                     nl_layer=nl_layer, gpu_ids=gpu_ids, vaeLike=vaeLike)
    if which_model_netE == 'conv_128':
        netE = E_NLayers(input_nc, output_nc, ndf, n_layers=4, norm_layer=norm_layer,
                         nl_layer=nl_layer, gpu_ids=gpu_ids, vaeLike=vaeLike)
    elif which_model_netE == 'conv_256':
        netE = E_NLayers(input_nc, output_nc, ndf, n_layers=5, norm_layer=norm_layer,
                         nl_layer=nl_layer, gpu_ids=gpu_ids, vaeLike=vaeLike)
    else:
        raise NotImplementedError(
            'Encoder model name [%s] is not recognized' % which_model_netE)

    return init_net(netE, init_type, gpu_ids)

##############################################################################
# Classes
##############################################################################

class ResidualNetWrapper(nn.Module):
    # use to perform y = f(x) + x
    def __init__(self, net, no_last_tanh, output_nc=1):
        super(ResidualNetWrapper, self).__init__()
        self.no_last_tanh = no_last_tanh
        self.net = net
        self.take_first_channel = output_nc == 1

    def forward(self, x):
        # kspace has two channels input (real and imag.)
        residual = self.net(x)
            
        y = residual + (x[:,:1,:,:] if self.take_first_channel else x)

        if self.no_last_tanh:
            return y, residual
        else:
            return F.tanh(y), residual

class FuseModel(nn.Module):
    # perform y = f(x) + x and adding distribution
    def __init__(self, base_net, n_in, n_out, no_last_tanh):
        super(FuseModel, self).__init__()
        self.no_last_tanh = no_last_tanh
        self.take_first_channel = n_out == 1

        self.base_net = base_net

        self.fuse_net = nn.Sequential(
            nn.Conv2d(n_in, n_in, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_in, n_in//2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_in//2, n_out, 1),
        )

    def forward(self, x, z):
        h = self.base_net(x)
        z_fm = z.view(z.size(0), z.size(1), 1, 1).expand(
                z.size(0), z.size(1), h.size(2), h.size(3))
        residual = self.fuse_net(torch.cat([h, z_fm],1))

        y = residual + (x[:,:1,:,:] if self.take_first_channel else x)

        if self.no_last_tanh:
            return y, residual
        else:
            return F.tanh(y), residual

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

def unet_layers(fm_in, out_dim, to_output=True):
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
        nn.ReLU(True)] + ([convT(fm*2*1, out_dim)] if to_output else [convT(fm*2*1, fm)])

# def kspace_unet_layers(fm_in, out_dim, to_output=True):
#     fm = 256
#     conv = lambda fm_in, fm_out, stride=2: nn.Conv2d(fm_in, fm_out, (4,1), (stride,1), (1,0))
#     convT = lambda fm_in, fm_out: nn.ConvTranspose2d(fm_in, fm_out, (4,1), (2,1), (1,0))
#     return [
#         conv(fm_in, fm),                           Push(1),
#         nn.LeakyReLU(0.2, True), conv(fm*1, fm*2), Push(2),
#         nn.LeakyReLU(0.2, True), conv(fm*2, fm*4), Push(3),
#         nn.LeakyReLU(0.2, True), conv(fm*4, fm*8), Push(4),
#         nn.LeakyReLU(0.2, True), conv(fm*8, fm*8), Push(5),
#         nn.LeakyReLU(0.2, True), conv(fm*8, fm*8), Push(6),
#         nn.LeakyReLU(0.2, True), conv(fm*8, fm*8),
#         nn.ReLU(True), convT(fm*8*1, fm*8), Pop(6),
#         nn.ReLU(True), convT(fm*8*2, fm*8), Pop(5),
#         nn.ReLU(True), convT(fm*8*2, fm*8), Pop(4),
#         nn.ReLU(True), convT(fm*8*2, fm*4), Pop(3),
#         nn.ReLU(True), convT(fm*4*2, fm*2), Pop(2),
#         nn.ReLU(True), convT(fm*2*2, fm*1), Pop(1),
#         nn.ReLU(True)] + ([convT(fm*2*1, out_dim)] if to_output else [convT(fm*2*1, fm)])

def kspace_unet_layers(fm_in, out_dim, to_output=True):
    fm = 512
    conv = lambda fm_in, fm_out, stride=2: nn.Conv2d(fm_in, fm_out, (4,1), (stride,1), (1,0))
    convT = lambda fm_in, fm_out: nn.ConvTranspose2d(fm_in, fm_out, (4,1), (2,1), (1,0))
    return [
        conv(fm_in, fm),                           Push(1),
        nn.LeakyReLU(0.2, True), conv(fm*1, fm*2), Push(2),
        nn.LeakyReLU(0.2, True), conv(fm*2, fm*4), Push(3),
        nn.LeakyReLU(0.2, True), conv(fm*4, fm*4), Push(4),
        nn.LeakyReLU(0.2, True), conv(fm*4, fm*4),
        nn.ReLU(True), convT(fm*4*1, fm*4), Pop(4),
        nn.ReLU(True), convT(fm*4*2, fm*4), Pop(3),
        nn.ReLU(True), convT(fm*4*2, fm*2), Pop(2),
        nn.ReLU(True), convT(fm*2*2, fm*1), Pop(1),
        nn.ReLU(True)] + ([convT(fm*2*1, out_dim)] if to_output else [convT(fm*2*1, fm)])

# def kspace_nn_layers(fm_in, out_dim, to_output=True):
#     fm = 512
#     conv = lambda fm_in, fm_out, stride=2: nn.Conv2d(fm_in, fm_out, (4,1), (stride,1), (1,0))
#     convT = lambda fm_in, fm_out: nn.ConvTranspose2d(fm_in, fm_out, (4,1), (2,1), (1,0))
#     return [
#         conv(fm_in, fm),
#         nn.LeakyReLU(0.2, True), conv(fm*1, fm*2),
#         nn.LeakyReLU(0.2, True), conv(fm*2, fm*4),
#         nn.ReLU(True), convT(fm*4, fm*2),
#         nn.ReLU(True), convT(fm*2, fm),
#         nn.ReLU(True), convT(fm, out_dim),
#     ]

class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0):
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(input)

    def __call__(self, input, target_is_real):
        target_tensor = self.get_target_tensor(input, target_is_real)
        return self.loss(input, target_tensor)

class ResnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6, padding_type='reflect'):
        assert(n_blocks >= 0)
        super(ResnetGenerator, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0,
                           bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2**i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3,
                                stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]

        mult = 2**n_downsampling
        for i in range(n_blocks):
            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]

        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=use_bias),
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True)]
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        return self.model(input)

# zz implementation
# use Upsample rather than TransposeConv
# Remove zero padding 
class ResnetGenerator2(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6, padding_type='reflect', no_last_tanh=False, n_downsampling=3):
        assert(n_blocks >= 0)
        super(ResnetGenerator2, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        self.no_last_tanh = no_last_tanh
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0,
                           bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        for i in range(n_downsampling):
            mult = 2**i
            model += [nn.ReflectionPad2d(1),
                     nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3,
                                stride=2, padding=0, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]

        mult = 2**n_downsampling
        for i in range(n_blocks):
            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]

        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            model += [nn.Upsample(scale_factor=2),
                      nn.ReflectionPad2d(1),
                      nn.Conv2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=1,
                                         padding=0,
                                         bias=use_bias),
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True)]

        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        if not self.no_last_tanh:
            model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        return self.model(input)

# zz implementation
# without 7x7 kernel at top and bottom
class ResnetGenerator3(nn.Module):

    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6, padding_type='reflect', no_last_tanh=False, n_downsampling=3, to_output=True):
        assert(n_blocks >= 0)
        super(ResnetGenerator3, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        self.no_last_tanh = no_last_tanh
        self.to_output = to_output
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [nn.ReflectionPad2d(1),
                     nn.Conv2d(input_nc, ngf*2, kernel_size=3,
                                stride=2, padding=0, bias=use_bias),
                      norm_layer(ngf*2),
                      nn.ReLU(True)]

        for i in range(1, n_downsampling):
            mult = 2**i
            model += [nn.ReflectionPad2d(1),
                     nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3,
                                stride=2, padding=0, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]

        mult = 2**n_downsampling
        for i in range(n_blocks):
            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]

        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            model += [nn.Upsample(scale_factor=2),
                      nn.ReflectionPad2d(1),
                      nn.Conv2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=1,
                                         padding=0,
                                         bias=use_bias),
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True)]

        if self.to_output:
            model += [nn.Conv2d(ngf, output_nc, kernel_size=1, padding=0)]

        if not self.no_last_tanh:
            model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        return self.model(input)

from .fft_utils import *
class ResnetGeneratorAttResidual(nn.Module):
    
    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, 
                n_blocks=6, padding_type='reflect', no_last_tanh=False, n_downsampling=3, imgSize=128, use_psp=False, mask_cond=True):
        assert(n_blocks >= 0)
        super(ResnetGeneratorAttResidual, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        self.no_last_tanh = no_last_tanh
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [nn.ReflectionPad2d(1),
                    nn.Conv2d(input_nc, ngf*2, kernel_size=3,
                            stride=2, padding=0, bias=use_bias),
                    norm_layer(ngf*2),
                    nn.ReLU(True)]

        for i in range(1, n_downsampling):
            mult = 2**i
            model += [nn.ReflectionPad2d(1),
                     nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3,
                                stride=2, padding=0, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]

        mult = 2**n_downsampling
        for i in range(n_blocks):
            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]

        self.model = nn.Sequential(*model)
        self.mask_cond = mask_cond
        attention_ngf = ngf * mult + (imgSize if mask_cond else 0) # conditioned on mask. imgSize is the mask row dimension

        model = []
        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            model += [nn.Upsample(scale_factor=2),
                      nn.ReflectionPad2d(1),
                      nn.Conv2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=1,
                                         padding=0,
                                         bias=use_bias),
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True)]

        if use_psp:
            model += [PSPModule(ngf, output_nc, norm_layer)]
        else:
            model += [nn.Conv2d(ngf, output_nc, kernel_size=1, padding=0)]

        if not self.no_last_tanh:
            model += [nn.Tanh()]

        self.model_decode = nn.Sequential(*model)

        # attention 
        self.model_att = nn.Sequential(
            nn.Conv2d(attention_ngf, attention_ngf//2, 1, bias=use_bias),
            nn.ReLU(True),
            nn.Conv2d(attention_ngf//2, imgSize, 1, bias=use_bias),
            nn.Sigmoid(),
        )
        self.IFFT = IFFT()
        self.FFT = FFT()

    def forward(self, input, mask):
        # mask_vector [B, imSize, 1, 1]
        hidden = self.model(input)
        res = self.model_decode(hidden)

        avg_hidden = F.avg_pool2d(hidden, hidden.shape[2])
        if self.mask_cond:
            att_w = self.model_att(torch.cat([avg_hidden, mask], 1)) # [B,imSize, 1,1]
        else:
            att_w = self.model_att(avg_hidden) # [B,imSize, 1,1]

        att_w = att_w.view(att_w.shape[0], 1, att_w.shape[1], 1) # [B, 1, imSize, 1]

        # attention residual in kspace
        ft_x = self.FFT(res)
        ft_in = self.FFT(input) # visible part
        ft_fuse = att_w * ft_x + (1-att_w) * ft_in
        fuse = self.IFFT(ft_fuse)

        ## to check the attention weight dist
        # import pdb; pdb.set_trace()
        # inv_ratio = ((1-att_w[0].squeeze()) * mask[0].squeeze()).mean()
        # vis_ratio = (att_w[0].squeeze() * (1-mask[0].squeeze())).mean()
        # print('visiable ratio {vis_ratio} and invisible ratio {inv_ratio}' )

        return fuse, res

class ResnetGeneratorMaskingResidual(nn.Module):
    
    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, 
                n_blocks=6, padding_type='reflect', no_last_tanh=False, n_downsampling=3, imgSize=128, use_psp=False):
        assert(n_blocks >= 0)
        super(ResnetGeneratorMaskingResidual, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        self.no_last_tanh = no_last_tanh
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [nn.ReflectionPad2d(1),
                    nn.Conv2d(input_nc, ngf*2, kernel_size=3,
                            stride=2, padding=0, bias=use_bias),
                    norm_layer(ngf*2),
                    nn.ReLU(True)]

        for i in range(1, n_downsampling):
            mult = 2**i
            model += [nn.ReflectionPad2d(1),
                     nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3,
                                stride=2, padding=0, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]

        mult = 2**n_downsampling
        for i in range(n_blocks):
            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]

        self.model = nn.Sequential(*model)
        attention_ngf = ngf * mult + imgSize # conditioned on mask. imgSize is the mask row dimension

        model = []
        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            model += [nn.Upsample(scale_factor=2),
                      nn.ReflectionPad2d(1),
                      nn.Conv2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=1,
                                         padding=0,
                                         bias=use_bias),
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True)]

        if use_psp:
            model += [PSPModule(ngf, output_nc, norm_layer)]
        else:
            model += [nn.Conv2d(ngf, output_nc, kernel_size=1, padding=0)]

        if not self.no_last_tanh:
            model += [nn.Tanh()]

        self.model_decode = nn.Sequential(*model)

        self.IFFT = IFFT()
        self.FFT = FFT()

    def forward(self, input, mask):
        # mask_vector [B, imSize, 1, 1]
        hidden = self.model(input)
        res = self.model_decode(hidden)

        # mask residual in kspace
        mask = mask.view(mask.shape[0],1,mask.shape[1],1)
        ft_x = self.FFT(res)
        ft_in = self.FFT(input) # visible part
        ft_fuse = (1 - mask) * ft_x + mask * ft_in
        fuse = self.IFFT(ft_fuse)

        return fuse, res

class PSPModule(nn.Module):
    def __init__(self, n_in, output_nc, norm_layer, psp_dim=16, scales=[32,16,8,4]):
        super(PSPModule, self).__init__()

        self.scales = scales
        self.model_in = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(n_in, psp_dim, kernel_size=3, padding=0),
            norm_layer(psp_dim),
            nn.ReLU(True)
        )

        self.parallel_models = nn.ModuleList()
        for scale in scales:
            self.parallel_models.append(
                    nn.Sequential(
                    nn.AvgPool2d(kernel_size=scale),
                    nn.Conv2d(psp_dim, 1, kernel_size=1,stride=1, padding=0),
                    nn.ReLU(True),
                    nn.Upsample(scale_factor=scale)
                )
            )            
        self.model_out = nn.Conv2d(n_in+len(scales), output_nc, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        out = [x]
        x_in = self.model_in(x)
        for module in self.parallel_models:
            out.append(module(x_in))
        fuse = torch.cat(out, 1)
        out = self.model_out(fuse)
        
        return out
class ResnetGeneratorPixelAttResidual(nn.Module):
    # Compared with ResnetGeneratorAttResidual, this generates pixel-wise soft attention and do not need to back to fourier space
    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, 
                n_blocks=6, padding_type='reflect', no_last_tanh=False, n_downsampling=3, imgSize=128):
        assert(n_blocks >= 0)
        super(ResnetGeneratorPixelAttResidual, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        self.no_last_tanh = no_last_tanh
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [nn.ReflectionPad2d(1),
                    nn.Conv2d(input_nc, ngf*2, kernel_size=3,
                            stride=2, padding=0, bias=use_bias),
                    norm_layer(ngf*2),
                    nn.ReLU(True)]

        for i in range(1, n_downsampling):
            mult = 2**i
            model += [nn.ReflectionPad2d(1),
                     nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3,
                                stride=2, padding=0, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]

        mult = 2**n_downsampling
        for i in range(n_blocks):
            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]

        self.model = nn.Sequential(*model)
        attention_ngf = ngf * mult + imgSize # conditioned on mask. imgSize is the mask row dimension

        model = []
        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            model += [nn.Upsample(scale_factor=2),
                      nn.ReflectionPad2d(1),
                      nn.Conv2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=1,
                                         padding=0,
                                         bias=use_bias),
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True)]

        model += [nn.Conv2d(ngf, output_nc, kernel_size=1, padding=0)]

        if not self.no_last_tanh:
            model += [nn.Tanh()]

        self.model_decode = nn.Sequential(*model)

        # attention
        self.model_att = nn.Sequential(*[
            nn.ConvTranspose2d(attention_ngf, attention_ngf//2,
                                         kernel_size=4, stride=2,
                                         padding=1, 
                                         bias=use_bias),
            nn.ReLU(True),
            nn.ConvTranspose2d(attention_ngf//2, attention_ngf//4,
                                         kernel_size=4, stride=2,
                                         padding=1, 
                                         bias=use_bias),
            nn.ReLU(True),
            nn.ConvTranspose2d(attention_ngf//4, 1,
                                         kernel_size=4, stride=2,
                                         padding=1, 
                                         bias=use_bias),
            nn.Sigmoid(),
        ])
  

    def forward(self, input, mask):
        # mask_vector [B, imSize, 1, 1]
        hidden = self.model(input)
        res = self.model_decode(hidden)

        b,_,h,w = hidden.shape
        att_w = self.model_att(torch.cat([hidden, mask.expand(b, mask.shape[1],h,w)], 1)) # per location attention [B, 1, h,w]
        # attention residual in kspace
        fuse = att_w * res + (1-att_w) * input

        return fuse, res

# Define a resnet block
class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim),
                       nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out


# Defines the Unet generator.
# |num_downs|: number of downsamplings in UNet. For example,
# if |num_downs| == 7, image of size 128x128 will become of size 1x1
# at the bottleneck
class UnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, num_downs, ngf=64,
                 norm_layer=nn.BatchNorm2d, use_dropout=False, no_last_tanh=False):
        super(UnetGenerator, self).__init__()

        # construct unet structure
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True)
        for i in range(num_downs - 5):
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer, no_last_tanh=no_last_tanh)

        self.model = unet_block

    def forward(self, input):
        return self.model(input)

# Defines the submodule with skip connection.
# X -------------------identity---------------------- X
#   |-- downsampling -- |submodule| -- upsampling --|
class UnetSkipConnectionBlock(nn.Module):
    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False, nearest_upsample=True, no_last_tanh=False):
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            if nearest_upsample:
                upconv = nn.Sequential(nn.Upsample(scale_factor=2),
                                nn.ReflectionPad2d(1),
                                nn.Conv2d(inner_nc * 2, outer_nc,
                                        kernel_size=3, stride=1,
                                        padding=0, bias=use_bias)
                )
            else:
                upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                            kernel_size=4, stride=2,
                                            padding=1)
            down = [downconv]
            if no_last_tanh:
                up = [uprelu, upconv]
            else:
                up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            if nearest_upsample:
                   upconv = nn.Sequential(nn.Upsample(scale_factor=2),
                                nn.ReflectionPad2d(1),
                                nn.Conv2d(inner_nc, outer_nc,
                                        kernel_size=3, stride=1,
                                        padding=0, bias=use_bias)
                )
            else:
                upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                            kernel_size=4, stride=2,
                                            padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            if nearest_upsample:
                   upconv = nn.Sequential(nn.Upsample(scale_factor=2),
                                nn.ReflectionPad2d(1),
                                nn.Conv2d(inner_nc * 2, outer_nc,
                                        kernel_size=3, stride=1,
                                        padding=0, bias=use_bias)
                )
            else:
                upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                            kernel_size=4, stride=2,
                                            padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:
            return torch.cat([x, self.model(x)], 1)


# Defines the PatchGAN discriminator with the specified arguments.
class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False):
        super(NLayerDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True)
        ]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                          kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                      kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]

        if use_sigmoid:
            sequence += [nn.Sigmoid()]
        
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        return self.model(input)


class PixelDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, norm_layer=nn.BatchNorm2d, use_sigmoid=False):
        super(PixelDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.net = [
            nn.Conv2d(input_nc, ndf, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf * 2, kernel_size=1, stride=1, padding=0, bias=use_bias),
            norm_layer(ndf * 2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 2, 1, kernel_size=1, stride=1, padding=0, bias=use_bias)]

        if use_sigmoid:
            self.net.append(nn.Sigmoid())

        self.net = nn.Sequential(*self.net)

    def forward(self, input):
        return self.net(input)



class VGGLoss(nn.Module):
    def __init__(self, gpu_ids, input_channel):
        super(VGGLoss, self).__init__()        
        self.vgg = Vgg19().cuda()
        self.criterion = nn.L1Loss()
        self.weights = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]     
        self.input_channel = input_channel   

    def forward(self, x, y):      
        if self.input_channel == 1:
            b,c,h,w = x.shape
            x = x.expand(b,3,h,w)
            y = y.expand(b,3,h,w)    

        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        for i in range(len(x_vgg)):
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())        
        return loss

from torchvision import models
class Vgg19(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)        
        h_relu3 = self.slice3(h_relu2)        
        h_relu4 = self.slice4(h_relu3)        
        h_relu5 = self.slice5(h_relu4)                
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out

class E_NLayers(nn.Module):
    def __init__(self, input_nc, output_nc=1, ndf=64, n_layers=3,
                 norm_layer=None, nl_layer=None, gpu_ids=[], vaeLike=False):
        super(E_NLayers, self).__init__()
        self.gpu_ids = gpu_ids
        self.vaeLike = vaeLike

        kw, padw = 4, 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw,
                              stride=2, padding=padw), nl_layer()]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 4)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                          kernel_size=kw, stride=2, padding=padw)]
            if norm_layer is not None:
                sequence += [norm_layer(ndf * nf_mult)]
            sequence += [nl_layer()]
        sequence += [nn.AvgPool2d(8)]
        self.conv = nn.Sequential(*sequence)
        self.fc = nn.Sequential(*[nn.Linear(ndf * nf_mult, output_nc)])
        if vaeLike:
            self.fcVar = nn.Sequential(*[nn.Linear(ndf * nf_mult, output_nc)])

    def forward(self, x):
        x_conv = self.conv(x)
        conv_flat = x_conv.view(x.size(0), -1)
        output = self.fc(conv_flat)
        if self.vaeLike:
            outputVar = self.fcVar(conv_flat)
            return output, outputVar
        return output


from torch.nn.parameter import Parameter
import math
class GraphConvolution(nn.Module):

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.in_d, self.out_d = in_features, out_features
        self.weight = Parameter(torch.FloatTensor(1, in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(1,out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(2))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        
        support = torch.bmm(input, self.weight.expand(input.shape[0], self.in_d, self.out_d))
        output = torch.bmm(adj, support)

        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

class FTGCN(nn.Module):
    def __init__(self, nfeat, hidden, out_dim, n_layers=6, dropout=0.0):
        super(FTGCN, self).__init__()

        self.gcn_modules = nn.ModuleList()

        self.gcn_modules.append(GraphConvolution(nfeat, hidden))
        for i in range(n_layers-2):
            self.gcn_modules.append(GraphConvolution(hidden, hidden))

        self.gcn_modules.append(GraphConvolution(hidden, out_dim))

        self.dropout = dropout

    def forward(self, input, adj, mask):
        
        x = input
        for gc in self.gcn_modules:
            x = F.relu(gc(x, adj))
            x = F.dropout(x, self.dropout, training=self.training)
        output = (1 - mask) * x + mask * input

        return output, x