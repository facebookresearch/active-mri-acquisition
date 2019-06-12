import functools
import torch
import torch.nn as nn
import torch.nn.functional as F
from fft_utils import RFFT, IFFT, FFT
from torch.nn import init


#TODO: can we use this inside the model? Can we simplify this block?
#TODO: do we need functools?
def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        # norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=True) # Original
        # We do not need to learn it. So we can use evaluation mode for testing and Dropout is appliable easily
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        norm_layer = None
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


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

#TODO: DataParallel should not be here...
def init_net(net, init_type='normal', gpu_ids=[]):
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)
    init_weights(net, init_type)
    return net

def define_G(input_nc, output_nc, ngf, which_model_netG, norm='batch', use_dropout=False,
             init_type='normal', gpu_ids=[], no_last_tanh=False):
    netG = None
    norm_layer = get_norm_layer(norm_type=norm)
    netG = PasNetPlus(input_nc, output_nc, ngf, norm_layer=norm_layer,
                      use_dropout=use_dropout, n_blocks=9, no_last_tanh=no_last_tanh,
                      n_downsampling=3, imgSize=128, mask_cond=True, use_deconv=True)
    return init_net(netG, init_type, gpu_ids)


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


## Currently the best Aug 10
class PasNetPlus(nn.Module):

    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False,
                n_blocks=6, padding_type='reflect', no_last_tanh=False, n_downsampling=3, imgSize=128,
                mask_cond=True, use_deconv=True, no_meta=True):
        assert(n_blocks >= 0)
        super(PasNetPlus, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        self.no_last_tanh = no_last_tanh
        self.use_deconv = use_deconv
        self.no_meta = no_meta
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.n_recurive = 3
        self.mask_cond = mask_cond
        mask_embed_dim = 6

        if mask_cond:
            input_nc += mask_embed_dim
            print('[PasNet] -> use masked embedding condition')

        for iii in range(1, self.n_recurive+1):
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
            setattr(self, 'model_encode'+str(iii), nn.Sequential(*model))

            model = []
            mult = 2**n_downsampling
            for i in range(n_blocks//self.n_recurive):
                model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer,
                                      use_dropout=use_dropout, use_bias=use_bias)]

            setattr(self, 'model'+str(iii), nn.Sequential(*model))

            model = []
            for i in range(n_downsampling):
                mult = 2**(n_downsampling - i)
                if self.use_deconv:
                    model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=4, stride=2,
                                         padding=1,
                                         bias=use_bias),
                                norm_layer(int(ngf * mult / 2)),
                                nn.ReLU(True)]
                else:
                    model += [nn.Upsample(scale_factor=2),
                            nn.ReflectionPad2d(1)] + \
                            [nn.Conv2d(ngf * mult, int(ngf * mult / 2),
                                                kernel_size=3, stride=1,
                                                padding=0,
                                                bias=use_bias),
                            norm_layer(int(ngf * mult / 2)),
                            nn.ReLU(True)]

            # model += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
            model += [nn.Conv2d(ngf, output_nc, kernel_size=1, padding=0, bias=False)] # better

            setattr(self, 'model_decode'+str(iii), nn.Sequential(*model))

        if mask_cond:
            mask_inc = imgSize if no_meta else imgSize+3
            self.mask_embed = nn.Sequential(nn.Conv2d(mask_inc, mask_embed_dim, 1, 1))

        self.IFFT = IFFT()
        self.FFT = FFT()
        self.use_sampling_at_stage = None

    def kspace_fuse(self, x, input, mask):
        ft_x = self.FFT(x)
        fuse = self.IFFT((1 - mask) * ft_x) + input

        return fuse

    def embed_condtions(self, mask):
        b,c,h,w = mask.shape
        mask = mask.view(b,w,1,1)
        cond_embed = self.mask_embed(mask)
        cond_embed = cond_embed.repeat(1,1,w,w)

        return cond_embed

    def reparam(self, mu, logvar):
        _mu = mu[:,:1,:,:]
        std = logvar.mul(0.5).exp()
        eps = torch.zeros_like(logvar).normal_()
        q_z = eps.mul(std).add_(_mu)
        mu[:,:1,:,:] = q_z
        return mu

    def forward(self, input, mask, use_sampling_at_stage=None):
        mask_embed = None
        # mask in [B,1,H,1]
        if self.mask_cond:
            mask_embed = self.embed_condtions(mask)
            input_ = torch.cat([input, mask_embed], 1)
        else:
            input_ = input

        hidden_in1 = self.model_encode1(input_)
        hidden_out1 = self.model1(hidden_in1)
        out1_ = self.model_decode1(hidden_out1)

        logvar1 = out1_[:,2:,:,:]
        out1 = self.kspace_fuse(out1_[:,:2,:,:], input, mask)

        if use_sampling_at_stage == 1:
            out1 = self.reparam(out1, logvar1)

        if self.mask_cond:
            out1_ = torch.cat([out1, mask_embed], 1)
        else:
            out1_ = out1

        hidden_in2 = self.model_encode2(out1_)
        hidden_in2 = hidden_in2 + hidden_out1
        hidden_out2 = self.model2(hidden_in2)
        out2_ = self.model_decode2(hidden_out2)

        logvar2 = out2_[:,2:,:,:]
        out2 = self.kspace_fuse(out2_[:,:2,:,:], input, mask)

        if use_sampling_at_stage == 2:
            out1 = self.reparam(out2, logvar2)

        if self.mask_cond:
            out2_ = torch.cat([out2, mask_embed], 1)
        else:
            out2_ = out2

        hidden_in3 = self.model_encode3(out2_)
        hidden_in3 = hidden_in3 + hidden_out2
        hidden_out3 = self.model3(hidden_in3)
        out3_ = self.model_decode3(hidden_out3)

        logvar3 = out3_[:,2:,:,:]
        out3 = self.kspace_fuse(out3_[:,:2,:,:], input, mask)

        if use_sampling_at_stage == 3:
            out3 = self.reparam(out3, logvar3)

        return [out1, out2, out3], [logvar1, logvar2, logvar3],  mask_embed
