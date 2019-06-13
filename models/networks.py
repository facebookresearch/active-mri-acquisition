import functools
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torch.optim import lr_scheduler
from .fft_utils import RFFT, IFFT, FFT

###############################################################################
# Helper Functions
###############################################################################


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

    
def define_G(input_nc, output_nc, ngf, which_model_netG, norm='batch', use_dropout=False,
             init_type='normal', gpu_ids=[], no_last_tanh=False):
    netG = None
    norm_layer = get_norm_layer(norm_type=norm)
    netG = PasNetPlus(input_nc, output_nc, ngf, norm_layer=norm_layer,
                      use_dropout=use_dropout, n_blocks=9, no_last_tanh=no_last_tanh,
                      n_downsampling=3, imgSize=128, mask_cond=True, use_deconv=True)
    return init_net(netG, init_type, gpu_ids)


class SimpleSequential(nn.Module):
    def __init__(self, net1, net2):
        super(SimpleSequential, self).__init__()
        self.net1 = net1
        self.net2 = net2

    def forward(self, x, mask):
        output = self.net1(x,mask)
        return self.net2(output,mask)


def define_D(input_nc, ndf, which_model_netD,
             n_layers_D=3, norm='batch', use_sigmoid=False, init_type='normal', gpu_ids=[], preprocess_module=None):
    netD = None
    norm_layer = get_norm_layer(norm_type=norm)

    netD = NLayerDiscriminatorChannel(input_nc, ndf, n_layers_D, norm_layer=norm_layer, use_sigmoid=use_sigmoid)

    if preprocess_module is not None:
        netD = SimpleSequential(preprocess_module, netD)

    return init_net(netD, init_type, gpu_ids)


##############################################################################
# Classes
##############################################################################
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
            self.RFFT = RFFT()
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
                w = gt.shape[2]
                ks_gt = self.RFFT(gt[:,:1,:,:], normalized=True) 
                ks_input = self.RFFT(pred, normalized=True) 
                ks_row_mse = F.mse_loss(
                    ks_input, ks_gt, reduce=False).sum(1, keepdim=True).sum(2, keepdim=True).squeeze() / (2*w)
                energy = torch.exp(-ks_row_mse * self.gamma)

                # do some bin process
                # import pdb; pdb.set_trace()
                # energy = torch.floor(energy * 10 / self.bin) * self.bin / 10
                
                target_tensor[:] = energy
            # force observed part to always
            for i in range(mask.shape[0]):
                idx = torch.nonzero(mask[i, 0, 0, :])
                target_tensor[i,idx] = 1 
        return target_tensor

    def __call__(self, input, target_is_real, mask, degree=1, updateG=False, pred_and_gt=None):
        # input [B, imSize]
        # degree is the realistic degree of output
        # set updateG to True when training G.
        target_tensor = self.get_target_tensor(input, target_is_real, degree, mask, pred_and_gt)
        b,w = target_tensor.shape
        if updateG and not self.grad_ctx:
            mask_ = mask.squeeze()
            # maskout the observed part loss
            return self.loss(input * (1-mask_), target_tensor * (1-mask_)) / (1-mask_).sum()
        else:
            return self.loss(input, target_tensor) / (b*w)


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


class NLayerDiscriminatorChannel(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, 
            norm_layer=nn.BatchNorm2d, use_sigmoid=False, imSize=128):
        print(f'[NLayerDiscriminatorChannel] -> n_layers = {n_layers}, n_channel {input_nc}')
        super(NLayerDiscriminatorChannel, self).__init__()
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
            nf_mult = min(2**n, 4)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                          kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        kw = imSize//2**n_layers
        sequence += [nn.AvgPool2d(kernel_size=kw)]
        sequence += [nn.Conv2d(ndf * nf_mult, imSize, kernel_size=1, stride=1, padding=0)]

        if use_sigmoid:
            sequence += [nn.Sigmoid()]
        
        self.model = nn.Sequential(*sequence)

    def forward(self, input, mask):
        # mask is not used
        return self.model(input).squeeze()


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
