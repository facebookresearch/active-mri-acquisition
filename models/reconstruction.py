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


class ReconstructorNetwork(nn.Module):

    def __init__(self, number_of_encoder_input_channels=2, number_of_decoder_output_channels=3, number_of_filters=128, norm_layer=nn.BatchNorm2d, use_dropout=False,
                 number_of_layers_residual_bottleneck=6, number_of_cascade_blocks=3, mask_embed_dim=6, padding_type='reflect', n_downsampling=3, img_width=128,
                 use_deconv=True):
        super(ReconstructorNetwork, self).__init__()
        self.number_of_encoder_input_channels = number_of_encoder_input_channels
        self.number_of_decoder_output_channels = number_of_decoder_output_channels
        self.number_of_filters = number_of_filters
        self.use_deconv = use_deconv
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.number_of_cascade_blocks = number_of_cascade_blocks
        self.use_mask_embedding = True if mask_embed_dim > 0 else False

        if self.use_mask_embedding:
            number_of_encoder_input_channels += mask_embed_dim
            print('[Reconstructor Network] -> use masked embedding condition')

        # Lists of encoder, residual bottleneck and decoder blocks for all cascade blocks
        self.encoders_all_cascade_blocks = []
        self.residual_bottlenecks_all_cascade_blocks = []
        self.decoders_all_cascade_blocks = []

        # Architecture for the Cascade Blocks
        for iii in range(1, self.number_of_cascade_blocks + 1):
            #TODO : may be clean up the local model variables

            # Encoder for iii_th cascade block
            encoder = [nn.ReflectionPad2d(1),
                     nn.Conv2d(number_of_encoder_input_channels, number_of_filters, kernel_size=3,
                               stride=2, padding=0, bias=use_bias),
                     norm_layer(number_of_filters),
                     nn.ReLU(True)]

            for i in range(1, n_downsampling):
                mult = 2 ** i
                encoder += [nn.ReflectionPad2d(1),
                          nn.Conv2d(number_of_filters * mult // 2, number_of_filters * mult, kernel_size=3,
                                    stride=2, padding=0, bias=use_bias),
                          norm_layer(number_of_filters * mult),
                          nn.ReLU(True)]
            # setattr(self, 'model_encode' + str(iii), nn.Sequential(*encoder)) #TODO remove this renaming
            self.encoders_all_cascade_blocks.append(nn.Sequential(*encoder))

            # Bottleneck for iii_th cascade block
            residual_bottleneck = []
            mult = 2 ** (n_downsampling - 1)
            for i in range(number_of_layers_residual_bottleneck):
                residual_bottleneck += [ResnetBlock(number_of_filters * mult, padding_type=padding_type, norm_layer=norm_layer,
                                                    use_dropout=use_dropout, use_bias=use_bias)]

            # setattr(self, 'model' + str(iii), nn.Sequential(*residual_bottleneck)) #TODO : remove
            self.residual_bottlenecks_all_cascade_blocks.append(nn.Sequential(*residual_bottleneck))

            # Decoder for iii_th cascade block
            decoder = []
            for i in range(n_downsampling):
                mult = 2 ** (n_downsampling - 1 - i)
                if self.use_deconv:
                    decoder += [nn.ConvTranspose2d(number_of_filters * mult, int(number_of_filters * mult / 2),
                                                   kernel_size=4, stride=2,
                                                   padding=1,
                                                   bias=use_bias),
                              norm_layer(int(number_of_filters * mult / 2)),
                              nn.ReLU(True)]
                else:
                    decoder += [nn.Upsample(scale_factor=2),
                              nn.ReflectionPad2d(1)] + \
                             [nn.Conv2d(number_of_filters * mult, int(number_of_filters * mult / 2),
                                        kernel_size=3, stride=1,
                                        padding=0,
                                        bias=use_bias),
                              norm_layer(int(number_of_filters * mult / 2)),
                              nn.ReLU(True)]

                    # model += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
            decoder += [nn.Conv2d(number_of_filters // 2, number_of_decoder_output_channels, kernel_size=1, padding=0, bias=False)]  # better

            # setattr(self, 'model_decode' + str(iii), nn.Sequential(*decoder)) #TODO
            self.decoders_all_cascade_blocks.append(nn.Sequential(*decoder))

        if self.use_mask_embedding:
            self.mask_embedding_layer = nn.Sequential(nn.Conv2d(img_width, mask_embed_dim, 1, 1))

        self.IFFT = IFFT()
        self.FFT = FFT()

    def data_consistency(self, x, input, mask):
        ft_x = self.FFT(x)
        fuse = self.IFFT((1 - mask) * ft_x) + input
        return fuse

    def embed_mask(self, mask):
        b, c, h, w = mask.shape
        mask = mask.view(b, w, 1, 1)
        cond_embed = self.mask_embedding_layer(mask)
        # cond_embed = cond_embed.repeat(1,1,w,w)

        return cond_embed

    def forward(self, zero_filled_input, mask):
        '''

        Args:
                    zero_filled_input:  masked input image
                                        shape = (batch_size, 2, height, width)
                    mask: mask used in creating the zero filled image from ground truth image
                                        shape = (batch_size, 1, 1, width)

        Returns:    reconstructed high resolution image
                    uncertainy map
                    mask_embedding

        '''
        if self.use_mask_embedding:
            mask_embedding = self.embed_mask(mask)
            mask_embedding = mask_embedding.repeat(1, 1, zero_filled_input.shape[2], zero_filled_input.shape[3])
            encoder_input = torch.cat([zero_filled_input, mask_embedding], 1)
        else:
            encoder_input = zero_filled_input
            mask_embedding = None

        for cascade_block, (encoder, residual_bottleneck, decoder) in enumerate(zip(self.encoders_all_cascade_blocks, self.residual_bottlenecks_all_cascade_blocks, self.decoders_all_cascade_blocks)):
            encoder_output = encoder(encoder_input)
            if cascade_block > 0:
                encoder_output += residual_bottleneck_output    #Skip connection from previous residual block

            residual_bottleneck_output = residual_bottleneck(encoder_output)

            decoder_output = decoder(residual_bottleneck_output)

            reconstructed_image = self.data_consistency(decoder_output[:,:-1,:,:], zero_filled_input, mask)
            uncertainty_map = decoder_output[:, -1:, :, :]

            if self.use_mask_embedding:
                encoder_input = torch.cat([reconstructed_image, mask_embedding], 1)
            else:
                encoder_input = reconstructed_image

        return reconstructed_image, uncertainty_map, mask_embedding

        # #TODO: write a test with one dicom (128 x 128) and one raw (640 x 368) noise and try random parameters
def test(data):
    batch = 4

    if data == 'dicom':
        # DICOM
        dicom = torch.rand(batch, 2, 128, 128)
        dicom = dicom.type(torch.FloatTensor)

        mask_dicom = torch.rand(batch, 1, 1, 128)
        mask_dicom.type(torch.FloatTensor)

        net_dicom = ReconstructorNetwork(number_of_encoder_input_channels=2,
                                         number_of_decoder_output_channels=3,
                                         number_of_filters=64,
                                         number_of_layers_residual_bottleneck=6,
                                         number_of_cascade_blocks=4,
                                         mask_embed_dim=0,
                                         n_downsampling=4,
                                         img_width=dicom.shape[3]
                                         )

        out = net_dicom.forward(dicom, mask_dicom)

    elif data == 'raw':
        # RAW
        raw = torch.rand(batch, 2, 640,368)
        raw = raw.type(torch.FloatTensor)

        mask_raw = torch.randint(0, 1, (batch, 1, 1, 368))
        mask_raw = mask_raw.type(torch.FloatTensor)


        net_raw = ReconstructorNetwork(number_of_encoder_input_channels=2,
                                         number_of_decoder_output_channels=3,
                                         number_of_filters=128,
                                         number_of_layers_residual_bottleneck=3,
                                         number_of_cascade_blocks=3,
                                         mask_embed_dim=6,
                                         n_downsampling=3,
                                         img_width=raw.shape[3]
                                         )

        out = net_raw.forward(raw, mask_raw)

    print('reconstruction shape :', out[0].shape)
    print('uncertainty shape :', out[1].shape)

    if out[2] is not None:
        print('embedding shape :', out[2].shape)


if __name__ == '__main__':
    test(data='raw')
    test(data='dicom')


