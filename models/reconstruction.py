import functools
import torch
import torch.nn as nn

from . import fft_utils


def get_norm_layer(norm_type="instance"):
    if norm_type == "batch":
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == "instance":
        norm_layer = functools.partial(
            nn.InstanceNorm2d, affine=False, track_running_stats=False
        )
    elif norm_type == "none":
        norm_layer = None
    else:
        raise NotImplementedError("normalization layer [%s] is not found" % norm_type)
    return norm_layer


def init_func(m):
    init_type = "normal"
    gain = 0.02
    classname = m.__class__.__name__
    if hasattr(m, "weight") and (
        classname.find("Conv") != -1 or classname.find("Linear") != -1
    ):
        if init_type == "normal":
            torch.nn.init.normal_(m.weight.data, 0.0, gain)
        elif init_type == "xavier":
            torch.nn.init.xavier_normal_(m.weight.data, gain=gain)
        elif init_type == "kaiming":
            torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode="fan_in")
        elif init_type == "orthogonal":
            torch.nn.init.orthogonal_(m.weight.data, gain=gain)
        else:
            raise NotImplementedError(
                "initialization method [%s] is not implemented" % init_type
            )
        if hasattr(m, "bias") and m.bias is not None:
            torch.nn.init.constant_(m.bias.data, 0.0)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, gain)
        torch.nn.init.constant_(m.bias.data, 0.0)


# Define a resnet block
class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, dropout_probability, use_bias):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(
            dim, padding_type, norm_layer, dropout_probability, use_bias
        )

    def build_conv_block(
        self, dim, padding_type, norm_layer, dropout_probability, use_bias
    ):
        conv_block = []
        p = 0
        if padding_type == "reflect":
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == "replicate":
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == "zero":
            p = 1
        else:
            raise NotImplementedError("padding [%s] is not implemented" % padding_type)

        conv_block += [
            nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
            norm_layer(dim),
            nn.ReLU(True),
        ]
        if dropout_probability > 0:
            conv_block += [nn.Dropout(dropout_probability)]

        p = 0
        if padding_type == "reflect":
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == "replicate":
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == "zero":
            p = 1
        else:
            raise NotImplementedError("padding [%s] is not implemented" % padding_type)
        conv_block += [
            nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
            norm_layer(dim),
        ]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out


class ReconstructorNetwork(nn.Module):
    def __init__(
        self,
        number_of_encoder_input_channels=2,
        number_of_decoder_output_channels=3,
        number_of_filters=128,
        dropout_probability=0.0,
        number_of_layers_residual_bottleneck=6,
        number_of_cascade_blocks=3,
        mask_embed_dim=6,
        padding_type="reflect",
        n_downsampling=3,
        img_width=128,
        use_deconv=True,
    ):
        super(ReconstructorNetwork, self).__init__()
        self.number_of_encoder_input_channels = number_of_encoder_input_channels
        self.number_of_decoder_output_channels = number_of_decoder_output_channels
        self.number_of_filters = number_of_filters
        self.use_deconv = use_deconv
        norm_layer = functools.partial(
            nn.InstanceNorm2d, affine=False, track_running_stats=False
        )

        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.number_of_cascade_blocks = number_of_cascade_blocks
        self.use_mask_embedding = True if mask_embed_dim > 0 else False

        if self.use_mask_embedding:
            number_of_encoder_input_channels += mask_embed_dim
            print("[Reconstructor Network] -> use masked embedding condition")

        # Lists of encoder, residual bottleneck and decoder blocks for all cascade blocks
        self.encoders_all_cascade_blocks = nn.ModuleList()
        self.residual_bottlenecks_all_cascade_blocks = nn.ModuleList()
        self.decoders_all_cascade_blocks = nn.ModuleList()

        # Architecture for the Cascade Blocks
        for iii in range(1, self.number_of_cascade_blocks + 1):

            # Encoder for iii_th cascade block
            encoder = [
                nn.ReflectionPad2d(1),
                nn.Conv2d(
                    number_of_encoder_input_channels,
                    number_of_filters,
                    kernel_size=3,
                    stride=2,
                    padding=0,
                    bias=use_bias,
                ),
                norm_layer(number_of_filters),
                nn.ReLU(True),
            ]

            for i in range(1, n_downsampling):
                mult = 2 ** i
                encoder += [
                    nn.ReflectionPad2d(1),
                    nn.Conv2d(
                        number_of_filters * mult // 2,
                        number_of_filters * mult,
                        kernel_size=3,
                        stride=2,
                        padding=0,
                        bias=use_bias,
                    ),
                    norm_layer(number_of_filters * mult),
                    nn.ReLU(True),
                ]

            self.encoders_all_cascade_blocks.append(nn.Sequential(*encoder))

            # Bottleneck for iii_th cascade block
            residual_bottleneck = []
            mult = 2 ** (n_downsampling - 1)
            for i in range(number_of_layers_residual_bottleneck):
                residual_bottleneck += [
                    ResnetBlock(
                        number_of_filters * mult,
                        padding_type=padding_type,
                        norm_layer=norm_layer,
                        dropout_probability=dropout_probability,
                        use_bias=use_bias,
                    )
                ]

            self.residual_bottlenecks_all_cascade_blocks.append(
                nn.Sequential(*residual_bottleneck)
            )

            # Decoder for iii_th cascade block
            decoder = []
            for i in range(n_downsampling):
                mult = 2 ** (n_downsampling - 1 - i)
                if self.use_deconv:
                    decoder += [
                        nn.ConvTranspose2d(
                            number_of_filters * mult,
                            int(number_of_filters * mult / 2),
                            kernel_size=4,
                            stride=2,
                            padding=1,
                            bias=use_bias,
                        ),
                        norm_layer(int(number_of_filters * mult / 2)),
                        nn.ReLU(True),
                    ]
                else:
                    decoder += [nn.Upsample(scale_factor=2), nn.ReflectionPad2d(1)] + [
                        nn.Conv2d(
                            number_of_filters * mult,
                            int(number_of_filters * mult / 2),
                            kernel_size=3,
                            stride=1,
                            padding=0,
                            bias=use_bias,
                        ),
                        norm_layer(int(number_of_filters * mult / 2)),
                        nn.ReLU(True),
                    ]
            decoder += [
                nn.Conv2d(
                    number_of_filters // 2,
                    number_of_decoder_output_channels,
                    kernel_size=1,
                    padding=0,
                    bias=False,
                )
            ]  # better

            self.decoders_all_cascade_blocks.append(nn.Sequential(*decoder))

        if self.use_mask_embedding:
            self.mask_embedding_layer = nn.Sequential(
                nn.Conv2d(img_width, mask_embed_dim, 1, 1)
            )

        self.apply(init_func)

    def data_consistency(self, x, input, mask):
        ft_x = fft_utils.fft(x)
        fuse = (
            fft_utils.ifft(
                torch.where((1 - mask).byte(), ft_x, torch.tensor(0.0).to(ft_x.device))
            )
            + input
        )
        return fuse

    def embed_mask(self, mask):
        b, c, h, w = mask.shape
        mask = mask.view(b, w, 1, 1)
        cond_embed = self.mask_embedding_layer(mask)
        return cond_embed

    # noinspection PyUnboundLocalVariable
    def forward(self, zero_filled_input, mask):
        """

        Args:
                    zero_filled_input:  masked input image
                                        shape = (batch_size, 2, height, width)
                    mask: mask used in creating the zero filled image from ground truth image
                                        shape = (batch_size, 1, 1, width)

        Returns:    reconstructed high resolution image
                    uncertainty map
                    mask_embedding
        """
        if self.use_mask_embedding:
            mask_embedding = self.embed_mask(mask)
            mask_embedding = mask_embedding.repeat(
                1, 1, zero_filled_input.shape[2], zero_filled_input.shape[3]
            )
            encoder_input = torch.cat([zero_filled_input, mask_embedding], 1)
        else:
            encoder_input = zero_filled_input
            mask_embedding = None

        for cascade_block, (encoder, residual_bottleneck, decoder) in enumerate(
            zip(
                self.encoders_all_cascade_blocks,
                self.residual_bottlenecks_all_cascade_blocks,
                self.decoders_all_cascade_blocks,
            )
        ):
            encoder_output = encoder(encoder_input)
            if cascade_block > 0:
                # Skip connection from previous residual block
                encoder_output = encoder_output + residual_bottleneck_output

            residual_bottleneck_output = residual_bottleneck(encoder_output)

            decoder_output = decoder(residual_bottleneck_output)

            reconstructed_image = self.data_consistency(
                decoder_output[:, :-1, ...], zero_filled_input, mask
            )
            uncertainty_map = decoder_output[:, -1:, :, :]

            if self.use_mask_embedding:
                encoder_input = torch.cat([reconstructed_image, mask_embedding], 1)
            else:
                encoder_input = reconstructed_image

        return reconstructed_image, uncertainty_map, mask_embedding


def test(data):
    batch = 4

    if data == "dicom":
        # DICOM
        dicom = torch.rand(batch, 2, 128, 128)
        dicom = dicom.type(torch.FloatTensor)

        mask_dicom = torch.rand(batch, 1, 1, 128)
        mask_dicom.type(torch.FloatTensor)

        net_dicom = ReconstructorNetwork(
            number_of_encoder_input_channels=2,
            number_of_decoder_output_channels=3,
            number_of_filters=64,
            number_of_layers_residual_bottleneck=6,
            number_of_cascade_blocks=4,
            mask_embed_dim=0,
            n_downsampling=4,
            img_width=dicom.shape[3],
            dropout_probability=0.5,
        )

        out = net_dicom.forward(dicom, mask_dicom)

    elif data == "raw":
        # RAW
        raw = torch.rand(batch, 2, 640, 368)
        raw = raw.type(torch.FloatTensor)

        mask_raw = torch.randint(0, 1, (batch, 1, 1, 368))
        mask_raw = mask_raw.type(torch.FloatTensor)

        net_raw = ReconstructorNetwork(
            number_of_encoder_input_channels=2,
            number_of_decoder_output_channels=3,
            number_of_filters=128,
            number_of_layers_residual_bottleneck=3,
            number_of_cascade_blocks=3,
            mask_embed_dim=6,
            n_downsampling=3,
            img_width=raw.shape[3],
        )

        out = net_raw.forward(raw, mask_raw)

    print("reconstruction shape :", out[0].shape)
    print("uncertainty shape :", out[1].shape)

    if out[2] is not None:
        print("embedding shape :", out[2].shape)


if __name__ == "__main__":
    test(data="raw")
    test(data="dicom")
