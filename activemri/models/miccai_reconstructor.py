import torch.nn as nn

import miccai_2020.models.reconstruction


# This is just a wrapper for the model in miccai_2020 folder
class MICCAIReconstructor(nn.Module):
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
        super(MICCAIReconstructor, self).__init__()
        self.reconstructor = miccai_2020.models.reconstruction.ReconstructorNetwork(
            number_of_encoder_input_channels=number_of_encoder_input_channels,
            number_of_decoder_output_channels=number_of_decoder_output_channels,
            number_of_filters=number_of_filters,
            dropout_probability=dropout_probability,
            number_of_layers_residual_bottleneck=number_of_layers_residual_bottleneck,
            number_of_cascade_blocks=number_of_cascade_blocks,
            mask_embed_dim=mask_embed_dim,
            padding_type=padding_type,
            n_downsampling=n_downsampling,
            img_width=img_width,
            use_deconv=use_deconv,
        )

    # noinspection PyUnboundLocalVariable
    def forward(self, zero_filled_input, mask):
        reconstructed_image, uncertainty_map, mask_embedding = self.reconstructor(
            zero_filled_input, mask
        )
        reconstructed_image = reconstructed_image.permute(0, 2, 3, 1)
        uncertainty_map = uncertainty_map.permute(0, 2, 3, 1)

        return reconstructed_image, uncertainty_map, mask_embedding

    def init_from_checkpoint(self, checkpoint):
        return self.reconstructor.init_from_checkpoint(checkpoint)
