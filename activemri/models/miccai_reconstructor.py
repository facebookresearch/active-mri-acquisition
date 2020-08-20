from typing import Any, Dict

import torch
import torch.nn as nn

import miccai_2020.models.reconstruction


# This is just a wrapper for the model in miccai_2020 folder
# noinspection PyAbstractClass
class MICCAIReconstructor(nn.Module):
    def __init__(
        self,
        number_of_encoder_input_channels: int = 2,
        number_of_decoder_output_channels: int = 3,
        number_of_filters: int = 128,
        dropout_probability: float = 0.0,
        number_of_layers_residual_bottleneck: int = 6,
        number_of_cascade_blocks: int = 3,
        mask_embed_dim: int = 6,
        padding_type: str = "reflect",
        n_downsampling: int = 3,
        img_width: int = 128,
        use_deconv: bool = True,
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
    def forward(
        self, zero_filled_input: torch.Tensor, mask: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        reconstructed_image, uncertainty_map, mask_embedding = self.reconstructor(
            zero_filled_input, mask
        )
        reconstructed_image = reconstructed_image.permute(0, 2, 3, 1)
        uncertainty_map = uncertainty_map.permute(0, 2, 3, 1)

        return {
            "reconstruction": reconstructed_image,
            "uncertainty_map": uncertainty_map,
            "mask_embedding": mask_embedding,
        }

    def init_from_checkpoint(self, checkpoint: Dict[str, Any]):
        return self.reconstructor.init_from_checkpoint(checkpoint)
