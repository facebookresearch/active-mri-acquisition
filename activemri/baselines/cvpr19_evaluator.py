from typing import Any, Dict, List

import torch

from . import Policy
import miccai_2020.models.evaluator


# This is just a wrapper for the model in miccai_2020 folder
# noinspection PyAbstractClass
# TODO fix evaluator height
class CVPR19Evaluator(Policy):
    def __init__(
        self, evaluator_path: str, device: torch.device, add_mask: bool = False,
    ):
        super().__init__()
        evaluator_checkpoint = torch.load(evaluator_path)
        assert (
            evaluator_checkpoint is not None
            and evaluator_checkpoint["evaluator"] is not None
        )
        self.evaluator = miccai_2020.models.evaluator.EvaluatorNetwork(
            number_of_filters=evaluator_checkpoint[
                "options"
            ].number_of_evaluator_filters,
            number_of_conv_layers=evaluator_checkpoint[
                "options"
            ].number_of_evaluator_convolution_layers,
            use_sigmoid=False,
            width=evaluator_checkpoint["options"].image_width,
            height=640,
            mask_embed_dim=evaluator_checkpoint["options"].mask_embed_dim,
        )
        self.evaluator.load_state_dict(
            {
                key.replace("module.", ""): val
                for key, val in evaluator_checkpoint["evaluator"].items()
            }
        )
        self.evaluator.eval()
        self.evaluator.to(device)
        self.add_mask = add_mask
        self.device = device

    def get_action(self, obs: Dict[str, Any], **_kwargs) -> List[int]:
        with torch.no_grad():
            mask_embedding = (
                None
                if obs["extra_outputs"]["mask_embedding"] is None
                else obs["extra_outputs"]["mask_embedding"].to(self.device)
            )
            mask = obs["mask"].bool().to(self.device)
            mask = mask.view(mask.shape[0], 1, 1, -1)
            k_space_scores = self.evaluator(
                obs["reconstruction"].permute(0, 3, 1, 2).to(self.device),
                mask_embedding,
                mask if self.add_mask else None,
            )
            # Just fill chosen actions with some very large number to prevent from selecting again.
            k_space_scores.masked_fill_(mask.squeeze(), 100000)
            return torch.argmin(k_space_scores, dim=1).tolist()
