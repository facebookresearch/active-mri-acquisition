from typing import Optional, Dict, Any

import fastmri.models
import torch

import activemri.models


# noinspection PyAbstractClass
class Unet(activemri.models.Reconstructor):
    def __init__(
        self, in_chans=2, out_chans=2, chans=32, num_pool_layers=4, drop_prob=0.0
    ):
        super().__init__()
        self.unet = fastmri.models.Unet(
            in_chans,
            out_chans,
            chans=chans,
            num_pool_layers=num_pool_layers,
            drop_prob=drop_prob,
        )

    def forward(self, image: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {"reconstruction": self.unet(image).squeeze(1)}

    def init_from_checkpoint(self, checkpoint: Dict[str, Any]) -> Optional[Any]:
        self.unet.load_state_dict(checkpoint["state_dict"])
        return None
