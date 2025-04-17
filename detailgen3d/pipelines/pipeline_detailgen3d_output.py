from dataclasses import dataclass

import torch
from diffusers.utils import BaseOutput


@dataclass
class DetailGen3DPipelineOutput(BaseOutput):
    r"""
    Output class for DetailGen3D pipelines.
    """

    samples: torch.Tensor
