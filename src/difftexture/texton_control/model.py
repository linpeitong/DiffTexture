"""A zero-initialized spatial control branch for a frozen DDPM U-Net."""

from __future__ import annotations

import torch
from torch import nn


class TextonControlModel(nn.Module):
    """Inject texton-map features into a frozen DDPM through zero convolutions.

    The adapter starts as the identity with respect to the main DDPM, so loading
    a trained inpainting checkpoint reproduces its output before fine-tuning.
    """

    def __init__(self, unet: nn.Module, channels: int = 3):
        super().__init__()
        self.unet = unet
        for parameter in self.unet.parameters():
            parameter.requires_grad_(False)
        self.condition_encoder = nn.Sequential(
            nn.Conv2d(channels, 64, kernel_size=3, padding=1), nn.SiLU(),
            nn.Conv2d(64, channels, kernel_size=3, padding=1),
        )
        nn.init.zeros_(self.condition_encoder[-1].weight)
        nn.init.zeros_(self.condition_encoder[-1].bias)

    def forward(self, sample, timestep, texton_map, conditioning_scale: float = 1.0):
        conditioned_sample = sample + conditioning_scale * self.condition_encoder(texton_map)
        return self.unet(conditioned_sample, timestep).sample

    def save_pretrained(self, directory: str) -> None:
        torch.save(self.condition_encoder.state_dict(), f"{directory}/texton_control_adapter.pt")

    def load_adapter(self, path: str) -> None:
        self.condition_encoder.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
