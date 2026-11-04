"""Context-aware DDPM sampling with noise restricted to an inpainting mask."""

from __future__ import annotations

from typing import Optional

import torch
from diffusers.image_processor import VaeImageProcessor
from diffusers.pipelines.pipeline_utils import DiffusionPipeline, ImagePipelineOutput
from diffusers.utils.torch_utils import randn_tensor


class DDPMInpaintingPipeline(DiffusionPipeline):
    """Pixel-space DDPM inpainting that keeps the unmasked context clean."""

    def __init__(self, unet, scheduler):
        super().__init__()
        self.register_modules(unet=unet, scheduler=scheduler)
        self.image_processor = VaeImageProcessor(vae_scale_factor=1, do_convert_rgb=True)
        self.mask_processor = VaeImageProcessor(vae_scale_factor=1, do_normalize=False, do_binarize=True, do_convert_grayscale=True)

    @torch.no_grad()
    def __call__(self, image, mask, num_inference_steps: int = 1000, generator: Optional[torch.Generator] = None, output_type: str = "pil"):
        init_image = self.image_processor.preprocess(image).to(self.device, dtype=self.unet.dtype)
        mask_tensor = self.mask_processor.preprocess(mask).to(self.device, dtype=self.unet.dtype)
        if init_image.shape[0] != mask_tensor.shape[0]:
            raise ValueError("image and mask must have the same batch size")
        noise = randn_tensor(init_image.shape, generator=generator, device=self.device, dtype=init_image.dtype)
        sample = noise * mask_tensor + init_image * (1 - mask_tensor)
        self.scheduler.set_timesteps(num_inference_steps, device=self.device)
        for timestep in self.progress_bar(self.scheduler.timesteps):
            noise_pred = self.unet(sample, timestep).sample
            sample = self.scheduler.step(noise_pred, timestep, sample, generator=generator).prev_sample
            sample = sample * mask_tensor + init_image * (1 - mask_tensor)
        result = (sample / 2 + 0.5).clamp(0, 1).cpu().permute(0, 2, 3, 1).numpy()
        return ImagePipelineOutput(images=self.numpy_to_pil(result) if output_type == "pil" else result)
