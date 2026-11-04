"""Texton-controlled context-aware DDPM sampling."""

from __future__ import annotations

import torch
from diffusers.image_processor import VaeImageProcessor
from diffusers.pipelines.pipeline_utils import DiffusionPipeline, ImagePipelineOutput
from diffusers.utils.torch_utils import randn_tensor


class TextonControlPipeline(DiffusionPipeline):
    def __init__(self, control_model, scheduler):
        super().__init__(); self.register_modules(control_model=control_model, scheduler=scheduler)
        self.image_processor = VaeImageProcessor(vae_scale_factor=1, do_convert_rgb=True)
        self.mask_processor = VaeImageProcessor(vae_scale_factor=1, do_normalize=False, do_binarize=True, do_convert_grayscale=True)

    @torch.no_grad()
    def __call__(self, image, mask, texton_map, num_inference_steps=1000, conditioning_scale=1.0, generator=None):
        init = self.image_processor.preprocess(image).to(self.device, dtype=self.control_model.unet.dtype)
        mask_tensor = self.mask_processor.preprocess(mask).to(self.device, dtype=init.dtype)
        control = self.image_processor.preprocess(texton_map).to(self.device, dtype=init.dtype)
        if len({init.shape[0], mask_tensor.shape[0], control.shape[0]}) != 1:
            raise ValueError("image, mask, and texton_map must have equal batch sizes")
        sample = randn_tensor(init.shape, generator=generator, device=self.device, dtype=init.dtype) * mask_tensor + init * (1 - mask_tensor)
        self.scheduler.set_timesteps(num_inference_steps, device=self.device)
        for timestep in self.progress_bar(self.scheduler.timesteps):
            prediction = self.control_model(sample, timestep, control, conditioning_scale)
            sample = self.scheduler.step(prediction, timestep, sample, generator=generator).prev_sample
            sample = sample * mask_tensor + init * (1 - mask_tensor)
        result = (sample / 2 + 0.5).clamp(0, 1).cpu().permute(0, 2, 3, 1).numpy()
        return ImagePipelineOutput(images=self.numpy_to_pil(result))
