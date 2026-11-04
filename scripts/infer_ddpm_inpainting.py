#!/usr/bin/env python
import argparse

import torch
from diffusers import DDPMScheduler, UNet2DModel
from PIL import Image

from difftexture.inpainting import DDPMInpaintingPipeline


def main():
    parser = argparse.ArgumentParser(description="Run context-aware DDPM inpainting.")
    parser.add_argument("--image", required=True); parser.add_argument("--mask", required=True)
    parser.add_argument("--checkpoint", required=True); parser.add_argument("--output", required=True)
    parser.add_argument("--steps", type=int, default=1000); parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    unet = UNet2DModel.from_pretrained(args.checkpoint).to(device)
    pipe = DDPMInpaintingPipeline(unet, DDPMScheduler(num_train_timesteps=1000, beta_schedule="linear")).to(device)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    result = pipe(Image.open(args.image), Image.open(args.mask), num_inference_steps=args.steps, generator=generator).images[0]
    result.save(args.output)


if __name__ == "__main__":
    main()
