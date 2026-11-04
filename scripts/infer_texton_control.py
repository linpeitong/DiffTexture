#!/usr/bin/env python
import argparse

import torch
from diffusers import DDPMScheduler, UNet2DModel
from PIL import Image

from difftexture.texton_control import TextonControlModel, TextonControlPipeline


def main():
    parser = argparse.ArgumentParser(description="Run texton-controlled DDPM inpainting.")
    parser.add_argument("--image", required=True); parser.add_argument("--mask", required=True); parser.add_argument("--texton-map", required=True)
    parser.add_argument("--inpainting-checkpoint", required=True); parser.add_argument("--control-checkpoint", required=True); parser.add_argument("--output", required=True)
    parser.add_argument("--steps", type=int, default=1000); parser.add_argument("--conditioning-scale", type=float, default=1.0); parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(); device = "cuda" if torch.cuda.is_available() else "cpu"
    model = TextonControlModel(UNet2DModel.from_pretrained(args.inpainting_checkpoint))
    model.load_adapter(args.control_checkpoint); model.to(device)
    pipeline = TextonControlPipeline(model, DDPMScheduler(num_train_timesteps=1000, beta_schedule="linear")).to(device)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    output = pipeline(Image.open(args.image), Image.open(args.mask), Image.open(args.texton_map), args.steps, args.conditioning_scale, generator).images[0]
    output.save(args.output)


if __name__ == "__main__":
    main()
