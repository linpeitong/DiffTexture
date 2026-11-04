from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from diffusers import DDPMScheduler, UNet2DModel
from torch.utils.data import DataLoader

from difftexture.data import SliceDataset
from .model import TextonControlModel


def train(args) -> None:
    accelerator = Accelerator(mixed_precision=args.mixed_precision, gradient_accumulation_steps=args.gradient_accumulation_steps)
    data = SliceDataset(args.manifest, args.resolution, with_texton=True)
    loader = DataLoader(data, batch_size=args.train_batch_size, shuffle=True, num_workers=args.num_workers)
    unet = UNet2DModel.from_pretrained(args.inpainting_checkpoint)
    model = TextonControlModel(unet)
    scheduler = DDPMScheduler(num_train_timesteps=args.num_train_timesteps, beta_schedule=args.beta_schedule)
    optimizer = torch.optim.AdamW(model.condition_encoder.parameters(), lr=args.learning_rate)
    model, optimizer, loader = accelerator.prepare(model, optimizer, loader)
    step = 0
    for _ in range(args.num_train_epochs):
        model.train()
        for batch in loader:
            with accelerator.accumulate(model):
                clean, mask, texton = batch["image"], batch["mask"], batch["texton_map"]
                noise = torch.randn_like(clean)
                timesteps = torch.randint(0, scheduler.config.num_train_timesteps, (clean.shape[0],), device=clean.device).long()
                noisy = scheduler.add_noise(clean, noise, timesteps) * mask + clean * (1 - mask)
                prediction = model(noisy, timesteps, texton)
                loss = F.mse_loss(prediction * mask, noise * mask, reduction="sum") / mask.sum().clamp_min(1)
                accelerator.backward(loss); optimizer.step(); optimizer.zero_grad()
            step += 1
            if accelerator.is_main_process and step % args.checkpointing_steps == 0:
                path = Path(args.output_dir) / f"checkpoint-{step}"; path.mkdir(parents=True, exist_ok=True)
                accelerator.unwrap_model(model).save_pretrained(str(path))
    if accelerator.is_main_process:
        path = Path(args.output_dir); path.mkdir(parents=True, exist_ok=True)
        accelerator.unwrap_model(model).save_pretrained(str(path))
