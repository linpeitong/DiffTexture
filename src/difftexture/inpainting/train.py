"""Training loop for the manuscript-consistent masked DDPM objective."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from diffusers import DDPMScheduler, UNet2DModel
from diffusers.optimization import get_scheduler
from torch.utils.data import DataLoader

from difftexture.data import SliceDataset


def build_unet(resolution: int) -> UNet2DModel:
    return UNet2DModel(
        sample_size=resolution, in_channels=3, out_channels=3,
        layers_per_block=2, block_out_channels=(128, 128, 256, 256, 512, 512),
        down_block_types=("DownBlock2D", "DownBlock2D", "DownBlock2D", "DownBlock2D", "AttnDownBlock2D", "DownBlock2D"),
        up_block_types=("UpBlock2D", "AttnUpBlock2D", "UpBlock2D", "UpBlock2D", "UpBlock2D", "UpBlock2D"),
    )


def train(args) -> None:
    accelerator = Accelerator(mixed_precision=args.mixed_precision, gradient_accumulation_steps=args.gradient_accumulation_steps)
    dataset = SliceDataset(args.manifest, args.resolution)
    loader = DataLoader(dataset, batch_size=args.train_batch_size, shuffle=True, num_workers=args.num_workers)
    unet = build_unet(args.resolution)
    scheduler = DDPMScheduler(num_train_timesteps=args.num_train_timesteps, beta_schedule=args.beta_schedule)
    optimizer = torch.optim.AdamW(unet.parameters(), lr=args.learning_rate)
    updates = max(1, args.num_train_epochs * len(loader) // args.gradient_accumulation_steps)
    lr_scheduler = get_scheduler("cosine", optimizer, num_training_steps=updates, num_warmup_steps=args.lr_warmup_steps)
    unet, optimizer, loader, lr_scheduler = accelerator.prepare(unet, optimizer, loader, lr_scheduler)
    global_step = 0
    for _ in range(args.num_train_epochs):
        unet.train()
        for batch in loader:
            with accelerator.accumulate(unet):
                clean = batch["image"]
                mask = batch["mask"]
                noise = torch.randn_like(clean)
                times = torch.randint(0, scheduler.config.num_train_timesteps, (clean.shape[0],), device=clean.device).long()
                # Eq. (5): the context is never noised.
                noisy = scheduler.add_noise(clean, noise, times) * mask + clean * (1 - mask)
                prediction = unet(noisy, times).sample
                loss = F.mse_loss(prediction * mask, noise * mask, reduction="sum") / mask.sum().clamp_min(1)
                accelerator.backward(loss)
                optimizer.step(); lr_scheduler.step(); optimizer.zero_grad()
            global_step += 1
            if accelerator.is_main_process and global_step % args.checkpointing_steps == 0:
                target = Path(args.output_dir) / f"checkpoint-{global_step}" / "unet"
                target.mkdir(parents=True, exist_ok=True)
                accelerator.unwrap_model(unet).save_pretrained(target)
    if accelerator.is_main_process:
        target = Path(args.output_dir) / "unet"
        target.mkdir(parents=True, exist_ok=True)
        accelerator.unwrap_model(unet).save_pretrained(target)
