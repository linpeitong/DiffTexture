#!/usr/bin/env python
import argparse

from difftexture.inpainting.train import train


def main():
    parser = argparse.ArgumentParser(description="Train context-aware DDPM inpainting.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--train-batch-size", type=int, default=2)
    parser.add_argument("--num-train-epochs", type=int, default=250)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--num-train-timesteps", type=int, default=1000)
    parser.add_argument("--beta-schedule", default="linear")
    parser.add_argument("--checkpointing-steps", type=int, default=10000)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--lr-warmup-steps", type=int, default=500)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--mixed-precision", choices=["no", "fp16", "bf16"], default="fp16")
    train(parser.parse_args())


if __name__ == "__main__":
    main()
