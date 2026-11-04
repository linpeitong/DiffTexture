#!/usr/bin/env python
import argparse
import json
from dataclasses import fields
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

from difftexture.data import read_manifest
from difftexture.procedural.procedural_bottom_up import ProceduralConfig, generate_texton_map


def main():
    parser = argparse.ArgumentParser(description="Generate procedural texton maps from mask images.")
    parser.add_argument("--manifest", required=True); parser.add_argument("--output-dir", required=True)
    parser.add_argument("--config"); parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    values = yaml.safe_load(Path(args.config).read_text()) if args.config else {}
    allowed = {field.name for field in fields(ProceduralConfig)}
    config = ProceduralConfig(**{k: v for k, v in values.items() if k in allowed})
    output_dir = Path(args.output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    for offset, record in enumerate(read_manifest(args.manifest, ("id", "mask"))):
        mask = np.asarray(Image.open(record["mask"]).convert("L"))
        texton_map, metadata = generate_texton_map(mask, config, args.seed + offset)
        Image.fromarray(texton_map.astype(np.uint16)).save(output_dir / f"{record['id']}.png")
        (output_dir / f"{record['id']}.json").write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()
