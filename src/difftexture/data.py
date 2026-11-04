"""Manifest-backed data loading shared by all release entrypoints."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable



def read_manifest(path: str | Path, required: Iterable[str]) -> list[dict]:
    path = Path(path)
    required = set(required)
    records: list[dict] = []
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        record = json.loads(line)
        missing = required.difference(record)
        if missing:
            raise ValueError(f"{path}:{line_number} is missing: {sorted(missing)}")
        for key in required.intersection({"image", "mask", "texton_map"}):
            record[key] = str((path.parent / record[key]).resolve())
        records.append(record)
    if not records:
        raise ValueError(f"Manifest is empty: {path}")
    return records


class SliceDataset:
    """CT slice dataset with an explicit, portable JSONL contract."""

    def __init__(self, manifest: str | Path, resolution: int, with_texton: bool = False):
        # Keep manifest/procedural tooling usable in environments without the
        # training stack installed.
        from PIL import Image
        from torchvision import transforms

        fields = ("id", "image", "mask", "texton_map") if with_texton else ("id", "image", "mask")
        self.records = read_manifest(manifest, fields)
        self.with_texton = with_texton
        self.image_transform = transforms.Compose(
            [transforms.Resize((resolution, resolution)), transforms.ToTensor(), transforms.Normalize([0.5] * 3, [0.5] * 3)]
        )
        self.mask_transform = transforms.Compose(
            [transforms.Resize((resolution, resolution), interpolation=transforms.InterpolationMode.NEAREST), transforms.PILToTensor()]
        )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        from PIL import Image

        item = self.records[index]
        image = self.image_transform(Image.open(item["image"]).convert("RGB"))
        mask = (self.mask_transform(Image.open(item["mask"]).convert("L")) > 0).float()
        output: dict[str, torch.Tensor | str] = {"id": item["id"], "image": image, "mask": mask}
        if self.with_texton:
            # Instance labels are represented as an RGB control image, preserving non-zero geometry.
            output["texton_map"] = self.image_transform(Image.open(item["texton_map"]).convert("RGB"))
        return output
