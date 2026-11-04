"""Bottom-up, Poisson-disk procedural generation of texton maps."""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
from PIL import Image, ImageDraw

from .procedural_func import poisson_disk_sampling, polygon_vertices


@dataclass(frozen=True)
class ProceduralConfig:
    scale_factor: int = 5
    poisson_radius_mean: float = 33.0
    poisson_radius_std: float = 14.0
    poisson_radius_min: float = 15.0
    poisson_radius_max: float = 55.0
    poisson_candidates: int = 30
    vertex_radius_mean_scale: float = 0.75
    vertex_radius_std_scale: float = 0.25
    min_vertices: int = 5
    max_vertices: int = 10


def generate_texton_map(mask: np.ndarray, config: ProceduralConfig = ProceduralConfig(), seed: int | None = None) -> tuple[np.ndarray, dict]:
    """Generate an instance-label texton map inside a binary lesion mask."""
    if mask.ndim != 2:
        raise ValueError("mask must be a two-dimensional array")
    if config.scale_factor < 1:
        raise ValueError("scale_factor must be at least one")
    rng = np.random.default_rng(seed)
    height, width = mask.shape
    high_mask = np.asarray(Image.fromarray((mask > 0).astype(np.uint8) * 255).resize((width * config.scale_factor, height * config.scale_factor), Image.Resampling.NEAREST)) > 0
    radius = float(np.clip(rng.normal(config.poisson_radius_mean, config.poisson_radius_std), config.poisson_radius_min, config.poisson_radius_max))
    points = poisson_disk_sampling(high_mask, radius, config.poisson_candidates, rng)
    labels = Image.new("I", (high_mask.shape[1], high_mask.shape[0]), 0); draw = ImageDraw.Draw(labels)
    for label, point in enumerate(points, start=1):
        polygon = polygon_vertices(point, radius, config.min_vertices, config.max_vertices, rng)
        draw.polygon(polygon, fill=label)
    labels = np.asarray(labels).copy()
    labels[~high_mask] = 0
    # Preserve instance boundaries by downsampling nearest-neighbour.
    output = np.asarray(Image.fromarray(labels.astype(np.int32), mode="I").resize((width, height), Image.Resampling.NEAREST), dtype=np.int32)
    return output, {"seed": seed, "poisson_radius": radius, "num_textons": len(points), "config": asdict(config)}
