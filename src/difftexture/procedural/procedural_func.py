"""Geometry primitives for bottom-up procedural texton generation."""

from __future__ import annotations

import math

import numpy as np


def poisson_disk_sampling(mask: np.ndarray, radius: float, candidates: int, rng: np.random.Generator) -> list[tuple[float, float]]:
    """Bridson Poisson-disk samples constrained to non-zero mask pixels."""
    if radius <= 0:
        raise ValueError("radius must be positive")
    height, width = mask.shape
    valid = np.argwhere(mask > 0)
    if not len(valid):
        return []
    cell = radius / math.sqrt(2)
    grid: dict[tuple[int, int], tuple[float, float]] = {}
    y, x = valid[rng.integers(len(valid))]
    first = (float(x), float(y)); points = [first]; active = [first]
    grid[(int(x // cell), int(y // cell))] = first
    while active:
        index = int(rng.integers(len(active))); px, py = active[index]; accepted = False
        for _ in range(candidates):
            angle = rng.uniform(0, 2 * math.pi); distance = rng.uniform(radius, 2 * radius)
            x, y = px + distance * math.cos(angle), py + distance * math.sin(angle)
            if not (0 <= x < width and 0 <= y < height) or mask[int(y), int(x)] == 0:
                continue
            gx, gy = int(x // cell), int(y // cell)
            neighbours = (grid.get((ix, iy)) for ix in range(gx - 2, gx + 3) for iy in range(gy - 2, gy + 3))
            if any(point is not None and math.dist((x, y), point) < radius for point in neighbours):
                continue
            point = (x, y); points.append(point); active.append(point); grid[(gx, gy)] = point; accepted = True
            break
        if not accepted:
            active.pop(index)
    return points


def polygon_vertices(center: tuple[float, float], radius: float, min_vertices: int, max_vertices: int, rng: np.random.Generator) -> list[tuple[float, float]]:
    count = int(rng.integers(min_vertices, max_vertices + 1))
    angles = np.sort(rng.uniform(0, 2 * math.pi, count))
    radii = np.clip(rng.normal(radius * 0.75, max(radius * 0.25, 1e-6), count), radius * 0.3, radius)
    return [(center[0] + r * math.cos(a), center[1] + r * math.sin(a)) for a, r in zip(angles, radii)]
