import numpy as np

from difftexture.procedural.procedural_bottom_up import ProceduralConfig, generate_texton_map


def test_texton_map_is_seeded_and_masked():
    mask = np.zeros((48, 48), dtype=np.uint8); mask[8:40, 8:40] = 1
    first, meta = generate_texton_map(mask, ProceduralConfig(scale_factor=2), seed=7)
    second, _ = generate_texton_map(mask, ProceduralConfig(scale_factor=2), seed=7)
    assert np.array_equal(first, second)
    assert not first[mask == 0].any()
    assert meta["num_textons"] > 0
