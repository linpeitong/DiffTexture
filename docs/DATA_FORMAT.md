# Data format

Each manifest line is a JSON object with `id`, `image`, and `mask`. Texton
control additionally requires `texton_map`. Paths are resolved relative to
the JSONL file, so a manifest can be moved with its data directory.

The loader accepts PNG, TIFF, JPEG, and other Pillow-readable formats. CT
intensities must already be windowed and scaled into an 8-bit image. A
grayscale CT slice is replicated to three channels because the DDPM U-Net has
three input channels. Mask values greater than zero indicate editable pixels.

No patient images, masks, texton maps, pseudo-labels, or SAM outputs are
distributed in this repository. Users are responsible for permissions,
de-identification, and all preprocessing of their own data.
