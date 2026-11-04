# Reproducibility

Use the YAML files in `configs/` as the paper-facing starting points. Record
the manifest revision, config, package versions, checkpoint hash, and random
seed for every run. For procedural maps, the output sidecar JSON contains the
seed and sampled spacing radius.

The released main branch follows the manuscript's context-aware formulation:
noise is confined to the inpainting mask during training and the unmasked
context remains clean during every sampling step. This is intentionally named
`ddpm_inpainting`, not the former experimental `copy_ddpm` name.
