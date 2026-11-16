# Local Newton Data

This directory is the runtime data root for the Newton training tasks. The
unified motion datasets and robot assets are included with the source
repository and should remain at these paths.

```text
data/
  robots/
    rp1/                 RP1 URDF, MJCF, meshes, and terrain assets
    rpo/                 RPO URDF, MJCF, meshes, and terrain assets
  motions/
    rp1/                 RP1 AMP, Parkour, and SSR motions
    rpo/                 RPO AMP and Parkour motions
    rp1_upper_run_bounded/
    rp1_lafan/           UFO and PUFO self-describing NPZ clips, directly in this directory
```

Set `ROBO_TRAIN_DATA_ROOT` to this directory when using a non-default data
location.
