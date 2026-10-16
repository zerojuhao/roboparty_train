# Local Newton Data

This directory is intentionally kept as a path skeleton. Install the private
motion datasets and robot assets here; the files themselves are ignored and
are not published with the source repository.

```text
data/
  robots/
    rp1/                 RP1 URDF, MJCF, meshes, and terrain assets
    rpo/                 RPO URDF, MJCF, meshes, and terrain assets
  motions/newton/
    rp1/                 RP1 AMP, Parkour, and SSR motions
    rpo/                 RPO AMP and Parkour motions
    rp1_upper_run_bounded/
    rp1_lafan/           UFO and PUFO motion manifest plus NPZ clips
```

Set `ROBO_TRAIN_DATA_ROOT` to this directory when using a non-default data
location.
