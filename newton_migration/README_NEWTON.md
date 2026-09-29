# Newton training

The task code runs on Isaac Lab 3.0 with Newton/MJWarp by default. The Isaac Lab SDK is `/home/zym/roboparty_train/IsaacLab-3.0` on `4090-48-2`. It is pinned to commit `4392f7ed1090ae7d5acaed218f136d353e80d299` on the `release/3.0.0` branch, matching `4090-48-amdin:/home/amdin/Luo1imasi/demo/IsaacLab`; this is a branch snapshot, not a tagged 3.0 GA release. Its Python environment contains Newton 1.6.0, MJWarp 3.12.0, Warp 1.17.0, and the `rsl_rl` directory here contains RSL-RL 5.5.1. The UFO agent and runner follow the working code in the demo; Parkour, SSR, AMP, and attention models adapt the existing RoboLab tasks to the native RSL-RL 5.5 API.

Run from `/home/zym/roboparty_train/newton_migration`:

```bash
PYTHONPATH=. /home/zym/roboparty_train/IsaacLab-3.0/.venv/bin/python \
  scripts/train_newton.py --rl_library rsl_rl --task RP1-Parkour \
  --num_envs 64 --device cuda:0 --logger tensorboard

PYTHONPATH=. /home/zym/roboparty_train/IsaacLab-3.0/.venv/bin/python \
  scripts/play_newton.py --rl_library rsl_rl --task RP1-Parkour-Play \
  --checkpoint /path/to/model.pt --device cuda:0
```

Registered training tasks: `RPO-Flat`, `RPO-Rough`, `RPO-Interrupt`, `RPO-BeyondMimic`, `RPO-Getup-Mimic`, `RPO-AMP`, `RPO-AttnEnc`, `RPO-Parkour`, `RP1-Parkour`, `RP1-SSR`, and `RP1-UFO`. Play variants use the same Newton environment configs.

UFO needs `ufo_data/motions/rp1_lafan/manifest.json` and its referenced NPZ files. The other tasks use the original project's `data` directory.

```bash
PYTHONPATH=. /home/zym/roboparty_train/IsaacLab-3.0/.venv/bin/python scripts/validate_newton_configs.py
PYTHONPATH=. /home/zym/roboparty_train/IsaacLab-3.0/.venv/bin/python \
  scripts/smoke_newton_env.py --task RP1-SSR-Play --num_envs 2 --device cuda:0
```

These short checks establish integration only; policy convergence requires a full training run.
