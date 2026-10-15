# Newton training

PUFO and parkour now share the protected MuJoCo Warp backend. Install the
validated physics dependencies with
`python -m pip install -r robo_newton/requirements-physics.txt`
from the repository root. Startup rejects
unvalidated dependency/source changes. See [GUARDED_PHYSICS.md](GUARDED_PHYSICS.md)
for compatibility checks, task scope and upgrade tests.

The task code runs on Isaac Lab 3.0 with Newton/MJWarp by default. The Isaac Lab SDK is `/home/zym/roboparty_train/IsaacLab-3.0` on `4090-48-2`. It is pinned to commit `4392f7ed1090ae7d5acaed218f136d353e80d299` on the `release/3.0.0` branch, matching `4090-48-amdin:/home/amdin/Luo1imasi/demo/IsaacLab`; this is a branch snapshot, not a tagged 3.0 GA release. Training uses the Conda environment `isaaclab_zym`, with Newton 1.6.0, MJWarp 3.12.0, Warp 1.17.0, Torch 2.11.0+cu128, and RSL-RL 5.5.1. Entering `/home/zym/roboparty_train` automatically activates this environment in an interactive Bash shell.

The Newton project lives under `/home/zym/roboparty_train/robo_newton`, following the RoboLab layout. Its `data` directory is a real directory, not a symlink. Previous legacy source trees are archived under `archives/`; the repository-level `rsl_rl` directory remains a shared local dependency.

Run from `/home/zym/roboparty_train/robo_newton`:

```bash
source /data/conda/etc/profile.d/conda.sh
conda activate isaaclab_zym
PYTHONPATH=. python \
  scripts/train_newton.py --rl_library rsl_rl --task RP1-Parkour \
  --num_envs 64 --device cuda:0 --logger tensorboard

PYTHONPATH=. python \
  scripts/play_newton.py --rl_library rsl_rl --task RP1-Parkour-Play \
  --checkpoint /path/to/model.pt --device cuda:0
```

Registered training tasks: `RPO-Flat`, `RPO-Rough`, `RPO-Interrupt`, `RPO-BeyondMimic`, `RPO-Getup-Mimic`, `RPO-AMP`, `RPO-AttnEnc`, `RPO-Parkour`, `RP1-Parkour`, `RP1-SSR`, and `RP1-UFO`. Play variants use the same Newton environment configs.

UFO needs `data/motions/rp1_lafan/manifest.json` and its referenced NPZ files. UFO, AMP, Parkour, and SSR motion assets all live under the migrated `data` directory.

```bash
PYTHONPATH=. python project_ops/scripts/validate_newton_configs.py
PYTHONPATH=. python \
  scripts/smoke_newton_env.py --task RP1-SSR-Play --num_envs 2 --device cuda:0
```

These short checks establish integration only; policy convergence requires a full training run.

## RP1 AMP Resume Defaults

Run 5000 additional iterations unless a different length is explicitly requested.
Save checkpoints every 500 iterations and run
`project_ops/scripts/watch_amp_checkpoint_evaluations.py` alongside training for
the baseline, each 500-iteration checkpoint, and the final checkpoint. Include
`--turn-tests` to retain the fixed straight-running, stopping, and turning checks.
Resume preserves the selected checkpoint's curriculum and optimizer state;
reward and optimizer changes must be explicitly chosen for each experiment.

To extend an already running short run, `project_ops/scripts/queue_amp_resume.py`
waits for its normal completion and resumes its final checkpoint with the same
training arguments. It launches both training and checkpoint evaluation in tmux.
