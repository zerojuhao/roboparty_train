# Shared Code And Host Runtime

Repository: https://github.com/zerojuhao/roboparty_train.git

The shared code hosts are 5090D-5, 5090D-6, and H20-96-4. Use the same
commit on all three hosts; do not create three independent commits for the
same change. Commit on one host, push, then fetch and fast-forward the others
after checking that their working trees have no local changes.

## Repository Layout

- `robo_newton/robolab/`: tasks, environments, robot configuration, and utilities.
- `robo_newton/scripts/`: training, playback, deployment, and data preparation.
- `rsl_rl/rsl_rl/` and `rsl_rl/tests/`: learning algorithms and regression tests.
- `project_ops/`: shared operational scripts, host launchers, and documentation.

Host-specific launchers belong in `project_ops/` and are shared on all hosts.
Use the canonical paths above, not old copies in root-level `scripts/`,
`tests/`, or loose Python files.

## Local-Only Content

`.gitignore` excludes IsaacLab installations, virtual environments, datasets,
host-specific data symlinks, logs, checkpoints, videos, caches, and generated
experiment outputs. These remain on each training host. Do not force-add them.
Install IsaacLab separately; this repository does not distribute that SDK.

Ignoring a path does not remove files already tracked by Git. Review
`git diff --cached --name-status` before committing, especially existing
dataset deletions inherited from the earlier repository layout.

## Updating Training Hosts

1. Inspect `git status --short --branch` on all hosts before changing code.
2. Review and test a single integrated version. Avoid changing files while a
   deployment or checksum comparison is in progress.
3. Commit and push that version once. On the other hosts, fetch the same commit
   and use a fast-forward update only when their working trees are clean.
4. Compare `git rev-parse HEAD` and confirm there are no unintended local code
   changes. Identical commits alone do not establish identical working trees.
5. Keep running jobs on their existing code version. Start subsequent jobs from
   the verified version; do not assume running Python processes reload edits.

Do not copy `.git` directories between existing repositories, use force pushes,
or run broad cleanup commands to resolve host differences. Resolve local
changes first, and never delete host datasets or training outputs to align Git.
