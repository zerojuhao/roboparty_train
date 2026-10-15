# LAFAN Prefix Trim (2026-10-09)

## Dataset And Backup

- Active directory: `/data/zym/roboparty_train/ufo_data/motions/rp1_lafan`
- Complete original backup: `/data/zym/roboparty_train/ufo_data/motions/rp1_lafan_backup_before_prefix_trim_20261009`
- Per-file audit: active directory's `prefix_trim_report.json`.
- Dataset label: `rp1_lafan_v7_trimmed_prefix_v1`.
- Existing `/home/zym/roboparty_train/ufo_data` symlink and training manifest path are unchanged.

54 of 55 NPZ files had 31-51 initial frames removed (0.62-1.02 seconds at 50 Hz).
Total removal: 2169 frames / 43.38 seconds across files. Total retained frames:
567343, previously 569512. The original files are recoverable from the backup.
`walk1_subject2.npz` is byte-for-byte unchanged, as explicitly requested.
Its first sustained near-neutral waist interval is too late (9.22 seconds) for
a conservative prefix-only repair.

## Selection And Preservation

Select the earliest frame in the first two seconds for which waist roll and yaw
remain within +/-3 degrees for at least 0.5 seconds, with both base and upper
torso Z-axis tilt within 10 degrees of vertical. The 50 Hz check includes 26
samples so the first and last samples span 0.5 seconds. The new frame zero is
near-neutral, not an artificially imposed exact zero or static T-pose.

All six per-frame arrays are sliced at the same index. FPS, dtypes, world
coordinates, velocities and all retained values are unchanged. There is no
interpolation, smoothing, kinematic regeneration, or root recentering. Preserving
the original suffix also preserves all internal frame-to-frame transitions.

NPZ frame counts, hashes and total frames are updated in the manifest. Original
CSV/BVH provenance and retarget statistics describe the untrimmed source, as
explicitly noted in `prefix_trim.source_metadata_note`. Each clip contains its
source hash, trim offset, old/new waist angles and new initial root position.
The tool refuses duplicate trimming and checksum mismatches, validates staging,
and retains the entire original directory before publishing the replacement.

## Validation And Training

- All 55 original hashes match the backup manifest; all 55 current hashes match
  the updated manifest. Every retained array exactly equals its original suffix.
- All arrays are finite with matching frame dimensions and unit quaternions.
- All 54 trimmed files pass the selection criterion at their new frame zero.
- 10 CPU tool tests and four motion-indexing tests pass.
- The actual CPU motion loader loads the training selection of 40 source files
  and passes start, middle, end and out-of-range queries for every window.
- 39 training-selected sources changed; `walk1_subject2` was preserved. The
  10-second window count changes from 902 to 897 because short tails are dropped.

At completion of the data-only trim, training remained suspended and its
processes still held the original dataset in memory. Using the trimmed dataset
requires a new process from the chosen checkpoint, not SIGCONT the old process.
The runner rebuilds expert trajectories and does not save motion sampling
weights in its checkpoint, so there is no saved 902-element priority vector to
restore into 897 windows. Model/optimizer dimensions are unchanged, but old
online replay contains pre-trim experience; restarting with these data is not
an exact continuation of the old experiment. The subsequent GPU resume is
recorded below.

Do not compare per-window evaluation rows by numeric ID across the two dataset
versions: boundaries, tails and some IDs have changed. Keep separate run logs.

## GPU Resume After Cleanup

At the user's subsequent request on 2026-10-09, the suspended training ranks,
evaluation workers and associated logging processes were terminated. Before
relaunch, all eight GPUs reported 0 MiB used and no compute processes. No
dataset, checkpoint or training log was deleted.

The new run restores `model_240001024.pt` from
`2026-10-09_10-48-57_pufo_numa_start_resume_20261009`, including model, optimizer
and online replay (5000 time steps per rank, no warmup). The last old logged
step was 247521280, so 7520256 unsaved steps were discarded. All 4512 replay
chunks were present before restart; actual restoration subsequently succeeded.

- Run: `2026-10-09_15-28-40_pufo_trimmed_prefix_resume_20261009` (directory timestamp is host UTC).
- Run root: `/data/zym/roboparty_train/logs/rsl_rl/rp1_pufo/`.
- Launch log: `/data/zym/roboparty_train/logs/launch/pufo_trimmed_prefix_resume_20261009.log`.
- tmux session: `rp1-pufo-trimmed-resume`.
- W&B run ID: `yugij2lr`.
- Configuration: 8 ranks x 1024 environments; startup CPU binding enabled;
  temporary update profiler disabled; explicitly uses the active trimmed manifest.
- Dataset provenance snapshots: `dataset_manifest.json` and
  `dataset_prefix_trim_report.json` in the new run directory.

The first evaluation over 897 windows completed on all eight ranks. Ten logged
updates reached 240091136 global steps; all 116 TensorBoard scalar tags were
finite, and train/Reward entries remained 8192 steps apart. No traceback, CUDA
error, NCCL warning or child-process failure appeared in the launch log.
Expert trajectories were rebuilt from the trimmed files;
the restored online replay still contains earlier experience and is not silently
relabelled as newly collected data. This is a successful startup/resume check,
not a long-term stability or convergence claim.

## Tool

`robo_newton/scripts/tools/retarget/trim_lafan_prefix.py` defaults to inspection.
Publishing requires `--apply --backup NEW_SIBLING_DIRECTORY`. The existing backup
must never be overwritten. Tests are in `test_trim_lafan_prefix.py` alongside it.
