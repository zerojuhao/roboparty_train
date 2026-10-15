# UFO / PUFO Replay Checkpoints

`train.py` saves rank-local training replay by default. The CLI flags are
`--checkpoint-replay` and `--no-checkpoint-replay`. The equivalent Hydra
override is `agent.checkpoint_replay=true` or `false`.

```bash
bash start_local_pufo_8gpu.sh --checkpoint-replay --run_name fresh_replay
bash start_local_pufo_8gpu.sh --no-checkpoint-replay --run_name weights_only
bash start_local_pufo_8gpu.sh --resume --load_run RUN_DIRECTORY --checkpoint model_9601024.pt
```

Each model checkpoint references a unique `replay_STEP_ID/` directory, with
one `rank_N/` shard per training rank. All ranks participate in checkpointing.
The model file is atomically published only after every shard finishes.
Tensor data is saved in bounded chunks without duplicating the entire replay
in memory; unused capacity is not written. Replay is not uploaded to W&B.

`agent.replay_checkpoint_keep=1` retains only the latest replay snapshot in
the current run directory. Increase it to keep more. Historical model files
are retained; resuming a model whose replay was pruned uses warmup instead.
Allow space for the old and new snapshots to coexist while saving.

Resume restores available, complete rank-local replay regardless of whether
future replay saving is enabled. Episode boundaries are closed so samples
cannot join the old simulator state to the new one. Rank count, environment
count, replay capacity, keys, cursor and chunk coverage are validated.
Corrupt or incompatible snapshots fail loudly. Missing snapshots or rank
manifests cause every rank to use empty replay and warm up together.

With empty replay, `agent.resume_replay_warmup_steps=10240` controls the
number of policy rollout samples collected per rank before optimizer updates
resume. This is not a global step count or a number of gradient updates.
Warmup uses the loaded policy, not fresh random actions. A minimum of three
vector steps is enforced for a valid transition after the reset boundary.
Model, optimizer and global step counters are retained throughout warmup.

Replay snapshots do not restore simulator state or random-number generator
state, so resume is not a bit-for-bit continuation of an uninterrupted run.

PUFO uses compact uint8 depth replay: each transition stores the newest delayed
frame, and sampling reconstructs the eight-frame stack with the configured
history stride. The camera delay must stay fixed within each episode. Reset
padding repeats the first frame of that episode; normal resets and resume cannot
join camera histories across resets. Isolated tracking evaluation does not reset
the training simulator or insert replay boundaries. Training observations, action
history, depth history, rollout context and unfinished episode rewards continue
across evaluation. Motion/expert priorities still update after evaluation and
take effect on subsequent sampling. Online replay sampling probabilities are
unchanged. After ring-buffer overwrite, the oldest
35 time steps are excluded from sampling because their full history is gone.
This reduces depth storage by 8x (about 24.7 GiB to 3.1 GiB per rank with
5,120,000 transitions and 18x36 frames), excluding other observation tensors.
Snapshots validate compact mode and history offsets before loading.

Reset-boundary replay uses snapshot version 3. Action outcome flags are stored
with the action that produced them. Ordinary next observations still share the
following record, while pre-reset terminal observations (including their full
depth and proprioceptive histories) are stored only at reset boundaries.
The reset observation remains the next episode's policy input. Every complete
transition, including the last action before a reset and the first action after
one, is eligible for sampling; sequence windows cannot cross episode boundaries.
Terminal observations and their validity masks are saved and validated with
rank-local replay chunks.

Resuming a model with version 1/2 replay keeps its weights, optimizer and step
counters, but ignores that legacy replay and collects fresh data during the
configured warmup. Lost pre-reset outcomes cannot be reconstructed from the
old replay. Model architecture and the v5 observation/reward contract remain
compatible; the replay format changed independently of the model format.

The current PUFO training contract is `pufo_h200_alignment_v6`. V6 adds the
waist-yaw/shoulder/hip incoming-impact penalty (without static-load penalty), so V5 and older replay has
incompatible auxiliary reward targets. Model architecture is unchanged, but
the runner rejects loading older contracts under V6; start a fresh run or use
the original saved configuration and code to continue those checkpoints.
V4 introduced
ground-relative root height, the corrected terrain sign, and compact depth.
V5 changes contact rewards and adds global-map boundary truncation, so V4
replay still contains incompatible reward targets. V3 and older snapshots also
contain world-height and opposite-sign terrain inputs. Start a fresh run or use
the original code and configuration to continue those snapshots; the runner
rejects loading them under the current contract rather than silently mixing contracts.
