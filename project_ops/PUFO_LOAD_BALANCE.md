# PUFO Load Balance Investigation

## Current State

- Current run: `2026-10-09_10-48-57_pufo_numa_start_resume_20261009`.
- Launcher PID: 2591370; training rank PIDs: 2591629-2591636.
- Resumed from the phase/replay-fix run's complete `model_230400000.pt`, including
  optimizer state and 5000 replay time steps per rank, without replay warmup.
  All eight manifests and 4512 referenced chunks (269.32 GiB) were checked before
  stopping the first job. The first restart discarded 114688 uncheckpointed
  steps. The subsequent sync/profile control run ended at 231604224, so the
  NUMA comparison restarted from the same checkpoint with a 1204224-step rollback.
  Both control logs and the complete source checkpoint are retained.
- Randomized episode phases, weighted online replay and coalesced normalizer
  synchronization are now active. All-rank profiling at relative phase 4 has
  completed, and training continues afterward.
- This run explicitly enables `PUFO_BIND_GPU_LOCAL_CPUS=1` before training imports
  and allocations. The launcher default remains disabled. The earlier live
  affinity-only experiment was reverted and is distinct from this startup test.

## Evidence

The machine has four CPU NUMA nodes, each serving two RTX 5090 D GPUs. Most GPU
pairs communicate across NUMA nodes; there is no NVLink connection in the
reported topology. Training workers initially allow all 128 logical CPUs.

Both `nvidia-smi topo -p2p r` and `-p2p w` report `CNS` (chipset not supported)
for every distinct GPU pair, including pairs within a NUMA node. Do not assume
GPU peer-to-peer transfers are available. The resumed run's `NCCL_DEBUG=INFO` log
now identifies 32 channel edges as `SHM/direct/direct`. The network plugin's
Socket initialization message does not mean these intra-host edges use network
sockets. No transport policy was forced.

Process-level `nvidia-smi pmon` samples show GPU compute activity in the training
workers, with no reported compute activity in the idle persistent evaluators.
The extra contexts belonging to other training ranks have memory allocated but
no reported compute activity during those samples.

An online A/B/A experiment moved existing training threads to their GPU's local
CPU node, preserving any pre-existing non-overlapping thread restrictions. Each
phase lasted about 120 seconds. No phase overlapped tracking evaluation.

| Phase | Global step range | Updates | Mean FPS | Minimum FPS |
| --- | --- | ---: | ---: | ---: |
| Original affinity | 222093312-222265344 | 22 | 1572.18 | 1540 |
| Local NUMA affinity | 222281728-222437376 | 20 | 1397.35 | 1202 |
| Restored affinity | 222445568-222609408 | 21 | 1437.90 | 1388 |

Binding did not demonstrate a benefit and was reverted. The restored phase did
not recover the initial baseline, so the whole difference cannot be attributed
to affinity alone. GPU 1's utilization mean moved from 83% to 35%, while GPU 3's
moved from 43% to 83%. The identity of the high-utilization GPU is sensitive to
CPU scheduling. Utilization alone does not establish useful compute load;
communication/wait kernels remain a hypothesis until a training trace confirms
their contribution.

Raw per-rank CPU and GPU measurements and original affinities are stored under:

`/data/zym/roboparty_train/logs/rsl_rl/rp1_pufo/2026-10-09_08-34-58_pufo_phase_replayfix_resume_20261009/affinity_aba/`

### Memory Placement Follow-Up

Read-only `/proc/<pid>/numa_maps` and CPU-node inspection after the experiment
found the following anonymous resident-memory placement. These mappings are not
individually identified as replay tensors, but each worker has about 41 GiB of
anonymous memory and several 5-18 GiB mappings.

| Rank | GPU node | Last main-thread CPU node | Dominant anonymous-memory node | GiB on dominant node |
| --- | ---: | ---: | ---: | ---: |
| 0 | 0 | 3 | 3 | 40.29 |
| 1 | 0 | 1 | 1 | 39.44 |
| 2 | 1 | 1 | 1 | 39.46 |
| 3 | 1 | 1 | 1 | 37.15 |
| 4 | 2 | 2 | 2 | 39.07 |
| 5 | 2 | 2 | 2 | 34.13 |
| 6 | 3 | 3 | 3 | 40.52 |
| 7 | 3 | 3 | 3 | 34.23 |

Main threads currently run close to their dominant CPU-memory allocation, but
ranks 0 and 1 are remote from their GPU's NUMA node. Moving threads alone to GPU
locality can make CPU replay gathers remote instead. The previous affinity-only
experiment therefore does not test startup-time CPU plus memory placement. This
later snapshot cannot reconstruct placement during that earlier experiment or
prove that NUMA placement caused its slowdown. Keep the existing policy until
CUDA/host traces justify a separate controlled placement experiment; do not mix
that change into the normalizer synchronization comparison.

## Live Python Stack Sampling

A 60-second, 20 Hz, nonblocking `py-spy` capture sampled the live torchrun tree.
Only the eight training main threads are included below; evaluator, logging and
autograd worker threads are excluded. A nearly overlapping 59-sample GPU poll
shows rank 3 as the high-utilization rank in this window.

| Rank | GPU utilization mean | Main-thread samples | Replay | Backward | Finite checks |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0 | 23.5% | 1062 | 27.0% | 19.9% | 7.4% |
| 1 | 31.7% | 1000 | 25.9% | 19.8% | 8.5% |
| 2 | 33.0% | 1021 | 25.8% | 21.6% | 8.3% |
| 3 | 77.1% | 1086 | 14.8% | 10.1% | 22.9% |
| 4 | 36.0% | 1036 | 24.3% | 22.2% | 8.4% |
| 5 | 34.7% | 1017 | 25.7% | 21.7% | 9.6% |
| 6 | 27.6% | 1028 | 24.4% | 24.2% | 7.2% |
| 7 | 29.1% | 1008 | 26.1% | 25.1% | 6.7% |

Rank 3's most frequent leaf sites include `check_finite_tree`'s scalar `.item()`
(206 samples), creating the context-mixture probability tensor on the GPU (197),
and copying CPU replay reward/context metadata to the GPU (106 and 83).
Other ranks spend more samples gathering CPU replay data and reconstructing depth
history. This supports checking host/device synchronization and rank rendezvous,
not interpreting high GPU utilization as more model computation. A Python stack
at a blocking copy or `.item()` can reflect earlier queued CUDA work; it does not
attribute that elapsed time to the check/copy itself or prove NCCL is responsible.
Do not remove finite checks based on these percentages.

Sampling completed with 109180 samples across all threads and 1259 sampling
errors. Nonblocking sampling has missing/inconsistent-stack risk; these are
approximate sample proportions, not measured CPU self time or CUDA kernel time.
The GPU poll was offset by several seconds, so its mean is contextual rather than
a precisely aligned per-stack correlation.

An additional read-only capability probe, `sudo -n nsys profile
--gpu-metrics-devices=help`, reports every installed GB202 / RTX 5090 D as
unsupported with `Unknown error`. Hardware-counter sampling through this local
Nsight installation is therefore unavailable. This does not invalidate the
separately tested PyTorch CUDA-activity trace path.

Raw stacks:

`/data/zym/roboparty_train/logs/rsl_rl/rp1_pufo/2026-10-09_08-34-58_pufo_phase_replayfix_resume_20261009/python_stacks_20261009_0918.raw`

## Normalizer Synchronization

### Production Baseline Before Activation

A read-only 300.33-second sample, with no CPU-affinity changes or competing GPU
benchmark, covered 58 update phases from 225665024 to 226131968. No tracking
evaluation overlapped. Mean reported FPS was 1567.88, median 1567.5, standard
deviation 18.39, and range 1527-1603. Consecutive phases are not independent, so
this standard deviation is descriptive rather than a confidence interval.

Mean GPU utilizations for ranks 0-7 were respectively 27.47%, 56.73%, 31.67%,
30.34%, 37.06%, 35.57%, 31.51%, and 83.80%. The high-utilization rank shifted from
3 in the earlier stack-capture window to 7 in this window without a configuration
change. Main-thread CPU usage was approximately 80-81% on six ranks, 86% on rank
1 and 92% on rank 7. These counters do not distinguish useful computation from
busy waiting. Compare a matching post-resume steady-state window, excluding
startup, profiling, evaluation and checkpoint I/O; account for changed weights
and replay contents when interpreting any small FPS difference.

Raw phase metrics and GPU samples:

`/data/zym/roboparty_train/logs/rsl_rl/rp1_pufo/2026-10-09_08-34-58_pufo_phase_replayfix_resume_20261009/baseline_presync_20261009/`

### Implementation And Microbenchmark

The actual checkpoint contains 12 floating normalizer buffers with 2074 fp32
elements (8296 bytes). The old implementation calls all-reduce and division for
each buffer, resulting in 192 small collectives per 16-minibatch update phase.
The new implementation groups by device/dtype, reduces one flat tensor, divides,
and copies results back into the original buffers. This reduces the current
PUFO configuration to 16 such collectives per update phase. Integer counters,
normalization formulas, buffer storage and checkpoint keys are preserved.

Validation:

- Two-process Gloo test checks exact equality with legacy reduction, including
  mixed fp32/fp64 groups, local integer counters, repeated updates, original
  storage pointers and loading the normalizer state dict.
- Eight-GPU NCCL benchmark checks exact equality on this machine before timing.
- Six rounds alternate measurement order, 1000 calls per variant per round.
  The production training process shares these GPUs, so this is a bounded
  comparative benchmark, not an isolated estimate of training speedup.

| Round | Legacy ms/call (slowest rank) | Coalesced ms/call (slowest rank) |
| --- | ---: | ---: |
| 0 | 2.05146 | 0.27034 |
| 1 | 1.80275 | 0.36212 |
| 2 | 2.09691 | 0.68203 |
| 3 | 2.43311 | 0.30455 |
| 4 | 2.08721 | 0.20722 |
| 5 | 2.09348 | 0.51213 |

Median: 2.09 ms to 0.33 ms, about 6.3x faster for this synchronization operation.
Extrapolating this local saving gives about 28 ms per update phase. Actual
end-to-end speedup and changes to rank imbalance remain unverified. The earlier
100-call trial had large order-dependent variance and was not used for the
conclusion. Training FPS during the GPU benchmark must not be used as a normal
throughput baseline.

## Training CUDA Trace After Activation

All eight ranks captured their real update at step 230440960, with startup
barrier and pending CUDA work outside the CPU `fbcpr/minibatch_update` marker.
Only device events overlapping that marker are included below. Times are unions
of event intervals within each category, not sums of nested CPU/GPU annotations.

| Rank | Marked update ms | Non-NCCL kernels ms | NCCL kernels ms | All GPU activity ms |
| --- | ---: | ---: | ---: | ---: |
| 0 | 470.94 | 45.88 | 51.51 | 98.27 |
| 1 | 470.96 | 45.95 | 92.12 | 138.94 |
| 2 | 470.86 | 45.94 | 87.59 | 134.40 |
| 3 | 470.96 | 46.20 | 109.73 | 156.80 |
| 4 | 481.95 | 46.24 | 46.55 | 93.68 |
| 5 | 485.42 | 47.24 | 43.53 | 91.65 |
| 6 | 470.95 | 45.88 | 95.93 | 142.70 |
| 7 | 470.93 | 47.67 | 119.77 | 168.43 |

Each rank has eight NCCL all-reduce kernels and about 4050 other kernels. The
first collective's start times span 40.99 ms across ranks while its end times
span just 0.06 ms; the second has a 27.93 ms start spread and 0.35 ms end spread.
The traces share the same time base. The large difference in NCCL kernel duration
therefore includes waiting for later-arriving ranks, not simply different model
compute workloads. Non-NCCL kernel time differs by less than 2 ms. This is direct
evidence for rank rendezvous contributing to utilization imbalance in this
captured update, not a claim that all future imbalance has a single cause.

Collective metadata reports seven gradient reductions followed by one 2074-float
normalizer reduction, confirming that coalescing is active. The startup barrier
is excluded from this count. Full profiler text summaries include GPU annotations
and the startup barrier and must not be added together as physical kernel time.

Profiler overhead makes absolute timings nonrepresentative of ordinary updates.
Early unprofiled throughput in the new run is around 1380 FPS, below the prior
1568 FPS baseline, so there is no demonstrated end-to-end speedup yet. A new
NUMA snapshot shows dominant CPU-memory nodes by rank as [2, 3, 1, 1, 2, 3, 3, 3],
with four ranks concentrated on node 3 and little anonymous memory on node 0.
This restart changed memory placement despite unchanged affinity policy; it is
a confounder to investigate, not proof of the regression's cause.

Trace files and launch log:

`/data/zym/roboparty_train/logs/rsl_rl/rp1_pufo/2026-10-09_10-28-55_pufo_sync_profile_resume_20261009/update_profile_rank<RANK>_230440960.json`

`/data/zym/roboparty_train/logs/launch/pufo_sync_profile_resume_20261009.log`

## Startup CPU Placement Trial

The sync/profile control run's 300.40-second steady-state sample covered 51
updates at 231145472-231555072, with mean 1396.43 FPS (median 1400, range
1359-1438), without evaluation overlap. Artifacts are in that run's
`baseline_postsync_20261009/`. Node 3 held about 175 GB of anonymous memory while
node 0 held only about 4 GB at the subsequent placement inspection.

Added `robo_newton/scripts/bind_gpu_local_rank.py` and the default-off launcher
switch `PUFO_BIND_GPU_LOCAL_CPUS=1`. The wrapper selects each rank's visible GPU,
queries its PCI-local CPUs, intersects with the existing allowed CPU set, binds
before imports, and execs the unchanged training entrypoint. It supports numeric
and UUID GPU selectors and fails on invalid topology or an empty intersection.
No strict memory policy, page migration or global system change is applied.
Seven unit tests, shell syntax checks and an eight-rank topology dry-run passed.

Production trial: `2026-10-09_10-48-57_pufo_numa_start_resume_20261009`, using the
same 230400000 checkpoint, communication optimization, profiling flags and model
settings as the control. A new restart was needed to apply affinity before CPU
replay allocation; 1204224 uncheckpointed control-run steps were rolled back.
Replay restored completely and all eight profiling captures finished.

Observed anonymous-memory locality by rank was 99.65%, 99.72%, 99.81%, 99.81%,
99.80%, 99.80%, 99.98%, and 99.98%. All main-thread CPU affinities matched the
respective GPU-local node, and node-level anonymous memory became approximately
94 GB on each of the four nodes.

The bound run's 300.79-second unprofiled sample covered 54 updates at
230547456-230981632, with no evaluation overlap:

| Same global-step window | Mean FPS | Median FPS | Range |
| --- | ---: | ---: | ---: |
| Unbound sync/profile control | 1378.09 | 1376.0 | 1290-1427 |
| Startup GPU-local CPU binding | 1492.74 | 1491.5 | 1453-1527 |

This is an 8.3% improvement against the matched-step unbound control, but still
4.8% below the earlier phase/replay-fix baseline of 1567.88 FPS. Both restarts
begin from the same checkpoint; simulation/evaluation and floating-point paths
are not bitwise identical. This is one sequential trial, not an A/B/A causal
estimate or a convergence result. Startup CPU placement is promising and remains
enabled for this trial, but is not enabled by default in the launcher.

A second matched-step window, 231145472-231555072 (51 updates), gives 1438.49
FPS for startup binding versus 1396.43 FPS for the unbound control, a smaller
3.0% improvement. The bound range is 1392-1485 FPS. Later instantaneous throughput
is around 1415-1435 FPS, so the first-window 8.3% improvement is not a durable
speedup guarantee. Keep both matched windows in the conclusion instead of
selecting only the more favorable window.

Utilization is still uneven: mean GPU utilization was [30.70, 32.19, 61.04,
25.46, 30.12, 28.68, 30.36, 84.31] percent. The bound-run trace still shows nearly
equal non-NCCL computation (about 46-47 ms), with an unusually long NCCL interval
on rank 7. Binding has not eliminated rank rendezvous or all host-side delays.

Raw bound baseline: current run's `baseline_numa_start_20261009/`. Console log:

`/data/zym/roboparty_train/logs/launch/pufo_numa_start_resume_20261009.log`

### Remaining Host Delays

In the bound trace, rank 7 reaches several collective kernels first and waits
for peers. For the seventh update all-reduce, start times span 76.14 ms while
end times span only 0.18 ms. Rank 7's kernel lasts 77.87 ms and the latest rank
(rank 2) takes 1.88 ms. Other collective start spreads are 11.43, 38.41, 0.34,
33.20, 0.35, 28.89, and 8.00 ms. Host `cudaStreamSynchronize` time on rank 7 is
195.84 ms versus roughly 8.49-37.96 ms on other ranks in this profiled update.
This supports the interpretation that a high-utilization rank can be an early
arriver waiting, not the slowest worker or the one with more compute.

The bound trace also contains 64.8-75.1 ms of main-thread `aten::nonzero` time.
The replay trajectory builder scans a transposed time-major boundary mask. A
CPU-only probe with a 5000x1024 bool mask, one Torch thread, and alternating
variant order found exact equality between `mask.T.nonzero()` and
`mask.T.contiguous().nonzero()`. Four 20-call rounds measured approximately
33.0-33.2 ms versus 11.9-12.3 ms median, including the contiguous copy. This
suggests a small future layout optimization, but trajectory refresh is cached
across the 16 minibatches: the saving would be once per phase, not 16 times.
It is likely less than 1% of total phase time. The production replay code was
not changed for this microbenchmark, and no restart is justified for it alone.

A subsequent 60-second, 20 Hz nonblocking stack capture of normal training
produced 101632 samples across all threads and 1143 sampling errors. Restricting
to main threads, ranks 2 and 7 had 25.7%/25.9% of samples in finite-check paths,
many at `.item()`, and frequently stopped at GPU tensor creation or replay
metadata copies. The other six ranks had about 22-24% in CPU replay paths,
20-25% in backward calls, and 8-12% in finite checks. This is consistent with
the trace's early-arrival synchronization pattern, but is not CPU self-time
attribution. Do not remove finite checks merely because a blocking `.item()`
exposes earlier queued CUDA/NCCL work. Further meaningful overlap would require
replay prefetch/transfer and scheduling work, with a larger validation burden
than the current scoped changes.

Stack artifact: current run's `python_stacks_bound_20261009.raw`.

### Runtime Regression Checks

The bound run completed its scheduled evaluation at 233603072 across all eight
ranks and 902 clips, then continued training. At 233652224, all 116 TensorBoard
scalar tags contained only finite values. The 396 `train/critic_loss` and
`Reward/weighted/total` records were exactly 8192 global steps apart, preserving
one record per update phase. Sparse timeout diagnostics were 385024 steps apart
(the 384000-step threshold rounded to the 8192-step collection grid).
Runner tests were rerun: 35 tests, 34 passed and one GPU opt-in test skipped.
All seven startup-binding tests passed, and `git diff --check` passed.

The final cycle audit at 234676224 found all eight training ranks still running,
no traceback, CUDA error, NCCL warning or child-process failure in the launch
log, and only finite values across all 117 TensorBoard scalar tags. The 521
`train/critic_loss` and `Reward/weighted/total` records remained exactly 8192
steps apart. All 12 timeout-diagnostic records remained 385024 steps apart.

A late 300.61-second sample covered 52 updates from 234119168 to 234536960,
without evaluation overlap. Mean reported FPS was 1430.94 (range 1379-1463).
GPU utilization means were 30.36%, 26.00%, 26.59%, 28.67%, 30.43%, 27.24%,
30.92%, and 80.56%. Thus the early 1492.74 FPS was not sustained, and the
utilization imbalance remains. The earlier matched-window gains of 3.0-8.3%
are limited sequential comparisons, not evidence of a permanent speedup or
an improvement over the older 1567.88 FPS baseline. Raw late measurements are
in the current run's `baseline_numa_late_20261009/` directory.

## Periodic Loss Spikes Before Phase Randomization

The old launch log provides stronger evidence than isolated utilization samples.
In its final 1628 updates (210001920 through 223330304 global steps), the three
largest absolute one-update reward changes are all at phase 2 modulo 500,
relative to that run's resume point 105603072:

| Global step | Absolute reward jump | Absolute auxiliary critic loss jump |
| --- | ---: | ---: |
| 212115456 | 0.0608 | 0.2076 |
| 216211456 | 0.0617 | 0.2316 |
| 220307456 | 0.0645 | 0.2506 |

Each step differs by exactly 4096000 global samples (500 vector steps). This is
consistent with synchronized timeout and the replay sampler admitting the new
short trajectories shortly afterward. It establishes the periodic correlation,
not a controlled separation of timeout and replay-sampling effects.

The phase-randomized run's first 312 logged updates, ending at 223371264, had
maximum one-update reward and auxiliary critic changes of 0.0134 and 0.0708.
This shorter observation window is insufficient to compare full-cycle maxima;
the first full post-resume cycle ends at 224903168. Check through at least phase
12 beyond that boundary (225001472) before assessing recurrence.

That first-cycle gate has now completed. Comparing 25 updates centered on the
last old synchronized boundary (220291072) and the first new boundary (224903168):

| Metric | Old boundary window | New boundary window |
| --- | ---: | ---: |
| Reward maximum one-update jump | 0.0645 | 0.0078 |
| Auxiliary critic maximum one-update jump | 0.2506 | 0.0474 |
| Auxiliary critic loss range | 1.1001-1.7005 | 1.0823-1.1665 |
| Critic maximum one-update jump | 0.6144 | 0.2926 |
| Median reported FPS | 1584 | 1560 |

Windows are 220192768-220389376 and 224804864-225001472. The old concentrated
reward/auxiliary-loss spike did not recur in this first complete new cycle.
This is one boundary, not evidence that every future spike is eliminated or that
GPU imbalance is fixed. The runs differ in weights, replay contents and reset
schedule; their FPS medians are descriptive, not a controlled speed comparison.
The normalizer communication optimization is still not loaded in this live run.

The second post-resume boundary at 228999168 has also been observed through
229097472. Its matching 25-update window (228900864-229097472) had maximum
one-update changes of 0.0153 for reward, 0.1042 for auxiliary critic loss, and
0.2471 for critic loss. Auxiliary critic loss ranged from 1.0851 to 1.2021; the
single value 1.2021 at phase +5 fell to 1.0979 on the next update. Thus isolated
fluctuations remain, but the old sustained rise toward 1.70 did not recur at
either of the first two new cycle boundaries. Median reported FPS in the second
window was 1547 (minimum 1441); this is not a communication-optimization result,
since that change is still inactive in these workers.

The current startup-bound, coalesced-sync run has also passed its first complete
500-control-step cycle at 234496000 and the subsequent 12-update gate at
234594304. Its 25-update boundary window (234397696-234594304) measured:

| Metric | Old boundary window | Current bound-run boundary |
| --- | ---: | ---: |
| Reward maximum one-update jump | 0.0645 | 0.0077 |
| Auxiliary critic maximum one-update jump | 0.2506 | 0.0622 |
| Auxiliary critic loss range | 1.1001-1.7005 | 1.0870-1.1743 |
| Critic maximum one-update jump | 0.6144 | 0.2179 |
| Median reported FPS | 1584 | 1424 |

Across all 511 logged updates from 230416384 through 234594304, not just the
boundary window, maximum one-update changes were 0.0153 for reward, 0.0777
for auxiliary critic loss, and 0.3611 for critic loss. Auxiliary critic loss
ranged from 1.0629 to 1.1743. The old concentrated periodic spike did not recur
in this observed cycle. Ordinary fluctuations remain; this does not establish
long-term convergence or guarantee that future spikes cannot occur. Throughput
and loss comparisons involve different training states and are not controlled
algorithm ablations.

## Reproduction

Run the CPU equivalence tests with the training Python environment:

```bash
PYTHONPATH=rsl_rl python rsl_rl/tests/algorithms/test_normalizer_sync.py -v
```

Run the bounded NCCL benchmark when a brief competing GPU workload is acceptable:

```bash
PYTHONPATH=rsl_rl OMP_NUM_THREADS=1 python -m torch.distributed.run \
  --standalone --nproc_per_node=8 \
  rsl_rl/tests/algorithms/test_normalizer_sync.py \
  --benchmark --iterations 1000 --rounds 6
```

The live affinity experiment is explicitly opt-in and restores original
affinities in a `finally` block after the local-affinity phase:

```bash
python project_ops/scripts/diagnose_rank_affinity.py \
  --parent TRAINING_TORCHRUN_PID --log LAUNCH_LOG \
  --output NEW_MEASUREMENT_DIRECTORY --seconds 120
```

## Profiling And Validation Outcome

The runner now supports an opt-in one-minibatch trace via
`agent.profile_update_phase=4 agent.profile_update_rank=-1`. Rank `-1` profiles
all ranks on the same minibatch; a nonnegative rank selects only that worker.
Capturing all ranks avoids profiling only a fast or slow worker, although each
profiler still adds overhead and can distort the normal waiting distribution.
All-rank capture waits at a barrier after profiler startup, then drains pending
CUDA work before entering the `fbcpr/minibatch_update` marker. The startup barrier
is outside that marker: exclude it when attributing NCCL time to the update. A
single-rank capture does not add this collective. The full text summary includes
startup activities; use the marked interval in the Chrome trace for attribution.
The phase index is
relative to the current invocation (including resume), and profiling is disabled
by default. The capture writes `update_profile_rank<RANK>_<STEP>.json` and a
device-time summary next to the training logs. Exclude that phase from throughput
comparisons: profiler overhead can make peers wait at their normal collectives.
The hook executes the update exactly once and does not add an optimizer step.

1. Activation, all-rank CUDA traces and matched-step throughput comparisons are
   complete. Startup CPU binding improved throughput versus the unbound restart,
   but has not eliminated utilization imbalance or beaten the earlier baseline.
2. Remaining host delays and the outlier NCCL wait have been quantified above.
   No further algorithm or scheduling changes are being added during this run.
3. The full 500-control-step period (4096000 global steps for 8x1024 environments)
   and its following 12 updates have now been checked using every logged update,
   not only sparse reset diagnostics. The old periodic spike did not recur;
   ordinary loss fluctuations and uneven utilization remain.
4. Resume, replay restoration, scheduled evaluation, finite scalar values and
   logging cadence have been verified. The scoped analysis and optimization
   trial are complete; the current training run remains running. Startup binding
   stays opt-in, and no further restart or algorithm change is needed for this
   observation task. Long-term convergence and a reproducible sustained speedup
   remain unproven, not claimed outcomes of this trial.
