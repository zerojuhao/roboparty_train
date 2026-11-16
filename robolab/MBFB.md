# RP1 FB Sequence Value With MBFB Dynamics

Default RP1-UFO world-model algorithm as of 2026-10-30. Reference:
[Xuewang01/MBFB a25f2b05](https://github.com/Xuewang01/MBFB/tree/a25f2b05e720fcdf8550e8a21bf951027ac2f333),
including the corrected H32 return and October 29 model-gradient cap.
RP1-UFO uses a smaller dynamics network and imagination budget, with a mandatory
lightweight sequence value. This is an RP1 performance adaptation, not the
reference repository's exact optimizer schedule.
Imagination starts after 500 successful dynamics updates and three consecutive
H32 validation passes. The detached value critic then receives 100 value-only
updates before its terminal value contributes to the actor gradient.
The online collect-fit-imagine loop follows
[SuperMimic Algorithm A.1](https://arxiv.org/pdf/2610.28480). Native/model gradient
mixing and the FB rewards remain local adaptations; this is not a reproduction
of the paper's full architecture, batch budget or reward function.
Tracking uses reference-derived B latents, not an explicit tracking reward.
[BFM-Zero 05ad6119](https://github.com/LeCAR-Lab/BFM-Zero/blob/05ad6119a25b06c341c191a72ccf936e987f59cf/humanoidverse/agents/fb/model.py#L156)
implements tracking inference as a projected mean of future B embeddings.
The learned sequence value here is our model-based extension, not a claim that
BFM-Zero itself trains this additional network.

## Configuration

| Setting | Default |
| --- | --- |
| `objective` | `mbfb` |
| `hidden_dims` | `[512, 512, 512]`, LayerNorm + SiLU |
| `window_length`, `batch_size` | 8, 1024 |
| `update_interval` | 2 agent updates per dynamics fit |
| `imagination_interval`, `imagination_batch_size` | 4 agent updates, 256 trajectories |
| `horizon` | 32 |
| `learning_rate` | 0.0002 |
| `validation_interval`, `validation_batch_size` | 200 agent updates, 1024 |
| `forward_horizon`, `forward_weight` | 4, 1.0 |
| `actor_weight`, `actor_grad_ratio` | 0.5, 1.0 |
| `tracking_coef` | 0; explicit pose loss is rejected on this path |
| `value_hidden_dims`, `sequence_context_length` | `[256, 128]`, 10 ordered future z values |
| `model_warmup_updates`, validation gate | 500 successful dynamics updates, then 3 consecutive H32 validation passes with model/baseline ratio <= 0.8 |
| `value_lambda`, value warmup/fit epochs | 0.95; 100 value-only updates; 3 critic epochs per detached rollout |
| `value_tau` | 0.01; EMA target sequence critic for lambda targets |
| `svg_action_mode` | `mean` |
| `actor_max_grad_norm` | 1000 after distributed averaging |
| `replay_steps` | 1000 vector-environment steps on each GPU |
| `compile_geometry` | true; compile the pure geometry observation inside the FP32 precision boundary |
| `cuda_graph_imagination` | true; capture and replay SVG forward/backward on CUDA |
| `cuda_graph_dynamics` | true; capture and replay the eight-step dynamics loss/gradients on CUDA |
| `cuda_graph_forward` | true; replay the short fixed-z F-target prediction on CUDA |
| `cuda_graph_value` | true; replay sequence-value targets and accumulated gradients on CUDA |

Both intervals count **agent updates**, not successful fits. The existing runner
still performs 16 agent updates per 1024 locally collected transitions: eight
world-model fits and four SVG updates per collection phase once windows exist.
Each phase imagines 32,768 actor transitions instead of the previous 131,072.
Real-data fit batch/window/frequency, H32, rewards and precision are unchanged.
This RP1-UFO override does not change the shared `UFOWorldModelCfg` or PUFO defaults.
Dynamics fits run as one full batch without activation checkpointing. The legacy
weight ramp, microbatch and absolute-p95 settings still configure only the
explicit legacy objectives; MBFB uses its dedicated warmup, validation gate,
value warmup, value epochs and covariance ridge.

To reproduce the previous larger budget, pass these Hydra overrides:

```bash
'agent.world_model.hidden_dims=[1024,1024,1024]' \
agent.world_model.imagination_batch_size=512 agent.world_model.imagination_interval=2
```

The smaller network requires a new training run or a matching 512-wide checkpoint.
To resume an existing 1024-wide world model, explicitly restore its `hidden_dims`;
the original budget above can also be restored. Checkpoint contract validation
continues to reject incompatible network shapes rather than discarding learned weights.

## Dynamics And Schedule

The physical state contains root position, xyzw quaternion, world-frame link-origin
linear/angular velocities, 24 joint positions/velocities and 24 executed actions
(85 numbers). The three-layer MLP predicts a residual over constant-velocity
integration. Features/integration and parameters are FP32; CUDA MLP forwards use
BF16. Fitting uses normalized autoregressive error over eight transitions.
Increment variances are averaged across ranks so all replicas use the same scale.

For controlled precision comparisons, set both network autocast switches:

| Mode | `agent.use_bf16_amp` | `agent.world_model.use_bf16_amp` | `agent.matmul_precision` |
| --- | --- | --- | --- |
| BF16 (default) | true | true | high |
| TF32 | false | false | high |
| Strict FP32 | false | false | highest |

The dynamics switch defaults to true for older configurations. It does not alter
checkpoint tensor shapes. Precision is chosen at startup; changing it requires
fresh compilation/CUDA graph capture. Imagined discriminator rewards use the
native FP32 logit conversion even when network forwards use BF16. Precision
switches do not change the model/value gates or fitting schedule.

A dedicated GPU ring contains only physical transitions and restart flags.
The main training replay resides in CPU RAM; only sampled minibatches move to
the training GPU. Its default capacity is 5,120,000 transitions per rank, allocated
on the first append or checkpoint restore. Raw motion frames also reside in CPU
RAM. Queries gather their two interpolation endpoints into a pinned batch and
transfer that batch to the GPU, preserving FP32 data and continuous-time SLERP.
The expert sampler stores only clip metadata, priorities and phase weights. It
reuses the existing trajectory sampler and builds observations from the requested
frames, instead of materializing an additional full expert-observation dataset.
Clip overlap weights, current/next alignment, short-clip padding and XYZ commands
retain their existing semantics. Latent buffers, online histories, physical replay
and active imagination graphs remain on GPU. Dataset-size-dependent GPU storage
is limited to clip metadata; batch memory still depends on the requested window.

This follows the host-storage/batch-transfer approach in
[MimicLite's replay](https://github.com/EGalahad/mimic-lite/blob/3253810ce21dbcc05d8f4942a017ea3eaffc6cf1/mimic_lite_learning/offpolicy/buffer.py)
and its
[any4hdmi windowed motion loader](https://github.com/EGalahad/any4hdmi/blob/bf83602f36ff6fc9796367cfc2b707a2526e1a54/src/any4hdmi/dataset/windowed.py).
The local path performs on-demand queries without a background prefetch thread.
It loads NPZ files into RAM at startup, not from disk on every batch; host RAM still
scales with dataset size. Missing-body-track FK can still use temporary GPU memory
for one clip during loading. Checkpoint chunks and lossless history reconstruction
retain their existing formats.
Environment indices divisible by 16 form the held-out lanes. Validation compares
the **final eighth-step mean squared error** against constant-velocity prediction
on the same windows. MBFB requires three consecutive H32 model/baseline passes
below 0.8 after the 500-update warmup; the gate is collective across ranks.
On each scheduled update, fit dynamics first, then construct F targets and
differentiate the imagined actor return, then fit the sequence value on that
rollout. A nonfinite dynamics fit is skipped collectively, along with that
update's model targets and SVG. The next scheduled fit can resume the full
sequence immediately. `wm/fit_succeeded` reports numerical fit success, while
`wm/model_ready` reports the validation gate.

Without sufficient replay windows or a successful fit in the current update,
imagination sampling, tracking encoding and SVG are skipped.
The old absolute H32 p95 and action-distance gates are not used. Terminal value
evaluation and value fitting start after the model gate; the actor model weight
remains zero for the first 100 value updates.

## F And Actor Objectives

F retains its real one-step objective and adds the reference four-step successor
measure objective. Imagined B embeddings and terminal target F are detached;
the extra term trains only online F. Terminated rows retain the real first-state
diagonal term and omit imagined future rewards and bootstrap.

SVG differentiates 32 steps through frozen dynamics and B. Its implicit FB reward
is `B(s[t+1])^T Cov(B(real_next_state))^-1 z[t]`. The covariance uses the real
update batch and online B. The existing discriminator regularizer is evaluated
at current states t=0..H-1. The t=0 term has no action derivative, since s0 and z0
are detached. There is no posture/position/velocity reference-error reward in SVG.
The discriminator and real-data reward/critic training remain unchanged.

Terminal values come from a `[256,128]` MLP conditioned on local physical state,
the actor's normalized observation/history, environment context, body-relative
commands and **K ordered future z values**. Separate scalar outputs estimate the
implicit FB return and the discriminator return; actor mixing retains the
existing detached FB return scale. With `discriminator_reward=false`, only the FB
head is built. Sequence value itself is mandatory and has no enable/disable switch.

Each SVG trajectory supplies detached features and rewards for three value
optimizer epochs. Lambda-return targets use an EMA target sequence-value network
with `value_tau=0.01`, held fixed across those epochs. The output layer starts at zero. The actor
uses the H-step reward sum plus the terminal estimate from the first rollout.
The network parameters are frozen during actor differentiation, while
terminal-state derivatives still reach the policy. Value fitting runs after the
actor backward, in batches of at most `batch_size=1024`, with one global gradient
average and norm-1 clip per epoch. Nonfinite value updates are skipped on all ranks.

Only detached base features, z and rewards are stored for value fitting. Ordered
windows are materialized per minibatch, avoiding a full B x H x K feature copy.
No additional dynamics rollout or backward through dynamics is used for fitting V.

Tracking rows are those selected by the ordinary expert-latent relabeling, not
all trajectories previously collected in tracking mode. Fresh contiguous expert
windows produce z[0..H+K-1], so the terminal value receives a complete future
window rather than padding with the final z. Only selected rows are sampled and
encoded. Heading and XYZ reference velocity advance with the same sampled expert
transitions, including the command at the terminal state. Non-tracking rows keep
a fixed z and commands. Reference poses never enter the loss.
K-step conditioning is still a finite-lookahead approximation; it does not expose
the entire remaining motion to V.

Let main be the original FB/discriminator/context gradient and aux the original
auxiliary-reward gradient. The full original actor gradient is the cap reference.
The update is `(1-c)*main + aux + c*capped_model`, where `c=actor_weight`.
Model gradients are capped against the unscaled reference norm.
RP1's existing supervised velocity-estimator loss retains full weight with aux.
The actor passes named loss terms to MBFB, which builds main and aux directly.
This avoids traversing auxiliary-only branches with zero gradients when main is
formed by subtracting aux from a combined loss. Updates without imagination
backpropagate the sum of all real loss terms once. Estimator diagnostics reduce
XYZ together; scalar fit metrics are filled on the device to avoid host copies.
The FP32 final actor block/tanh and post-averaging actor norm limit match the
reference stability update. Five logged intervals of jointly vanishing gradient,
output update and action spread write `actor_stalled.json` and stop all ranks.
With `torch_compile_mode` enabled, dynamics and reconstructed observations are
also compiled. Retained gradient graphs disable AOT donated buffers and Inductor
in-place buffers so separate main/aux backward passes remain valid.
With `compile_geometry=true`, the pure FK/observation computation is also compiled
inside its existing highest-precision FP32 context. Geometry recomputation in
first-order backward uses this compiled path; higher-order derivatives retain
the eager path. This changes execution, not H32, rewards, batch sizes or stored
data precision. Set `agent.world_model.compile_geometry=false` to disable this
additional compilation. Checkpoint tensor keys and geometry contracts are unchanged.
`compile_model_math=true` also compiles the normalized dynamics fitting error,
using fixed shapes for the captured batches. It preserves the fitting loss and
gradient clipping; compilation can introduce small FP32 rounding differences.

For RP1, the actor consumes proprioception while B consumes `state` plus
`privileged_state`. SVG therefore defers full-body geometry reconstruction until
the dynamics rollout is complete and processes it in `scoring_chunk_size=2048`
chunks. Normalized proprioception is retained in trajectory order. B and
discriminator evaluation keep the original per-step batch shape to limit BF16
rounding changes. Actor observation normalization is reused within each step;
detached sequence-value training features are built without an autograd graph.
The terminal value and all H32 actor derivatives remain differentiable. Unusual
observation-key configurations retain the stepwise geometry path.

On CUDA, SVG forward and actor-gradient computation use one cached CUDA Graph.
Every replay receives fresh states, observations, commands and ordered latents,
and reads the current actor, dynamics, B, discriminator, value and normalizer
buffers. The real-data B covariance solve stays outside capture, retaining its
error check. The detached sequence-value training batch is refreshed by replay;
its optimizer update and all distributed reductions remain outside the graph.
Initial capture occurs during rollout preparation, before the ordinary actor
loss creates default-stream gradient nodes. Warmup/capture restore RNG state;
replay advances the normal training RNG. Shape changes recapture the graph and
checkpoint loading discards it. CUDA graphs add a private memory pool and an
initial capture cost. Set `agent.world_model.cuda_graph_imagination=false` for
the ordinary autograd path; CPU execution also uses that path. This execution
optimization preserves the objective and the configured sample/update budget.

The dynamics fit also has a separate fixed-shape CUDA Graph. Sampling,
increment-variance updates, gradient averaging/clipping, finite checks and Adam
remain outside capture; only the autoregressive loss and parameter gradients
are replayed. `cuda_graph_dynamics=false` restores ordinary fit autograd.

The fixed-z F-target rollout and sequence-value gradient calculation also have
independent graphs (`cuda_graph_forward`, `cuda_graph_value`). Value targets and
gradients reuse the detached H32 actor trajectory, with the original minibatch
size and accumulation order. F targets retain their own real-successor anchor,
sampled actions and fixed z; they cannot reuse the actor's mean-action,
time-varying-z trajectory without changing the learning target. Live weights,
normalizer buffers and fresh input tensors are read on every replay. Finite
checks, distributed reduction, clipping and optimizer steps remain outside.
All four graph caches are ephemeral, have separate memory pools and reset on
checkpoint load. Each flag independently restores ordinary execution without
changing the imagination horizon, batch size, rewards or update schedule.

Main replay remains on CPU. Each sampled training batch is transferred once to
the training device, then reused by FB, command critics and model targets.
CPU-to-CUDA copies are nonblocking on the current stream; CPU destinations keep
synchronous semantics. Scalar finite checks are aggregated per device, retaining
the existing failure diagnostics. These changes do not compress stored values.

RP1 enables `agent.cuda_graph_actor` for ordinary actor objectives under MBFB.
Eager and captured updates call the same `_actor_objectives` function. Separate
graphs handle the combined real-only gradient and the main/auxiliary gradients
needed for imagination mixing. Fresh observations, commands and z are copied
before replay; current actor/critic weights are read directly. Returned metrics
are cloned so later replays cannot overwrite them. RNG-preserving capture,
shape checks and checkpoint invalidation use the existing graph helper.
Mixing with H32 gradients, distributed averaging, finite-checking gradient
clipping and Adam remain outside capture. CPU, visual actors, depth diagnostics
and non-MBFB objectives retain eager execution. Set `agent.cuda_graph_actor=false`
to disable the two additional private graph pools without changing the losses.

RP1 also enables `agent.cuda_graph_fb` and `agent.cuda_graph_critics` for the
native FB update and the shared critics. FB keeps separate graphs
for real-only and MBFB-augmented targets; both preserve the B regularizer and
the F-only model loss. Critic targets retain their distinct sampled actions and
read current discriminator/target weights. Only loss evaluation and gradients
are captured: gradient averaging and Adam remain outside. Returned metrics are
copied before the next replay. CPU execution and FB updates with legacy
non-MBFB targets retain eager evaluation; checkpoint loading clears these caches.
Each flag can be set to
false independently; private graph pools increase memory use. Neither option
changes H32, network widths, minibatch sizes, rewards or update frequency.

The shared heading/velocity critic also uses `agent.cuda_graph_critics`. Its
eager and captured updates call the same `_shared_context_loss` function.
The two target actions are still sampled separately before capture/replay;
velocity bootstrap validity and per-head valid-sample masks are unchanged.
Gradient averaging, Adam and the target-network update remain outside the
graph. One additional cache is rebuilt on shape changes and cleared on
checkpoint load. CPU and independent heading/velocity heads stay eager.

The same `agent.cuda_graph_critics` setting captures discriminator loss and
parameter gradients, including the WGAN gradient penalty's double backward.
Interpolation coefficients are sampled outside capture to preserve the RNG
sequence. Both paths use one loss function; changed shapes, loss mode or penalty
rebuild the cache, and checkpoint loading clears it. Gradient averaging and Adam
remain outside capture. This does not change the discriminator reward or loss.

Command-critic optimizers honor the existing `agent.use_fused_adam` setting,
as the other native optimizers do. Fused Adam keeps FP32 master weights and
moments but can introduce small floating-point rounding differences.

`agent.overlap_critic_communication=true` lets distributed shared-critic gradient
averaging run while the independent heading/velocity critic computes. The
shared critic's Adam step finishes before actor training; collective order,
per-update optimizer counts and target-update positions are preserved. Initial
graph capture and checkpoint reload use synchronous updates until both graphs
exist. A command-graph rebuild drains pending communication before capture.
CPU, single-rank and independent-head configurations keep synchronous execution.
Set the option to false to restore the original scheduling without changing
model/checkpoint tensors or objectives. The worker's existing local CPU affinity
remains the default; split-core/preferred-memory experiments stay in project_ops.

Graph replay stages its input tensors with a batched copy. Expert observation
reconstruction also follows `torch_compile_mode` on CUDA, including variable
query lengths. Motion data remains on CPU and only requested frames are copied;
sampling probabilities, RNG draws and trajectory boundaries are unchanged.
CPU execution and `torch_compile_mode=null` retain eager reconstruction.

Expert sampling reconstructs current and next frames in a single query while
preserving trajectory selection, padding and live curriculum weights. The raw
motion arrays stay in CPU memory; requested frame rows are gathered directly
into pinned memory before transfer. The transient batch is larger than either
individual query, but no dataset-sized GPU cache is created. Changing query
batch shape can introduce small FP32 rounding differences.

`agent.compile_context_critics=true` also compiles the heading/velocity modules
after PUFO constructs them. Their FP32 execution and optimizers are retained;
both critic training and the action derivative used by the actor use these
compiled forwards. The existing `torch_compile_mode=None` disables this path.
`agent.world_model.compile_gradient_mixing=true` compiles the pure tensor part
of MBFB gradient mixing on CUDA: norm calculation, ratio cap, finite masking,
weighted combination and diagnostics. Assignment to parameter gradients stays
outside compilation. CPU uses the ordinary function. These startup flags can
be disabled independently and do not change checkpoint keys or training budgets.
FP32 fusion introduces small rounding differences, so parity is tolerance-based
for these helpers rather than the exact parity of the native CUDA Graph paths.

For bounded performance runs, `profile_update_phase` selects one update phase,
`profile_update_rank` selects its rank, and `profile_update_index` selects the
minibatch within that phase. Index 3 captures a scheduled imagined update with
the default four-update imagination schedule on a fresh run. Index 1 captures
one with the previous two-update schedule. Exclude compilation and the
profiled phase from throughput comparisons. Profile ranges
also separate dynamics fitting, model F targets and actor-rollout preparation.
TensorBoard records collection and
learning time plus `Perf/gpu_allocated_mib`, `Perf/gpu_reserved_mib`, and cumulative
`Perf/gpu_peak_allocated_mib`. These allocator counters exclude simulator/driver
allocations; use process/device telemetry for total GPU memory.

## RP1 Pushes And Environment Contracts

Training pushes **remain enabled**, as requested. RP1's simulator, joint ordering,
action scaling, observation noise/history, FK, task rewards, command critics and
velocity estimator remain in use. This is an algorithm port, not a G1 environment
or motion-data reproduction.

UFO reference commands are `[vx, vy, vz, valid]` in the robot yaw frame (m/s).
Yaw alignment rotates XY and preserves world-vertical Z. The reference VZ comes
from the motion's root link velocity; current XYZ velocity still comes from the
existing proprioceptive estimator, supervised by simulator link velocity during
training. No additional estimator or vertical tracking reward is introduced.
The existing context velocity reward remains horizontal. B inputs and z dimensions
are unchanged; the actor input grows from 395 to 396. PUFO keeps its XY command.

The adapter records the physical result before a random push and separately logs
the velocity override. Dynamics fitting uses this clean target. On the following
step, a push flag restarts prediction from the real post-push state; an episode
boundary does the same using the new episode. The transition preceding the push
is still fitted. Main FB/critic replay retains every real transition and push.
F imagination starts from the actual post-push state matching its real next
observation; the COM-to-link velocity conversion uses the recorded orientation.
Future random pushes are not predicted during SVG. The upstream corrected H32
experiment disables pushes; this is an intentional local difference.

## World-Model Diagnostics

`agent.world_model.diagnostics=true` enables return distributions, actual
imagination budgets and multi-horizon validation. Metrics use the existing
TensorBoard/W&B writer with the `train/` prefix. They do not control losses,
sampling, gates or optimizer steps. Existing runs need a restart to load this code.

| Logged prefix or tag | Interpretation |
| --- | --- |
| `train/wm/returns/{fb,discriminator}/std_local`, `p10_local`, `p50_local`, `p90_local` | Distribution of trajectory returns, including terminal value, within one rank and one imagined batch. |
| `train/wm/returns/.../rollout_mean`, `terminal_mean` | Separate discounted rollout and terminal contributions. Their sum is the corresponding imagined return. |
| `train/wm/returns/.../terminal_abs_fraction` | Mean of `abs(terminal) / (abs(rollout) + abs(terminal))`; avoids cancellation between opposite-signed components. |
| `train/wm/returns/.../nonfinite_fraction` | Fraction of invalid returns or terminal values. Distribution summaries substitute zero for invalid inputs; discard those summaries when this is nonzero. |
| `train/wm/budget/imagination_fraction` | Fraction of **all agent updates** that actually executed imagination, including zeros for skipped/non-imagination updates. Normally 0.25 with interval 4. |
| `train/wm/budget/trajectories_global_per_update`, `transitions_global_per_update` | Actual global imagination budget averaged over all agent updates. Four ranks at batch256/H32/interval4 normally give 256 trajectories and 8192 transitions per agent update. |
| `train/wm/budget/trajectories_per_imagination_local`, `horizon` | Batch size and rollout length on executed imagined updates; normally 256 and 32. |
| `train/mb/actor_model_base_ratio`, `actor_model_clipped` | Relative weighted model-gradient norm and frequency of the existing model-gradient cap, measured before distributed actor-gradient averaging. |
| `train/wm/validation/h{1,4,8,16,32}/model_error`, `baseline_error`, `ratio` | Held-out normalized prediction error at that horizon, compared with constant velocity on the same uninterrupted windows. These are real recorded-action rollouts, not an on-policy imagination accuracy guarantee. |
| `train/wm/validation/h.../valid_samples_global`, `uninterrupted_fraction`, `nonfinite_fraction` | Coverage and numerical validity of each horizon. Ignore errors/ratios when valid samples are zero; ratios are uninformative when baseline error is near zero. |
| `train/wm/validation/horizon_available` | Whether physical replay contains a full requested validation window yet. |

Return statistics remain inside the existing imagined CUDA Graph and require no
additional rollout or backward pass. Multi-horizon validation runs at the existing
`validation_interval` (200 agent updates by default), with an extra held-out
rollout to H32. Reset/push boundaries exclude affected windows from long-horizon
errors instead of silently counting restarted trajectories as full H32 predictions.
Validation sampling restores the training RNG. The original `train/wm/val_ratio`
continues to report the eight-step diagnostic, with its original restart semantics.

The runner averages scalar metrics over logging intervals and ranks. In particular,
averaged local percentiles/std are **not** pooled global percentiles/std, and are
not confidence intervals. Trajectories and adjacent time steps can be correlated.
Multi-horizon validation first sums error/count statistics across ranks, so its
ratios use all valid held-out samples. Probe/validation metrics only contribute on
updates that execute them; budget metrics explicitly include zero-budget updates.

For a targeted sampling-gradient check, enable:

```bash
agent.world_model.gradient_probe_interval=1000
```

Zero is the default and disables extra backward passes. Choose an interval divisible
by both fitting and imagination intervals; the probe runs only on a scheduled,
successful imagined update. It randomly partitions the current trajectories into
two disjoint groups and computes their actor gradients with fixed network weights,
full network batch shapes, covariance, rollout noise and full-batch reward scaling.
It restores RNG, model buffers and the value-training batch, and never applies the
probe gradients. It costs two extra H32 forward/backward passes on probe updates.

`train/wm/probe/gradient_cosine_local` and
`train/wm/probe/gradient_relative_difference_local` compare the two groups. Check
`available`, `finite`, `nonzero` and both `half_batch_size_local` /
`other_half_batch_size_local` before interpreting them. At batch256 this is a
128-versus-128 partition diagnostic, **not** two independently sampled batches of
256, a global distributed gradient, or an automatic batch-size recommendation.
Low agreement suggests testing more trajectories, but task heterogeneity can also
cause disagreement. Full-batch versus half-batch BF16 shape changes are avoided.

Compare batch256 and batch512 at the same H32/update interval, using both equal
real-environment-step and equal-wall-time evaluation. Pair these diagnostics with
existing `eval/` tracking metrics and `Perf/` costs: rising imagined return alone
can reflect model bias and does not establish improved policy behavior. These
diagnostics do not automatically enable evaluation or change its interval.

## Checkpoints And Validation

The default uses checkpoint version 14 and saves the sequence value and its
optimizer. Loading a version-12/13 checkpoint keeps compatible UFO policy/optimizers
and starts fresh world-model/value fitting. An incompatible same-version network
width, return convention or value context is rejected.
The physical GPU ring is not checkpointed; resume collects fresh windows and
uses the first successful fit immediately, preserving total counters. Validation
does not participate in resume readiness. Older v14 checkpoints retain learned
weights and optimizers; quality-gate state, `gate_ratio` and `value_warmup_updates`
are ignored. The current-update fit-success flag is not checkpointed.
Checkpoints now record `obs_dims`. Old XY-command checkpoints cannot resume into
the XYZ-command networks; the runner rejects this mismatch before loading weights.
Start a new training run for XYZ commands. `export_ufo_deploy.py` supports both old
XY checkpoints and new XYZ checkpoints, writing version-1 or version-2 deployment
metadata respectively; the MuJoCo consumer selects the matching reference sidecar.
The extra root-position observation remains training-only.

Checks live in `project_ops`:

```bash
.venv/bin/python -m unittest discover -s project_ops/tests -p 'test_ufo_mbfb*.py'
.venv/bin/python -m unittest discover -s project_ops/tests -p 'test_ufo_vertical_velocity.py'
.venv/bin/python -m unittest discover -s project_ops/tests -p 'test_ufo_motion_streaming.py'
.venv/bin/python project_ops/scripts/check_mbfb_dynamics_parity.py /path/to/MBFB
.venv/bin/python project_ops/scripts/check_ufo_mbfb_runtime.py \
  --device cuda:0 --visualizer none --production_networks --production_math \
  --output_dir /data/zym/roboparty_train/mbfb_checks/new_check physics=newton_mjwarp
```

The runtime check uses 32 real Newton environments, forced frequent pushes and
resets, batch 1024 dynamics fits and, with `--production_networks`, the production
policy/F/critic sizes and H32/batch512 SVG. It verifies imagined actor/value updates
from the first fit using the default schedule and checks checkpoint recovery.
It is not evidence of learned model quality,
training convergence or sustained eight-GPU FPS.
