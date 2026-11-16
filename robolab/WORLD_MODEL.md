# UFO World Model

Supported: `RP1-UFO`, Newton, flat terrain, 24 joints, context commands,
timeout-only resets. Deployment observations and the exported actor are unchanged.
The imagination objective scores the existing discriminator with frozen parameters
and retains its state-input derivatives. Its real-data training loss is unchanged.

The plugin is **enabled by default** for UFO. Set `agent.world_model.enabled=false`
to use the original UFO training path, which constructs no world-model networks, optimizer, physical-observation
group or extra replay fields. No imagination/F-target sampling or plugin losses
run. The original actor, FB, discriminator and critic update paths remain active.

The RP1-UFO default is **FB sequence value on MBFB H32**, `objective=mbfb`, with a
`512 x 512 x 512` dynamics MLP and 256 imagined trajectories every four agent
updates. Real dynamics still fit every two updates. See [MBFB.md](MBFB.md) for
the current algorithm, CPU-backed main replay, GPU physical replay, fit-then-imagine
startup without a quality gate, push handling and diagnostic validation.
The `[256,128]` sequence-value network is mandatory on this path. It learns the
FB implicit return and original discriminator regularizer with ordered future z;
no explicit pose tracking reward is used. The existing
`fb_sequence`, `fb_value` and `context` paths remain explicit legacy ablations.
The sequence-return theory and implementation notes below describe those legacy
paths, not the default MBFB objective. A gate opening is not evidence of policy benefit.
Shared UFO/PUFO motion interpolation normalizes quaternions, uses the shortest
rotation arc, and respects the interpolation weight even for small angles.
Root quaternions are also normalized by default at simulator reset. These numeric
fixes apply with or without the plugin; continuous-time motion sampling is unchanged.

## Run And Ablate

```bash
uv run --no-sync python robolab/scripts/rsl_rl/train.py \
  --task RP1-UFO --device cuda:0 --visualizer none physics=newton_mjwarp

# Original UFO without the world-model plugin:
uv run --no-sync python robolab/scripts/rsl_rl/train.py \
  --task RP1-UFO --device cuda:0 --visualizer none physics=newton_mjwarp \
  agent.world_model.enabled=false
```

All plugin ablations below assume `agent.world_model.enabled=true` (the default).

| Variant | Override | Purpose |
| --- | --- | --- |
| UFO | `agent.world_model.enabled=false` | Original policy learning |
| Physics only | `agent.world_model.actor_weight=0 agent.world_model.forward_weight=0` | Model fitting and diagnostics |
| Full plugin | Default, or `agent.world_model.enabled=true` | F targets and trusted physical-gradient addition |
| Without F targets | `agent.world_model.forward_weight=0` | Keep the original real-only F objective |
| F targets only | `agent.world_model.actor_weight=0` | Train F with short model targets, no imagined actor loss |
| No pose supervision | Default `agent.world_model.tracking_coef=0` | Tracking is represented by B-derived z |
| Recorded tasks | `agent.world_model.tracking_coef=0 agent.world_model.tracking_z=false` | Disable fresh expert relabeling |
| FB-only + model | `agent.world_model.discriminator_reward=false` | Remove imagined discriminator reward and value |
| Fixed task | `agent.world_model.objective=fb_value agent.world_model.tracking_z=false agent.world_model.tracking_coef=0` | Fixed-z value-expansion ablation |

Compare identical initial policy, seeds, motion sampling, environment count,
randomization and evaluation conditions. Report environment steps and wall time.
UFO actuator delay is zero in these comparisons; PUFO retains its original delay.
Fitting a separate physics network alone cannot improve the actor because they
do not share parameters.

## Legacy Sequence Objective And Paper Comparison

[SuperMimic](https://arxiv.org/pdf/2610.28480), Sections 3.2-3.3, separates physical
prediction from task optimization. Its tracking interface uses explicit reference
errors and a matching terminal critic. The transferable idea is differentiating
the policy through predicted physical outcomes. Table A.4 specifies a 2048-wide,
three-hidden-layer dynamics MLP. SuperMimic has a task-conditioned scalar value
critic, not an FB forward map; model-generated F targets are a separate MBFB idea.

[UFO's official report](https://roboparty.github.io/UFO/assets/UFO.pdf) describes
unsupervised control with reference-conditioned latent tasks. In this code,
`B` encodes target states and `F(s,a,z)` represents discounted future occupancy,
not physical transitions. The ordinary actor maximizes `F(s,a,z)^T z` alongside
its existing regularizers. The WM addition combines FB and discriminator returns;
the original discriminator continues its separate real-data training.

The [FB-CPR formulation](https://arxiv.org/html/2504.11054v1#S2) gives
`M_z(s,a,ds') ~= F(s,a,z)^T B(s') rho(ds')`, hence
`Q_r ~= F^T E_rho[B r]`. A compatible instantaneous reward is constructed as:

The legacy `objective=fb_sequence` advances a contiguous task sequence at every step:

```text
C       = E_real_next[B_target B_target^T]
w[k]    = (C + ridge * mean(diag(C)) * I)^(-1) z[k]
r_fb[k] = B_target(obs_hat[k+1])^T w[k]
r_d[k]  = discriminator_reward(obs_hat[k], z[k])
x[k]    = (s_hat[k], xi, actor_obs[k], context[k:k+K])
G_fb    = sum(k=0..H-1, gamma^k * r_fb[k]) + gamma^H * V_fb_seq_target(x[H])
G_d     = sum(k=0..H-1, gamma^k * r_d[k]) + gamma^H * V_d_seq_target(x[H])
alpha   = reg_coeff * stop_gradient(mean_real_actor_batch(abs(Q_fb)))
beta    = tracking_coef * max(stop_gradient(mean_real_actor_batch(abs(Q_fb))), 1)
P       = sum(k=0..H-1, gamma^k * tracking_mask[k] * pose_cost[k]) / sum(gamma^k)
G_H     = G_fb + alpha * G_d - beta * P
# Independently for each raw reward/value head:
y[H]    = V_seq_target(x[H])
y[k]    = r[k] + gamma * ((1-lambda)*V_seq_target(x[k+1]) + lambda*y[k+1])
```

`K=10`, `H=32`, and `lambda=0.95`. Each context entry contains z,
world heading/velocity targets converted to the imagined root heading frame,
and command-validity masks. The critic therefore requires H+K=42 aligned replay
records from one context. Its input dimension is
`59 + xi_dim + actor_obs_dim + K*(z_dim+6)`. It sees a finite future window, not
the exact motion phase or complete future: continuation outside that window is
a conditional approximation. With `tracking_z=true`, eligible expert-tracking rows
receive a fresh contiguous expert reference encoded by the current online B.
Other rows retain recorded behavior z and commands.

The online sequence critic has two outputs (FB and discriminator) on one shared
trunk and fits detached imagined lambda-return targets separately. Neither value
target includes alpha: changing the actor's reward scale does not change what each
head predicts. With `scale_reg=false`, alpha is just `reg_coeff`. Production updates
reuse the detached FB scale from the ordinary real-batch actor loss, averaged over
ranks once and fixed across all microbatches. This deliberately differs from MBFB's
imagined-return batch scale. MBFB uses `mean(abs(G_fb))` for alpha and its
clamp-at-one value for beta. Those scales are not mathematically equivalent;
a full-batch imagined scale could also be implemented independently of microbatch
size. Here one real-batch scale is shared by all microbatches and ranks.
Direct diagnostic loss calls default to scale 1 unless explicitly supplied.

Discriminator reward uses the same logit conversion as real critic training and
the current imagined state with its current z. FB reward uses the successor state.
The same noisy observation is reused; there is no extra observation-noise draw.
Actor
updates wait for 100 sequence-critic updates; critic fitting can proceed during
that warmup once dynamics passes its quality gate. Both use the same stochastic
policy, frozen target B/covariance, and shifted future windows. The terminal
critic retains state-input derivatives but never receives actor-loss parameter
gradients. No fixed-z F value or original discriminator critic is used for sequence
continuation: those critics do not observe the future task sequence. Disabling
`discriminator_reward` builds the original single-output FB sequence critic.

The `fb_value` ablation retains the paired fixed-task objective:

```text
C       = E_real_next[B_target B_target^T]             # uncentered
w       = (C + ridge * mean(diag(C)) * I)^(-1) z0
z_eff   = C w
r(x)    = B_target(x)^T w

s_hat[k+1] = model(s_hat[k], actor(obs[k], z0), xi)
G_fb = sum(k=0..H-1, gamma^k * r(obs[k+1]))
      + gamma^H * pessimistic(F_target(obs[H], a[H], z0)^T z_eff)
G_d = sum(k=0..H-1, gamma^k * discriminator_reward(obs[k], z0))
      + gamma^H * pessimistic(Q_discriminator_target(obs[H], a[H], z0))
G_H = G_fb + alpha * G_d
L_WM = -actor_weight * mean(trusted * G_H)
```

The ridge solve changes the effective reward task. Its terminal value must use
`z_eff=Cw`, not silently use `z0`. Actor and F policy conditioning remain `z0`.
A projection-residual gate rejects goals distorted too far by this solve.
Pessimistic reduction and approximate learned models make this a conservative
value-expansion surrogate, not an exact Bellman identity. This fixed-z ablation has
no pose reward or reward prediction head. The fixed-z discriminator continuation uses
the existing target critic (the discriminator head when the trunk is shared).
Its parameters are frozen and its state/action derivatives are retained.

### F Targets, Tracking Latents And Pose Loss

The optional F auxiliary objective teaches predicted future occupancy to the
ordinary FB actor's value signal. It augments the existing real one-step objective;
it does not replace it or treat `F^T z` as an instantaneous reward. Each original
FB batch supplies the real `(s0, a0, s1)` anchor and real goal marginal. The model
predicts only `s2..sH` with fixed relabeled z and fixed world heading/velocity
commands reconstructed from the relabeled next observation. `H=4` is independent
of the 32-step actor horizon. The first predicted action is the same sampled
`next_action` used by the real one-step F target.

```text
M_z(s0,a0) = delta(s1) + gamma*delta(s2) + ...
             + gamma^(H-1)*delta(sH) + gamma^H*M_z(sH,aH)
L_total_FB = L_real_FB + forward_weight * ramp * L_model_F
```

The successor-measure loss uses pessimistic target F/B for the bootstrap, detached
online B for its goal factors and imagined occupancy terms, and the original FB
off-diagonal objective. Only online F receives auxiliary gradients. Both losses
share a single existing F optimizer step; B's gradients/optimizer step remain the
real-data ones. No synthetic transitions enter replay, and no extra critic or
optimizer is created. `forward_weight=0` bypasses target construction entirely.

Targets require warmup and fresh held-out validation. Terminated or externally
pushed anchors are masked out; action deviation from the real-next-state policy action
rejects unsupported continuations. Timeout
alone does not zero bootstrap. Validation must cover at least H-1 model steps.
The weight ramps over `ramp_updates` model-fitting phases after `warmup_updates`.
The original batch denominator is retained for rejected rows, so masking never
amplifies the surviving trajectories. `wm/forward_loss`, `wm/forward_weight` and
`wm/forward_trusted_fraction` expose the contribution. These conservative settings
still need trained-policy comparisons because model errors can bias F.

Before imagination microbatching, `tracking_z=true` samples a contiguous expert
sequence for each row whose replay source is exact tracking. It requests
`H+K+seq_length-1` valid transitions, encodes expert **next** observations with the
current online B under no-grad, averages sliding `seq_length` windows, and projects
each mean into z. The pose target is the first next-observation frame of that same
window. Heading and velocity use the same frames, with the reference heading aligned
to the physical starting heading. Short clips with incomplete windows retain their
recorded task and receive no pose supervision; padding is never encoded. Random/reward
rows retain their intended tasks. References are sampled once per whole update,
not independently per step or microbatch. Setting `tracking_coef=0` still allows this
fresh z relabeling, making the pose ablation independent.

The pose term compares raw heading-local RP1 body observations, before normalization:
SmoothL1 position/height errors scaled by 0.1 m, plus tangent/normal rotation chordal
error scaled by 0.5 squared. FK is reused from FB scoring; gradients reach the actor
through predicted states, while dynamics, reference frames and B are frozen.
Only valid tracking rows contribute, and trust gates apply to the entire trajectory.
The discounted cost is averaged over time and the original whole batch (including
masked/rejected rows); MBFB averages this cost over its selected tracking rows.
Consequently identical `tracking_coef=0.1` is not necessarily identical effective
weight. Metrics expose accepted tracking fraction, position/height MSE and rotation
chordal error, pose return and coefficient.

As in MBFB, pose is a finite-horizon auxiliary penalty, with no pose terminal value.
The two sequence-value heads continue to fit only FB and discriminator returns.
This is not an infinite-horizon pose-return objective. A fixed-z F/original critic
cannot exactly represent changing-z continuation: this implementation retains the
future-window sequence critic, whereas MBFB bootstraps F and its original critic
using the last z as an approximation.

One control step is 20 ms; 32 imagined steps cover 0.64 s. Gradients propagate
through all 32 predicted transitions without detaching intermediate states,
actions or history. With discount 0.98, the terminal-value coefficient is about
0.524. The 8-step dynamics fitting window and 32-step policy expansion serve
different purposes: fitting follows recorded actions, while imagination samples
new actions from the policy at its own predicted states.

In the `fb_value` ablation the task latent z stays fixed throughout each expansion, matching F's fixed-z
value objective. Real tracking still advances z every environment step. This
does not create a sequence-conditioned value or guarantee a complete acrobatic
motion. In particular, a common replay context ID does not imply a constant
recorded z. Future recorded z values are diagnostic only, not silently inserted
into the actor or reward. The original 1-step variant remains available with
`agent.world_model.horizon=1` for controlled ablations.
Do not sum `F^T z` as an instantaneous reward, which would count future value twice.

C is recomputed from 4096 independent real replay **next observations** using
target B, matching FB's goal marginal. It includes existing observation noise
and post-event observations, not clean pre-push model targets. It is sampled from
all real environments, not experts or only the dynamics training split.
Distributed workers aggregate outer-product sums and counts before the solve.
The 1024-sample chunks accumulate sums and counts, not equally weighted chunk means.

## Observations And Gradients

- The physical state is 59D: root height, XYZW orientation, world root-link
  linear/angular velocity, and 24 joint positions/velocities. Two MLPs predict
  58 structured local-frame increments with normalized quaternion updates and
  finite rotation derivatives at zero.
- `UFOPrivilegedKinematics` reconstructs the original 373D privileged input using
  `pytorch-kinematics==0.10.0` and analytic link velocities. Joint/body ordering is
  resolved by name. URDF hash, ordering and height offset are checkpointed.
  Initialization checks parity against the actual simulator observation.
  Geometry forward and first-order backward locally disable TF32 and autocast;
  backward recomputes the geometry graph under the same precision guard. Network
  precision settings are restored afterward. This was checked under the task's
  `high` matmul and BF16 settings without relaxing simulator-parity tolerances.
- Each imagined state receives one observation-noise sample, reused by B, the
  next actor and terminal F. History and previous actions evolve through imagined
  states. Recorded world heading/velocity targets are converted using the
  imagined orientation, rather than copying real local command observations.
- Target B, the applicable terminal evaluator, dynamics and velocity-estimator parameters are frozen, with input
  derivatives retained. In `fb_value`, terminal actor weights are frozen while preserving its
  state-to-action derivative. Rollout and terminal actions use UFO's clipped,
  reparameterized sampling.
- The original UFO actor gradient is averaged unchanged. WM gradients are
  averaged and clipped separately before addition. `wm/actor_grad_ratio` and
  `wm/actor_grad_cosine` expose magnitude and alignment. Imagined transitions
  are not inserted into FB replay.
- FK and B scoring flatten the predicted time and trajectory dimensions and use
  chunks of 2048 states. Actor actions/history remain sequential, and B receives
  exactly the same noisy observation as the next policy step.
- A 1024-trajectory update accumulates 64-trajectory microbatches. Every accepted
  trajectory retains all 32 transitions. Losses use the full original denominator;
  gradients are reduced/clipped once, the actor steps once, and the sequence critic
  steps once. Frozen evaluators and target weights do not change between chunks.

## Fitting And Trust Checks

Each dynamics fitting phase runs one optimizer batch of 1024 eight-step windows
per rank. The single model resamples one real window. A phase runs after every
UFO update (`update_interval=1`, equivalent to `wm_every=1`); `imagination_interval=1`
schedules one full 1024-trajectory update per eligible phase. `updates` counts
phases and `optimizer_steps` counts optimizer batches; both advance once per WM
update with `fit_batches=1`. Warmup is explicitly 100 phases (100 optimizer
batches), and validation runs every 10 phases. The original UFO FB batch remains
1024. The network scale follows SuperMimic while the one-batch/every-update
schedule follows MBFB; this is not a claim of complete training equivalence.

Context includes every collision shape's friction, joint calibration, torso COM,
body masses and controller gains. Prediction loss and delta normalization use
pre-push physical outcomes. Recorded events are applied before the next step.

FB imagination requires all of the following:

1. More than 100 completed dynamics fitting phases.
2. Recent held-out H-step worst-time P95 normalized error at most 0.5, taking the
   worst rank under distributed training.
3. H+K records (H+1 for `fb_value`) in one episode with one context ID. A context refresh can re-anchor
   commands to the behavior trajectory and cannot be crossed.
4. No future push in the H transitions; the initial real state may be post-push.
5. Relative covariance projection residual at most 0.1 and executed-action deviation
   from replay at most 0.5 of `action_rescale` on every joint.

Before the first valid horizon check, `wm/quality_validated=0` and
`wm/quality_ready=0`; its logged `wm/quality_p95=0` is a missing-value sentinel,
not a measured zero prediction error. The internal quality value remains infinity.

Masks are detached. Any failed step rejects the whole expansion and bootstrap.
The original batch denominator is retained so rejection reduces update strength.
Accepted updates ramp the weight over 1000 updates. Passing the error gate does
not prove correct dynamics or action derivatives in rare, unsupported states.

For long expansions, inspect `wm/trusted_fraction_h1`, `h4`, `h8`, `h16` and
`h32`, plus `wm/action_radius_fraction` (the whole-window action gate pass rate).
`wm/trusted_prefix_steps_mean` counts accepted prefixes
after the full-window context, push and projection checks. Recorded latent drift
is reported separately as `wm/recorded_z_relative_drift`. Comparing imagined
actions with later behavior actions can reject valid trajectory changes,
especially when recorded z changes. Do not interpret low acceptance as proof
of bad physics, or silently relax gates merely to activate updates. There is no
fallback to shorter rollouts: an accepted H=32 update always uses all 32 steps.

Validation follows the same held-out window for all 1/4/8/16/32-step checks and
gates the complete path, not only its endpoint. It reports normalized error plus physical
component RMSE for height, rotation vector, root velocities, joint positions and
velocities. Rotation-vector component RMSE is not the mean rotation-angle norm.
Clean target-B embedding and covariance-decoded FB-score errors are reported too.
These remain value-prediction diagnostics, not proof of correct action derivatives.
Inspect errors at the actual imagination horizon and compare real tracking.

The older explicit `objective=context` option retains its heading/velocity reward
and separate value head. Its value warmup and lambda returns are not used by FB.

## External Events

Pushes overwrite root COM world linear/angular velocity after integration.
Replay stores the first pre-push state, final absolute target and a separate mask:

```text
v_link = v_com_target - cross(w_target, rotate(q_predicted, root_com_offset))
```

This removes dependence on old velocity but retains the orientation derivative.
A zero target is still an event. Interval pushes occur after resets; outgoing
terminal transitions exclude pushes belonging to the new episode. Future random
pushes are not imagined. Random torque injection (`rfi_lim=0.1`) remains active,
so deterministic dynamics still approximates a conditional mean.

## Evaluation And Data Coverage

Flat motion resets preserve reference root height. Reference link velocities use
Newton's link-velocity API, which applies each environment's current COM offset.
After all joint poses are written, actual collision contacts determine the
minimum upward penetration correction. COM randomization invalidates articulated
velocity/FK caches; otherwise child link velocities can retain the old COM frame.
Evaluation records the resulting `reset_root_lift_m` instead of hiding this correction.

Joint EMD is retained. `tracking_kinematics_clips.csv` adds time-aligned root
rotation/tilt, height, XY trajectory, whole-body position and link-velocity errors,
and inversion coverage. Only initial heading and XY translation are aligned;
later drift and absolute height remain visible. Contact-force and body-origin
clearance diagnostics are literal measurements, not verified flight labels or
an acrobatic success classifier.

UFO now uses ten-second windows with two seconds of overlap. This retains context
around former hard boundaries; it does not guarantee coverage of every longer
event. Each new window receives a baseline weight `old_window_count/new_window_count`
for its source, in both reset and expert sampling. Difficulty weights multiply
this baseline once. This preserves source-level sampling mass; overlapping frames
still change the local frame distribution. Two explicitly selected stair-descending sources requiring absent terrain
are excluded from flat UFO loading; source files are retained, and PUFO's selection
is unchanged. The current selection is 109 sources and 1193 overlapping windows.
The updated audit counts unique source frames: 6.9 seconds with root tilt above
120 degrees among 9244.48 seconds. Inversion also includes ground rolls; it is not
a flip-success label. A world model cannot supply missing dynamics coverage.

One-second phase bins use a within-clip mixture of 50% uniform, 40% current
difficulty, and 10% historical difficulty. Nominal evaluation computes aligned
body-position, root-height and root-rotation errors, then updates this sampler.
These errors never enter the actor reward. Episode-reset return is not used to
label a reference phase because reset motion and rollout context are independent.
The phase sampler's layout and history are checkpointed.

Expert tracking contexts now include short clips: actual transition lengths and
valid masks delimit averaging and refresh, while padding never crosses a source
boundary or becomes an executed reference target. The ordinary FB/discriminator
sampler does not consume phase priorities. Fixed-length tracking remains available
with `agent.rollout_expert_variable_length=false`.

Inspection candidates include `dance1_subject1` at 105-108 s and 114-117 s, and
`dance2_subject4` around 59-61 s. Some events cross ten-second boundaries, so
evaluate complete motions before interpreting clip scores. UFO's report Section
6.2 separately studies fixed foundation/cartwheel dataset mixing. Preserve and
report dataset proportions in WM comparisons.

Reproduce the read-only coverage audit with:

```bash
uv run --no-sync python project_ops/scripts/audit_ufo_motion_coverage.py \
  --output /tmp/ufo_motion_coverage.json
```

Default evaluation runs `tracking_eval_modes=(nominal, robustness)` with seed 42.
Nominal disables push, observation noise, torque noise and physical randomization;
robustness retains the training distribution. Environments run sequentially to
bound memory. Outputs and protocol metadata live under `eval/<mode>/`; only
nominal updates sampling priorities and phase difficulty. Fixed seeds make the
comparison repeatable, not necessarily bitwise deterministic on GPU. Include
landing/recovery footage and physical metrics; lower joint EMD alone is insufficient.

## Checkpoints And Verification

Flat UFO replay keeps `z` and `heading_z_next` in FP32. Instead of storing
`history_actor` on every transition, it reconstructs the history from the stored
single-step observations and explicit reset markers. A four-frame prefix preserves
the histories of the oldest ring entries without excluding them from sampling.
Pre-reset terminal observations retain their complete histories. This applies with
the world-model plugin enabled or disabled; PUFO depth replay is unchanged.
Replay snapshot version 4 saves the history layout and prefix. Resuming an older
uncompressed snapshot retains model/optimizer weights and collects fresh replay.
At 5,120,000 transitions per rank, eight ranks save approximately 47.6 GiB versus
full FP32 history storage, excluding small reset-marker/prefix overheads.

Version 12 saves model/optimizer states, covariance, fitting phases/optimizer steps,
warmup/ramp and validation counters, and the one- or two-output sequence
critic/target/optimizer state. The return contract includes the discriminator
reward switch and logit conversion, tracking relabeling/pose switches and the expert
encoding window. Version 10 and earlier retain the UFO policy
but restart WM/value/optimizer warmup and replay collection; their old value
return/data contracts do not match this configuration.
Resume checks physical, observation/action, robot geometry, network and
return contracts, including imagined horizon. Changing only the `fb_value` imagination
horizon in a same-version checkpoint retains learned dynamics, optimizers and
compatible replay, but resets horizon-validation results and the actor ramp.
Policy gradients remain disabled until the new horizon passes validation.
Sequence horizon, context length and objective changes remain rejected. Older-version checkpoints retain UFO
policy and optimizer weights but restart WM and replay collection.
The F auxiliary target adds no checkpoint tensors and keeps the version-12
physical/return contract. Its horizon and weight are training settings, recorded
in the model config; changing them does not invalidate learned dynamics.
With the plugin disabled, model and optimizer loads ignore WM entries. Switching
between enabled and disabled retains the UFO networks and their optimizer states,
but recollects replay because the physical-field contracts differ. Baseline-to-baseline
resume retains compatible baseline replay normally.

```bash
uv run --no-sync python -m unittest discover -s project_ops/tests -v
uv run --no-sync python project_ops/scripts/check_ufo_world_model_events.py
uv run --no-sync python project_ops/scripts/check_ufo_reset_consistency.py
uv run --no-sync python project_ops/scripts/check_ufo_tracking_protocol.py \
  --device cuda:0 --visualizer none physics=newton_mjwarp
uv run --no-sync python project_ops/scripts/check_ufo_world_model_runtime.py \
  --device cuda:0 --visualizer none physics=newton_mjwarp
uv run --no-sync python project_ops/scripts/check_ufo_world_model_training.py \
  --device cuda:0 --visualizer none physics=newton_mjwarp
uv run --no-sync python project_ops/scripts/check_ufo_world_model_training.py \
  --production_math --output_dir /tmp/ufo_world_model_training_compiled \
  --device cuda:0 --visualizer none physics=newton_mjwarp
uv run --no-sync python project_ops/scripts/check_ufo_world_model_training.py \
  --horizon 32 --production_math --production_networks --production_batches \
  --output_dir /tmp/ufo_world_model_training_h32 \
  --device cuda:0 --visualizer none physics=newton_mjwarp
```

Runtime checks verify reconstruction, event ordering and real replay fitting.
The training check uses a small random policy: default-gate rejection, explicitly
labeled smoke-only permissive gates for gradient connectivity, full UFO updates,
and physical evaluation/CSV output. `--horizon` overrides its imagination length;
the default is `fb_sequence`; `--objective fb_value` tests the original ablation.
`--production_networks` retains network widths and latent dimension.
`--production_batches` retains 1024 windows x 1 fitting batch, 1024 imagined
trajectories, and the ordinary 1024-sample FB batch. Simulator count remains four;
this is an integration/memory test, not full collector throughput. Sequence H32
checks collect continuous H+K records and disable pushes in the smoke environment; event
handling is exercised separately by the event/runtime checks. The summary records
actual rollout steps, synchronized timing and PyTorch allocator peaks (excluding
external simulator allocations). These checks do not establish improved
tracking, sample efficiency or wall time. No trained-policy improvement is
claimed without a controlled baseline comparison.

CPU regressions compare complete UFO updates with an absent plugin config versus
an explicitly disabled plugin: model/optimizer tensors, metrics, replay sample
counts, latent-buffer writes and PyTorch RNG state match exactly under equal seeds.
Separate checks cover the multi-step F equation, detached B targets, rejected-row
gradients, a single combined F optimizer step and checkpoint switches in both
directions. These checks do not establish simulator throughput or tracking gains.

The 2026-10-27 real-Newton smoke passed with production network dimensions
(one 2048x3 dynamics model), default torch.compile/BF16/fused Adam/TF32 settings,
four environments, 84 collector steps, eight-step fitting and 32-step imagination.
It exercised three complete UFO updates, F auxiliary targets, FB/discriminator/pose
actor gradients, physical tracking evaluation, and checkpoint plus replay recovery.
The run explicitly used small batches and relaxed smoke-only trust gates; the
production gate defaults were not changed. Its report is at
`/data/zym/ufo_smoke_20261027/production_math/summary.json`.
This establishes end-to-end connectivity, not full-batch memory requirements,
multi-GPU throughput, convergence or tracking improvement.

The historical version-9, 512-wide, FB-only production-network H32 sequence smoke completed the full 2048 x 10 fitting
and 4096-trajectory update. The measured complete agent update took 44.71 seconds
with a PyTorch peak allocation of 3.33 GiB on this machine. These measurements use
four simulator environments and exclude Newton/Warp allocations; they are not a
full-training memory budget or throughput estimate, and do not characterize the
new 2048-wide model or discriminator scoring. The random-model smoke
explicitly relaxes trust thresholds to check gradient connectivity. Normal
training retains the default thresholds and warmup.

`check_ufo_world_model_derivatives.py` compares learned action-response directions
with paired real Newton rollouts at H1/4/8/16/32. Each branch reconstructs a complete
fresh simulator from the same seed and prelude. Native integration state (including
solver warmstart), nested Newton states/controls and physical parameters must match, and
an independent baseline replay must reproduce before a derivative report is valid.
It uses GPU-to-GPU deterministic kernels, forward-mode JVP, multiple finite-difference
magnitudes and measured contact groups. JVP locally disables TorchScript's optimized
execution path to support CUDA dual tensors; training execution is unchanged.
Only the first action is perturbed; later
actions are fixed. This checks physical response derivatives, not the full closed-loop
actor/return derivative. Checkpoint geometry signatures and branch script versions
must agree. Bounded diagnostic solver capacities are checked after every physics
substep; any overflow or capacity exhaustion aborts the report. No checkpoint means an explicitly untrained diagnostic smoke, not a claim
that the model learned correct derivatives. Empty contact groups report null.

```bash
uv run --no-sync python project_ops/scripts/check_ufo_world_model_derivatives.py \
  --checkpoint /path/to/model.pt --output_dir /tmp/ufo_wm_derivatives \
  --device cuda:0 --visualizer none physics=newton_mjwarp
```
