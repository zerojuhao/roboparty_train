import shutil, time, re, sys
ts = time.strftime('%Y%m%d_%H%M%S')
B = '/home/zym/roboparty_train/robo_newton/robolab/tasks/manager_based/amp/'
R = B + 'mdp/rewards.py'; C = B + 'rp1_amp_env_cfg.py'
for f in (R, C):
    shutil.copy2(f, f + '.bak_c21_' + ts)
    shutil.copy2(f, '/data/zym/roboparty_train/backups/' + f.split('/')[-1] + '.bak_c21_' + ts)
print('TS', ts)
r = open(R).read()
assert 'C21_LEAN_GATED_FLIGHT' not in r
old_sig = '''    speed_gate: float = 1.5,
    max_flight_time: float = 0.30,
) -> torch.Tensor:
    """Reward double-flight (both feet airborne) at high commanded XY speed.
'''
new_sig = '''    speed_gate: float = 1.5,
    max_flight_time: float = 0.30,
    max_forward_lean: float | None = None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward double-flight (both feet airborne) at high commanded XY speed.

    C21_LEAN_GATED_FLIGHT: if ``max_forward_lean`` (radians) is set, the reward
    is zeroed whenever body forward lean exceeds it. Forward lean uses the same
    convention as ``lean_orientation_l2`` / ``asymmetric_pitch_roll_penalty``:
    root ``projected_gravity_b[:, 0] > 0`` is forward lean, compared against
    ``sin(max_forward_lean)``. ``None`` disables the gate (C20 behaviour).
'''
assert r.count(old_sig) == 1
r = r.replace(old_sig, new_sig)
old_tail = '''    reward = reward * (cmd_speed > float(speed_gate)).float()
    return reward
'''
new_tail = '''    reward = reward * (cmd_speed > float(speed_gate)).float()
    # C21_LEAN_GATED_FLIGHT: zero flight credit when forward lean > max_forward_lean.
    if max_forward_lean is not None:
        asset: RigidObject = env.scene[asset_cfg.name]
        projected_gravity = _as_torch(asset.data.projected_gravity_b)
        forward = torch.clamp(projected_gravity[:, 0], min=0.0)
        lean_ok = forward <= math.sin(float(max_forward_lean))
        reward = reward * lean_ok.float()
    return reward
'''
assert r.count(old_tail) == 1
r = r.replace(old_tail, new_tail)
open(R, 'w').write(r)
c = open(C).read()
old_c = '''            "speed_gate": 1.5,
            "max_flight_time": 0.30,
        },
    )'''
new_c = '''            "speed_gate": 1.5,
            "max_flight_time": 0.30,
            # C21_LEAN_GATED_FLIGHT: zero flight reward when forward lean > 10 deg.
            "max_forward_lean": 0.17453292519943295,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )'''
assert c.count(old_c) == 1
c = c.replace(old_c, new_c)
c = c.replace("    # Reversible: restore bak_c20_*.\n    high_speed_double_flight",
  "    # C21_LEAN_GATED_FLIGHT: + max_forward_lean=10deg gate (weight 4.0, clamp 0.30 unchanged). Reversible: restore bak_c21_*.\n    high_speed_double_flight")
open(C, 'w').write(c)
print('PATCHED')
