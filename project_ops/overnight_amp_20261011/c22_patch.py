#!/usr/bin/env python3
"""C22: softer lean gate 10° → 15° on high_speed_double_flight. Backup first."""
import shutil, time, math
ts = time.strftime('%Y%m%d_%H%M%S')
B = '/home/zym/roboparty_train/robo_newton/robolab/tasks/manager_based/amp/'
R = B + 'mdp/rewards.py'
C = B + 'rp1_amp_env_cfg.py'
BK = '/data/zym/roboparty_train/backups/'
for f in (R, C):
    shutil.copy2(f, f + '.bak_c22_' + ts)
    shutil.copy2(f, BK + f.split('/')[-1] + '.bak_c22_' + ts)
print('TS', ts)
print('BACKUPS', R + '.bak_c22_' + ts, C + '.bak_c22_' + ts)

r = open(R).read()
assert 'C22_LEAN_GATE_15' not in r
assert 'max_forward_lean' in r
# Update marker comments in rewards.py (logic unchanged; only threshold via cfg)
r = r.replace(
    'C21_LEAN_GATED_FLIGHT: if ``max_forward_lean`` (radians) is set, the reward',
    'C22_LEAN_GATE_15: if ``max_forward_lean`` (radians) is set, the reward',
)
r = r.replace(
    '# C21_LEAN_GATED_FLIGHT: zero flight credit when forward lean > max_forward_lean.',
    '# C22_LEAN_GATE_15: zero flight credit when forward lean > max_forward_lean (C22 default 15°).',
)
open(R, 'w').write(r)

c = open(C).read()
assert 'C22_LEAN_GATE_15' not in c
old_val = '0.17453292519943295'
assert c.count(f'"max_forward_lean": {old_val}') == 1
new_val = '0.2617993877991494'  # 15 deg
c = c.replace(
    f'"max_forward_lean": {old_val}',
    f'"max_forward_lean": {new_val}',
)
c = c.replace(
    '# C21_LEAN_GATED_FLIGHT: + max_forward_lean=10deg gate (weight 4.0, clamp 0.30 unchanged). Reversible: restore bak_c21_*.\n    high_speed_double_flight',
    '# C22_LEAN_GATE_15: max_forward_lean 10deg→15deg (weight 4.0, clamp 0.30 unchanged). Reversible: restore bak_c22_*.\n    high_speed_double_flight',
)
c = c.replace(
    '# C21_LEAN_GATED_FLIGHT: zero flight reward when forward lean > 10 deg.',
    '# C22_LEAN_GATE_15: zero flight reward when forward lean > 15 deg.',
)
open(C, 'w').write(c)

# Verify
c2 = open(C).read()
assert f'"max_forward_lean": {new_val}' in c2
assert old_val not in c2.split('high_speed_double_flight')[1].split('RP1AmpPathEnvCfg')[0]
assert 'C22_LEAN_GATE_15' in open(R).read()
assert 'C22_LEAN_GATE_15' in c2
assert abs(float(new_val) - math.radians(15)) < 1e-12
print('PATCHED max_forward_lean ->', new_val, '(15 deg)')
print('OK')
