import ast, sys, math, torch
B = '/home/zym/roboparty_train/robo_newton/robolab/tasks/manager_based/amp/'
for f in ('mdp/rewards.py', 'rp1_amp_env_cfg.py'):
    ast.parse(open(B + f).read()); print('AST OK', f)
# functional test of gate logic with a fake env
src = open(B + 'mdp/rewards.py').read()
import types
class S: pass
air = torch.tensor([[0.2, 0.25], [0.2, 0.25], [0.0, 0.3], [0.5, 0.4]])
pitch = torch.tensor([0.05, 0.25, 0.0, -0.3])  # rad, + forward
pg = torch.stack([torch.sin(pitch), torch.zeros(4), -torch.cos(pitch)], 1)
sensor = S(); sensor.data = S(); sensor.data.current_air_time = air
robot = S(); robot.data = S(); robot.data.projected_gravity_b = pg
env = S(); env.scene = {'robot': robot}; env.scene = type('Sc', (dict,), {})(robot=robot); env.scene.sensors = {'contact_forces': sensor}
env.command_manager = S(); env.command_manager.get_command = lambda n: torch.tensor([[4.0, 0, 0]] * 4)
sys.path.insert(0, '/home/zym/roboparty_train/robo_newton')
try:
    from robolab.tasks.manager_based.amp.mdp import rewards as R
    print('IMPORT OK (package)')
except Exception as e:
    print('package import failed:', repr(e)[:300]); sys.exit(0)
sc = S(); sc.name = 'contact_forces'; sc.body_ids = [0, 1]
ac = S(); ac.name = 'robot'
out_gate = R.high_speed_double_flight(env, 'x', sc, 1.5, 0.30, 0.17453292519943295, ac)
out_nogate = R.high_speed_double_flight(env, 'x', sc, 1.5, 0.30, None, ac)
print('gate', out_gate.tolist(), 'nogate', out_nogate.tolist())
assert torch.allclose(out_gate, torch.tensor([0.2, 0.0, 0.0, 0.3]))
print('GATE LOGIC OK')
