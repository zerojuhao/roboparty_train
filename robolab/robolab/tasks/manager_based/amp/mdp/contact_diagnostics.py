"""Episode peak foot load in body weights, independent of reward terms and reward scaling."""
import torch


class FootContactDiagnostics:
    def __init__(self, env):
        self.env = env
        params = env.reward_manager.get_term_cfg('feet_contact_force_l2').params
        self.sensor = env.scene.sensors[params['sensor_cfg'].name]
        self.body_ids = params['sensor_cfg'].body_ids
        self.asset = env.scene[params['asset_cfg'].name]
        self.samples = torch.zeros(env.num_envs, device=env.device)
        self.peak_sum = torch.zeros_like(self.samples)
        self.peak_max = torch.zeros_like(self.samples)

    def sample(self):
        cfg = self.env.cfg
        force_z = self.sensor.data.net_forces_w_history.torch[:, :cfg.decimation, self.body_ids, 2]
        peak = force_z.amax(dim=(1, 2)) / (self.asset.data.body_mass.torch.sum(dim=1) * abs(cfg.sim.gravity[2]))
        self.samples += 1
        self.peak_sum += peak
        torch.maximum(self.peak_max, peak, out=self.peak_max)

    def reset(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        valid = self.samples[ids] > 0
        if valid.any():
            peak = self.peak_max[ids][valid]
            log = self.env.extras.setdefault('log', {})
            log['Contact/foot_force_peak_bw_mean'] = self.peak_sum[ids].sum() / self.samples[ids].sum()
            log['Contact/foot_force_peak_bw_p95'] = torch.quantile(peak, .95)
            log['Contact/foot_force_peak_bw_max'] = peak.max()
        for values in (self.samples, self.peak_sum, self.peak_max):
            values[ids] = 0
