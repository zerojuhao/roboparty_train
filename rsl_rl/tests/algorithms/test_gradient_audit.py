import importlib.util
from pathlib import Path

import torch


spec = importlib.util.spec_from_file_location(
    'gradient_audit', Path(__file__).parents[2] / 'rsl_rl/algorithms/gradient_audit.py'
)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def test_parameter_audit_includes_separate_forwards_without_changing_backward():
    weight = torch.nn.Parameter(torch.tensor([1., 2.]))
    unused = torch.nn.Parameter(torch.tensor([3.]))
    terms = {'fb': weight.sum(), 'velocity': 2 * weight.sum(), 'estimator': -weight.sum()}
    weight.grad = torch.tensor([7., 8.])
    metrics = audit.parameter_gradient_metrics(terms, [weight, unused])
    torch.testing.assert_close(weight.grad, torch.tensor([7., 8.]))
    assert unused.grad is None
    torch.testing.assert_close(metrics['depth_gradient_local/cosine/actor_total_estimator'], torch.tensor(-1.))
    torch.testing.assert_close(metrics['depth_gradient_local/estimator_actor_ratio'], torch.tensor(1 / 3))
    weight.grad = None
    sum(terms.values()).backward()
    torch.testing.assert_close(weight.grad, torch.tensor([2., 2.]))


def test_zero_weight_loss_has_finite_diagnostics():
    weight = torch.nn.Parameter(torch.ones(2))
    terms = {'fb': weight.sum() * 0, 'estimator': weight.square().sum() * 0}
    assert all(torch.isfinite(v) for v in audit.parameter_gradient_metrics(terms, [weight]).values())


def test_compiled_audit_preserves_backward():
    audit.enable_compiled_gradient_audit()
    module = torch.nn.Linear(3, 2)
    compiled = torch.compile(module, backend='aot_eager')
    x = torch.ones(4, 3)
    first, second = compiled(x), compiled(2 * x)
    terms = {'fb': first.square().mean(), 'velocity': second.square().mean(),
             'estimator': first.mean()}
    metrics = audit.parameter_gradient_metrics(terms, module.parameters())
    assert all(torch.isfinite(v) for v in metrics.values())
    sum(terms.values()).backward()
    measured = module.weight.grad.clone()
    module.zero_grad()
    (module(x).square().mean() + module(2 * x).square().mean() + module(x).mean()).backward()
    torch.testing.assert_close(measured, module.weight.grad)


def test_diagnostic_temporarily_unwraps_and_restores_compiled_forward():
    class Agent:
        def __init__(self):
            self.net = torch.nn.Linear(3, 2)
            self.net.forward = torch.compile(self.net.forward, backend='aot_eager')

        @audit.eager_gradient_diagnostics
        def update(self, *, depth_gradient_diagnostics):
            assert not hasattr(self.net.forward, '_torchdynamo_orig_callable')
            x = torch.ones(4, 3)
            terms = {'fb': self.net(x).square().mean(), 'estimator': self.net(x * 2).mean()}
            audit.parameter_gradient_metrics(terms, self.net.parameters())
            sum(terms.values()).backward()

    agent = Agent()
    forward = agent.net.forward
    agent.update(depth_gradient_diagnostics=True)
    assert agent.net.forward is forward
