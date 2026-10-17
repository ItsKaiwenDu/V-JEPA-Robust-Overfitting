import pytest
import torch

from attacks.pgd import fgsm, native_objective, pgd_linf


@pytest.mark.parametrize("epsilon", [0, 1/255, 4/255, 8/255])
def test_projection_and_zero_epsilon(epsilon):
    clean = torch.tensor([[0., 0.5, 1.], [1., 0.5, 0.]])
    result = pgd_linf(lambda x: x.square().sum(1), clean, epsilon=epsilon,
                      step_size=1/255, steps=5, restarts=2)
    assert (result.adversarial >= 0).all() and (result.adversarial <= 1).all()
    assert (result.adversarial - clean).abs().max() <= epsilon + 1e-7
    assert not result.adversarial.requires_grad
    if epsilon == 0:
        assert torch.equal(result.adversarial, clean)


def test_fgsm_attains_linear_optimum_for_each_example():
    clean = torch.full((2, 3), 0.5)
    weights = torch.tensor([[1., -2., 3.], [-4., 5., -6.]])
    result = fgsm(lambda x: (weights * x).sum(1), clean, epsilon=0.1)
    torch.testing.assert_close(result.adversarial, clean + 0.1 * weights.sign())


def test_additional_steps_and_restarts_preserve_per_example_best():
    clean = torch.full((4, 5), 0.5)
    objective = lambda x: torch.sin(20 * x).sum(1)
    short = pgd_linf(objective, clean, epsilon=0.3, step_size=0.03, steps=2, seed=7)
    long = pgd_linf(objective, clean, epsilon=0.3, step_size=0.03, steps=8, seed=7)
    multiple = pgd_linf(objective, clean, epsilon=0.3, step_size=0.03, steps=8, seed=7, restarts=3)
    assert (long.losses >= short.losses).all()
    assert (multiple.losses >= long.losses).all()
    torch.testing.assert_close(multiple.losses, objective(multiple.adversarial))


def test_replay_rng_and_parameter_gradients_are_preserved():
    weight = torch.nn.Parameter(torch.tensor([1., -1., 2.]))
    weight.grad = torch.tensor([8., 7., 6.])
    before = weight.grad.clone()
    rng = torch.random.get_rng_state().clone()
    objective = lambda x: (x * weight).sum(1)
    kwargs = dict(epsilon=0.1, step_size=0.03, steps=3, seed=19)
    first = pgd_linf(objective, torch.full((2, 3), 0.5), **kwargs)
    second = pgd_linf(objective, torch.full((2, 3), 0.5), **kwargs)
    assert torch.equal(first.adversarial, second.adversarial)
    assert torch.equal(weight.grad, before)
    assert torch.equal(rng, torch.random.get_rng_state())


def test_eot_averages_per_example_objectives_before_differentiation():
    class Model:
        def loss_per_example(self, pixels, targets, masks):
            return (pixels * masks[0]).sum(1) + targets[:, 0]
    targets = torch.tensor([[2.], [3.]])
    objective = native_objective(Model(), targets, [[torch.tensor([1., 3.])], [torch.tensor([-3., 1.])]])
    pixels = torch.full((2, 2), 0.5, requires_grad=True)
    gradient, = torch.autograd.grad(objective(pixels).sum(), pixels)
    torch.testing.assert_close(gradient, torch.tensor([[-1., 2.], [-1., 2.]]))


def test_bad_objective_shape_fails():
    with pytest.raises(ValueError, match="one loss per example"):
        pgd_linf(lambda x: x.sum(), torch.full((2, 3), 0.5), epsilon=0.1, step_size=0.1, steps=1)
