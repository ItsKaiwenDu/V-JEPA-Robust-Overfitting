"""Independent analytic CPU controls. These never represent native route evidence."""
import torch
from attacks.pgd import native_objective
from models.vjepa2 import MaskPair

KIND = 'analytic-local-control-v1'


class AnalyticModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = torch.nn.Linear(4, 4, bias=False)
        self.predictor = torch.nn.Linear(4, 4, bias=False)
        self.target_encoder = torch.nn.Linear(4, 4, bias=False)
        with torch.no_grad():
            for module in (self.encoder, self.predictor, self.target_encoder):
                module.weight.copy_(torch.eye(4))
        self.target_encoder.requires_grad_(False)
        self.target_encoder.eval()

    def clean_targets(self, clean):
        with torch.no_grad():
            return self.target_encoder(clean.flatten(1)).detach()

    def loss_per_example(self, pixels, targets, masks):
        value = self.predictor(self.encoder(pixels.flatten(1)))
        # Two smooth mask-dependent terms, including interior selection optima.
        return torch.stack([-(value[:, p.target[0]]-(targets[:, p.target[0]]+.006)).square().mean(1)
                            + .0002*torch.sin(value[:, p.context[0]]*17).mean(1)
                            for p in masks]).mean(0)


def analytic_fixture():
    # nn.Linear construction consumes RNG; this control restores the caller's state.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(2026112301)
        model = AnalyticModel()
    model.eval()
    pixels = torch.tensor([0., .35, .65, 1.], dtype=torch.float32).reshape(1, 1, 2, 2)

    def masks_for(seed):
        order = torch.randperm(4, generator=torch.Generator(device='cpu').manual_seed(seed % (2**31-1)))
        return [MaskPair(order[:1].reshape(1, -1), order[1:].reshape(1, -1)),
                MaskPair(order[:3].reshape(1, -1), order[3:].reshape(1, -1))]
    return model, pixels, masks_for


class AnalyticRoute:
    def __init__(self, model):
        if type(model) is not AnalyticModel or any(p.device.type != 'cpu' for p in model.parameters()):
            raise ValueError('Only the explicit four-element analytic CPU control model.')
        self.model, self.active, self.entered, self.used = model, False, False, False
        self.tokens = None

    def __enter__(self):
        if self.entered:
            raise ValueError('Analytic control route is single use.')
        self.entered = self.active = True
        return self

    def loss(self, value, targets, bank):
        if not self.active or self.tokens is not None:
            raise ValueError('One analytic root only.')
        root = native_objective(self.model, targets, bank)(value).sum()
        self.tokens = root, value
        return root

    def gradient(self, root, value):
        if (not self.active or self.used or self.tokens is None
                or root is not self.tokens[0] or value is not self.tokens[1]):
            raise ValueError('One exact analytic root/input pair required.')
        self.used = True
        result, = torch.autograd.grad(root, value)
        self.tokens = None
        return result

    def __exit__(self, *unused):
        self.tokens, self.active = None, False
        return False


class AnalyticFactory:
    kind = KIND

    def create(self, model):
        return AnalyticRoute(model)

    def observe(self, route):
        if not route.used or route.tokens is not None:
            raise ValueError('Incomplete analytic control.')
        return {'kind': KIND, 'analytic_autograd_only': True, 'native_ownership_proven': False}

    def idle(self, model):
        if type(model) is not AnalyticModel:
            raise ValueError('Analytic CPU control model required.')
