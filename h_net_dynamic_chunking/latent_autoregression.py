import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch.nn import Module

from einops import rearrange
from torch_einops_utils import masked_mean

# helpers

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def cast_tuple(t):
    return t if type(t) is tuple else (t,)

# restricted backward wrapper

class RestrictedBackwardLoss:
    def __init__(self, loss: Tensor, modules):
        self.loss = loss
        self.modules = cast_tuple(modules)

    def backward(self, **kwargs):
        params = [p for m in self.modules if exists(m) for p in m.parameters() if p.requires_grad]
        self.loss.backward(inputs = params, retain_graph = True, **kwargs)

    def item(self):
        return self.loss.item()

    def __add__(self, other):
        raise ValueError('RestrictedBackwardLoss cannot be added to other losses. Call .backward() on it directly.')

    def __radd__(self, other):
        return self.__add__(other)

# distributional regularization

def sigreg_loss(
    x: Tensor,
    mask: Tensor | None = None,
    num_slices: int = 1024,
    domain: tuple[float, float] = (-5., 5.),
    num_knots: int = 17
) -> Tensor:
    # distributional regularization via empirical characteristic function
    # Randall Balestriero - https://arxiv.org/abs/2511.08544

    dim, device = x.shape[-1], x.device

    if exists(mask):
        x = x[mask]
    else:
        x = rearrange(x, '... d -> (...) d')

    # slice sampling

    rand_projs = torch.randn((num_slices, dim), device = device)
    rand_projs = F.normalize(rand_projs, dim = -1, eps = 1e-6)

    # integration points

    t = torch.linspace(*domain, num_knots, device = device)

    # theoretical cf for n(0, 1) and gauss window

    exp_f = (-0.5 * t.square()).exp()

    # empirical cf

    x_t = torch.einsum('n d, m d -> n m', x, rand_projs)
    x_t = rearrange(x_t, 'n m -> n m 1') * t
    ecf = (1j * x_t).exp().mean(dim = 0)

    # weighted l2 distance

    err = ecf.sub(exp_f).abs().square().mul(exp_f)

    return torch.trapezoid(err, t, dim = -1).mean()

# main class

class LatentAutoregressiveLoss(Module):
    def __init__(
        self,
        dim,
        use_rmsnorm = False,
        sigreg_loss_kwargs: dict | None = None,
        net: Module | None = None
    ):
        super().__init__()
        self.sigreg_loss_kwargs = default(sigreg_loss_kwargs, dict())

        if not exists(net):
            norm = nn.RMSNorm(dim) if use_rmsnorm else nn.Identity()

            net = nn.Sequential(
                norm,
                nn.Linear(dim, dim * 4),
                nn.SiLU(),
                nn.Linear(dim * 4, dim)
            )

        self.net = net

    def forward(
        self,
        x,
        mask = None
    ):
        pred_input, target = x[:, :-1], x[:, 1:]

        pred = self.net(pred_input)

        loss = F.mse_loss(pred, target, reduction = 'none')

        mask_out = mask[:, 1:] if exists(mask) else None

        loss = masked_mean(loss, mask_out)

        sigreg_loss_val = sigreg_loss(
            target,
            mask = mask_out,
            **self.sigreg_loss_kwargs
        )

        return loss, sigreg_loss_val, pred
