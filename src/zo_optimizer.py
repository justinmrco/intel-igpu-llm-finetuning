"""Zeroth-order two-sided finite-difference optimizer for LoRA parameters."""
import torch


class ZOOptimizer:
    """
    Two-sided finite-difference ZO optimizer.

    For z ~ N(0, I), the estimator is:
        g = (f(theta + eps*z) - f(theta - eps*z)) / (2*eps) * z

    Temporary perturbations are restored from exact parameter snapshots
    to prevent floating-point drift from repeated add/subtract operations.
    """

    def __init__(
        self,
        params,
        lr,
        eps,
        momentum=0.0,
        normalize_z=True,
        grad_clip=10.0,
    ):
        self.params = params
        self.lr = lr
        self.eps = eps
        self.momentum = momentum
        self.normalize_z = normalize_z
        self.grad_clip = grad_clip

        self.momentum_buffers = {
            n: torch.zeros_like(p, device=p.device)
            for n, p in params
        }

        device = params[0][1].device if params else "cpu"
        self.rng = torch.Generator(device=device)

        self.last_diagnostics = {}

    def set_seed(self, seed):
        self.rng.manual_seed(seed)

    def _sample_perturbation(self):
        zs = {}

        if self.normalize_z:
            raw = {}
            sq_sum = 0.0

            for n, p in self.params:
                z = torch.randn(
                    p.shape,
                    generator=self.rng,
                    device=p.device,
                    dtype=p.dtype,
                )
                raw[n] = z
                sq_sum += z.float().pow(2).sum().item()

            norm = max(sq_sum ** 0.5, 1e-8)

            for n in raw:
                zs[n] = raw[n] / norm

        else:
            for n, p in self.params:
                zs[n] = torch.randn(
                    p.shape,
                    generator=self.rng,
                    device=p.device,
                    dtype=p.dtype,
                )

        return zs

    def step(self, loss_fn):
        zs = self._sample_perturbation()

        originals = {
            n: p.detach().clone()
            for n, p in self.params
        }

        try:
            with torch.no_grad():
                for n, p in self.params:
                    p.copy_(originals[n] + self.eps * zs[n])

            with torch.no_grad():
                loss_plus = loss_fn().item()

            with torch.no_grad():
                for n, p in self.params:
                    p.copy_(originals[n] - self.eps * zs[n])

            with torch.no_grad():
                loss_minus = loss_fn().item()

        finally:
            with torch.no_grad():
                for n, p in self.params:
                    p.copy_(originals[n])

        # Two-sided directional derivative estimate.
        raw_proj_grad = (
            (loss_plus - loss_minus) / (2.0 * self.eps)
        )

        # Clip the scalar projected gradient.
        clipped_proj_grad = max(
            min(raw_proj_grad, self.grad_clip),
            -self.grad_clip,
        )

        was_clipped = (clipped_proj_grad != raw_proj_grad)
        proj_grad = clipped_proj_grad

        # Apply the actual optimizer update.
        update_sq_sum = 0.0

        with torch.no_grad():
            for n, p in self.params:
                g = proj_grad * zs[n]

                if self.momentum > 0:
                    buf = self.momentum_buffers[n]
                    buf.mul_(self.momentum).add_(g)
                    p.add_(buf, alpha=-self.lr)
                    update_sq_sum += (
                        self.lr * buf
                    ).float().pow(2).sum().item()
                else:
                    p.add_(g, alpha=-self.lr)
                    update_sq_sum += (
                        self.lr * g
                    ).float().pow(2).sum().item()

        self.last_diagnostics = {
            "raw_proj_grad": raw_proj_grad,
            "clipped_proj_grad": proj_grad,
            "was_clipped": was_clipped,
            "update_norm": update_sq_sum ** 0.5,
        }

        return loss_plus, loss_minus