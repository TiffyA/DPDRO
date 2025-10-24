import torch


class DPSGDA(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        loss_fn,
        lr_w,
        lr_v,
        sigma_w,
        sigma_v,
        clip_w=None,
        clip_v=None,
    ):
        try:
            self.a = loss_fn.a
            self.b = loss_fn.b
            self.alpha = loss_fn.alpha
        except AttributeError:
            self.a = None
            self.b = None
            self.alpha = None

        primal_params = list(params)
        if self.a is not None and self.b is not None:
            primal_params.extend([self.a, self.b])
        dual_params = []
        if self.alpha is not None:
            dual_params.append(self.alpha)

        self.lr_w = lr_w
        self.lr_v = lr_v
        self.sigma_w = sigma_w
        self.sigma_v = sigma_v
        self.clip_w = clip_w
        self.clip_v = clip_v
        self._primal_params = primal_params
        self._dual_params = dual_params

        defaults = dict()
        super().__init__(primal_params + dual_params, defaults)

    def _clip_tensor(self, grad, clip_value):
        if clip_value is None or clip_value <= 0:
            return grad
        norm = torch.linalg.norm(grad)
        if norm <= clip_value or norm == 0:
            return grad
        return grad * (clip_value / (norm + 1e-12))

    def zero_grad(self, set_to_none: bool = True) -> None:
        if self.alpha is not None:
            if set_to_none:
                self.alpha.grad = None
            elif self.alpha.grad is not None:
                self.alpha.grad.zero_()
        return super().zero_grad(set_to_none)

    @torch.no_grad()
    def step(self, closure=None):
        if closure is not None:
            with torch.enable_grad():
                closure()

        for param in self._primal_params:
            if param.grad is None:
                continue
            grad = param.grad.detach()
            grad = self._clip_tensor(grad, self.clip_w)
            noise = torch.randn_like(param) * self.sigma_w
            update = grad + noise
            param.add_(update, alpha=-self.lr_w)

        for param in self._dual_params:
            if param.grad is None:
                continue
            grad = param.grad.detach()
            grad = self._clip_tensor(grad, self.clip_v)
            noise = torch.randn_like(param) * self.sigma_v
            update = grad + noise
            param.add_(update, alpha=self.lr_v)

        self.zero_grad()
