import math

import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange

def modulate(x, shift, scale):
    """AdaLN-zero modulation"""
    return x * (1 + scale) + shift

class SIGReg(torch.nn.Module):
    """Sketch Isotropic Gaussian Regularizer (single-GPU!)"""

    def __init__(self, knots=17, num_proj=1024):
        super().__init__()
        self.num_proj = num_proj
        t = torch.linspace(0, 3, knots, dtype=torch.float32)
        dt = 3 / (knots - 1)
        weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        window = torch.exp(-t.square() / 2.0)
        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)

    def forward(self, proj):
        """
        proj: (T, B, D)
        """
        # sample random projections
        A = torch.randn(proj.size(-1), self.num_proj, device=proj.device)
        A = A.div_(A.norm(p=2, dim=0))
        # compute the epps-pulley statistic
        x_t = (proj @ A).unsqueeze(-1) * self.t
        err = (x_t.cos().mean(-3) - self.phi).square() + x_t.sin().mean(-3).square()
        statistic = (err @ self.weights) * proj.size(-2)
        return statistic.mean() # average over projections and time


def exponential_sigreg(u, target_rate=1.0):
    """
    Poisson-SIGReg: enforce an Exponential marginal over firing rates.

    u: (B, D) log-rates output by the encoder.
    target_rate: Exponential mean and standard deviation.
    """
    u_clamped = u.clamp(-20.0, 5.0)
    lam = torch.exp(u_clamped)

    B, D = lam.shape
    lam_mean = lam.mean(dim=0)
    target = torch.full_like(lam_mean, target_rate)

    mean_loss = F.mse_loss(lam_mean, target)
    std_loss = F.mse_loss(lam.std(dim=0, unbiased=False), target)

    if B <= 1 or D <= 1:
        cov_loss = lam.new_zeros(())
    else:
        lam_centered = lam - lam_mean
        cov = (lam_centered.T @ lam_centered) / (B - 1)
        off_diag = cov.flatten()[:-1].view(D - 1, D + 1)[:, 1:].flatten()
        cov_loss = off_diag.pow(2).sum() / D

    return mean_loss + std_loss + cov_loss


def poisson_kl_rates(lam_q, lam_p, eps: float = 1e-8):
    """Elementwise KL(Pois(lam_q) || Pois(lam_p)) for positive rates."""
    lam_q = lam_q.clamp_min(eps)
    lam_p = lam_p.clamp_min(eps)
    return lam_q * (torch.log(lam_q) - torch.log(lam_p)) - lam_q + lam_p


def poisson_kl_log_rates(r_q, r_p, lambda0: float = 1.0):
    """Elementwise KL(Pois(lambda0*exp(r_q)) || Pois(lambda0*exp(r_p))).

    This is algebraically identical to ``poisson_kl_rates`` after converting
    residual log-rates to rates, but avoids rate ratios/logs.
    """
    lam_q = float(lambda0) * torch.exp(r_q)
    lam_p = float(lambda0) * torch.exp(r_p)
    return lam_q * (r_q - r_p) - lam_q + lam_p


def two_sample_sigreg(x, target, knots: int = 17, num_proj: int = 1024):
    """SIGReg-style random-projection characteristic-function matching."""
    if x.shape != target.shape:
        raise ValueError(f"two_sample_sigreg shape mismatch: {x.shape} vs {target.shape}")
    if x.ndim != 2:
        raise ValueError(f"two_sample_sigreg expects [B, D], got {x.shape}")

    B, D = x.shape
    device = x.device
    dtype = x.dtype
    t = torch.linspace(0, 3, knots, device=device, dtype=dtype)
    dt = 3 / (knots - 1)
    weights = torch.full((knots,), 2 * dt, device=device, dtype=dtype)
    weights[[0, -1]] = dt
    weights = weights * torch.exp(-t.square() / 2.0)

    proj = torch.randn(D, num_proj, device=device, dtype=dtype)
    proj = proj / proj.norm(p=2, dim=0, keepdim=True).clamp_min(1e-8)

    x_t = (x @ proj).unsqueeze(-1) * t
    target_t = (target @ proj).unsqueeze(-1) * t
    err = (
        (x_t.cos().mean(dim=0) - target_t.cos().mean(dim=0)).square()
        + (x_t.sin().mean(dim=0) - target_t.sin().mean(dim=0)).square()
    )
    return ((err @ weights) * B).mean()


def sample_capacity_rates(rates, pi, shape, device, dtype):
    rates_t = torch.as_tensor(rates, device=device, dtype=dtype)
    pi_t = torch.as_tensor(pi, device=device, dtype=dtype)
    pi_t = pi_t / pi_t.sum().clamp_min(1e-12)
    idx = torch.multinomial(pi_t, num_samples=shape[0] * shape[1], replacement=True)
    return rates_t[idx].reshape(shape)


def scalar_poisson_mi(lam, nmax: int, eps: float = 1e-8):
    """Deterministic per-coordinate scalar Poisson-channel MI diagnostic."""
    B, _D = lam.shape
    with torch.amp.autocast(device_type=lam.device.type, enabled=False):
        lam = lam.float().clamp_min(eps)
        n = torch.arange(nmax + 1, device=lam.device, dtype=lam.dtype)
        log_fact = torch.lgamma(n + 1.0)
        log_lam = torch.log(lam)
        logW = (
            n.view(1, 1, -1) * log_lam.unsqueeze(-1)
            - lam.unsqueeze(-1)
            - log_fact.view(1, 1, -1)
        )
        logW = logW - torch.logsumexp(logW, dim=-1, keepdim=True)
        W = torch.exp(logW)
        logq = torch.logsumexp(logW, dim=0) - math.log(B)
        return (W * (logW - logq.unsqueeze(0))).sum(dim=-1).mean(dim=0)


def effective_rank(x, eps: float = 1e-8):
    """Entropy effective rank of centered features, computed in float32."""
    if x.shape[0] <= 1 or x.shape[1] <= 1:
        return x.new_tensor(1.0)
    with torch.amp.autocast(device_type=x.device.type, enabled=False):
        y = x.float() - x.float().mean(dim=0, keepdim=True)
        cov = (y.T @ y) / max(1, y.shape[0] - 1)
        eigvals = torch.linalg.eigvalsh(cov).clamp_min(0.0)
        probs = eigvals / eigvals.sum().clamp_min(eps)
        entropy = -(probs * torch.log(probs.clamp_min(eps))).sum()
        return torch.exp(entropy)


def make_rate_grid(mu: float, A: float, K: int = 512):
    import numpy as np

    eps = max(1e-6, 1e-4 * mu)
    grid = np.concatenate([
        np.array([0.0]),
        np.geomspace(eps, A, K // 2),
        np.linspace(0.0, A, K // 2),
    ])
    return np.unique(np.sort(grid))


def poisson_channel_matrix(rates, nmax: int):
    import numpy as np
    from scipy.special import gammaln

    rates = np.asarray(rates, dtype=np.float64)
    n = np.arange(nmax + 1, dtype=np.float64)
    W = np.zeros((len(rates), nmax + 1), dtype=np.float64)

    positive = rates > 0
    r = rates[positive]
    logW = (
        n[None, :] * np.log(r[:, None])
        - r[:, None]
        - gammaln(n[None, :] + 1.0)
    )
    W[positive] = np.exp(logW)
    W[~positive, 0] = 1.0

    tail = np.clip(1.0 - W.sum(axis=1, keepdims=True), 0.0, 1.0)
    W = np.concatenate([W, tail], axis=1)
    W = np.clip(W, 1e-300, 1.0)
    return W / W.sum(axis=1, keepdims=True)


def blahut_arimoto_cost(W, cost, beta: float, n_iter: int = 5000, tol: float = 1e-12):
    import numpy as np
    from scipy.special import logsumexp

    K = W.shape[0]
    pi = np.ones(K, dtype=np.float64) / K
    logW = np.log(W)

    for _ in range(n_iter):
        q = pi @ W
        D = np.sum(W * (logW - np.log(q)[None, :]), axis=1)
        logits = np.log(pi) + D - beta * cost
        logits = logits - logsumexp(logits)
        pi_new = np.exp(logits)
        if np.max(np.abs(pi_new - pi)) < tol:
            pi = pi_new
            break
        pi = pi_new

    q = pi @ W
    I = np.sum(pi[:, None] * W * (logW - np.log(q)[None, :]))
    mean_cost = np.sum(pi * cost)
    return pi, I, mean_cost


def solve_poisson_capacity_prior(mu: float, A: float, K: int = 512):
    import numpy as np

    rates = make_rate_grid(mu, A, K=K)
    nmax = int(np.ceil(A + 10.0 * np.sqrt(A + 1.0) + 20.0))
    W = poisson_channel_matrix(rates, nmax)

    pi0, C0, m0 = blahut_arimoto_cost(W, rates, beta=0.0)
    if m0 <= mu:
        return rates, pi0, C0, nmax

    lo, hi = 0.0, 1.0
    while True:
        _, _, m_hi = blahut_arimoto_cost(W, rates, beta=hi, n_iter=1500)
        if m_hi <= mu:
            break
        hi *= 2.0

    best = None
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        pi, C, m = blahut_arimoto_cost(W, rates, beta=mid, n_iter=2500)
        best = (pi, C, m, mid)
        if m > mu:
            lo = mid
        else:
            hi = mid

    pi, _C, _m, beta = best
    pi, C, _m = blahut_arimoto_cost(W, rates, beta=beta, n_iter=8000)
    return rates, pi, C, nmax
    
class FeedForward(nn.Module):
    """FeedForward network used in Transformers"""

    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    """Scaled dot-product attention with causal masking"""

    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)
        self.heads = heads
        self.scale = dim_head**-0.5
        self.dropout = dropout
        self.norm = nn.LayerNorm(dim)
        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = (
            nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
            if project_out
            else nn.Identity()
        )

    def forward(self, x, causal=True):
        """
        x : (B, T, D)
        """
        x = self.norm(x)
        drop = self.dropout if self.training else 0.0
        qkv = self.to_qkv(x).chunk(3, dim=-1)  # q, k, v: (B, heads, T, dim_head)
        q, k, v = (rearrange(t, "b t (h d) -> b h t d", h=self.heads) for t in qkv)
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=drop, is_causal=causal)
        out = rearrange(out, "b h t d -> b t (h d)")
        return self.to_out(out)


class ConditionalBlock(nn.Module):
    """Transformer block with AdaLN-zero conditioning"""

    def __init__(self, dim, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()

        self.attn = Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)
        self.mlp = FeedForward(dim, mlp_dim, dropout=dropout)
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(dim, 6 * dim, bias=True)
        )

        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c).chunk(6, dim=-1)
        )
        x = x + gate_msa * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class Block(nn.Module):
    """Standard Transformer block"""

    def __init__(self, dim, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()

        self.attn = Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)
        self.mlp = FeedForward(dim, mlp_dim, dropout=dropout)
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class Transformer(nn.Module):
    """Standard Transformer with support for AdaLN-zero blocks"""

    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        depth,
        heads,
        dim_head,
        mlp_dim,
        dropout=0.0,
        block_class=Block,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.layers = nn.ModuleList([])

        self.input_proj = (
            nn.Linear(input_dim, hidden_dim)
            if input_dim != hidden_dim
            else nn.Identity()
        )

        self.cond_proj = (
            nn.Linear(input_dim, hidden_dim)
            if input_dim != hidden_dim
            else nn.Identity()
        )

        self.output_proj = (
            nn.Linear(hidden_dim, output_dim)
            if hidden_dim != output_dim
            else nn.Identity()
        )

        for _ in range(depth):
            self.layers.append(
                block_class(hidden_dim, heads, dim_head, mlp_dim, dropout)
            )

    def forward(self, x, c=None):

        if hasattr(self, "input_proj"):
            x = self.input_proj(x)

        if c is not None and hasattr(self, "cond_proj"):
            c = self.cond_proj(c)

        for block in self.layers:
            x = block(x) if isinstance(block, Block) else block(x, c)
        x = self.norm(x)

        if hasattr(self, "output_proj"):
            x = self.output_proj(x)
        return x

class Embedder(nn.Module):
    def __init__(
        self,
        input_dim=10,
        smoothed_dim=10,
        emb_dim=10,
        mlp_scale=4,
    ):
        super().__init__()
        self.patch_embed = nn.Conv1d(input_dim, smoothed_dim, kernel_size=1, stride=1)
        self.embed = nn.Sequential(
            nn.Linear(smoothed_dim, mlp_scale * emb_dim),
            nn.SiLU(),
            nn.Linear(mlp_scale * emb_dim, emb_dim),
        )

    def forward(self, x):
        """
        x: (B, T, D)
        """
        x = x.float()
        x = x.permute(0, 2, 1)
        x = self.patch_embed(x)
        x = x.permute(0, 2, 1)
        x = self.embed(x)
        return x


class MLP(nn.Module):
    """Simple MLP with optional normalization and activation"""

    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim=None,
        norm_fn=nn.LayerNorm,
        act_fn=nn.GELU,
    ):
        super().__init__()
        norm_fn = norm_fn(hidden_dim) if norm_fn is not None else nn.Identity()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            norm_fn,
            act_fn(),
            nn.Linear(hidden_dim, output_dim or input_dim),
        )

    def forward(self, x):
        """
        x: (B*T, D)
        """
        return self.net(x)


class ARPredictor(nn.Module):
    """Autoregressive predictor for next-step embedding prediction."""

    def __init__(
        self,
        *,
        num_frames,
        depth,
        heads,
        mlp_dim,
        input_dim,
        hidden_dim,
        output_dim=None,
        dim_head=64,
        dropout=0.0,
        emb_dropout=0.0,
    ):
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.randn(1, num_frames, input_dim))
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer(
            input_dim,
            hidden_dim,
            output_dim or input_dim,
            depth,
            heads,
            dim_head,
            mlp_dim,
            dropout,
            block_class=ConditionalBlock,
        )

    def forward(self, x, c):
        """
        x: (B, T, d)
        c: (B, T, act_dim)
        """
        T = x.size(1)
        x = x + self.pos_embedding[:, :T]
        x = self.dropout(x)
        x = self.transformer(x, c)
        return x
