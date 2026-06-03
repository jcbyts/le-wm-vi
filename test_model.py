"""Smoke test for FONDJEPA (variants 3-6): shapes + gradient routing, no heavy
deps. Stub predictor/action_encoder match ARPredictor.forward(x,c)->(B,T,P) and
Embedder.forward(x)->(B,T,P), so this hits the exact forward path train.py will.

Verifies, for BOTH families (poisson, gaussian) and BOTH loss forms:
  - encode() runs inference, emb has shape (B,T,P)  (P=D poisson, 2D gaussian)
  - predict() -> (B,ctx_len,P); decode() -> (B,T,C,hw,hw) in [0,1]
  - vijepa_forward gives finite pred_loss + recon_loss + diagnostics
  - gradients reach decoder (recon), predictor (pred_loss), prior_param
  - DECODER gets ~0 gradient from the predictive term alone (stop-grad target +
    detached inference) — spec §2.3 routing property, per family.
"""

import types
import torch
from torch import nn

from model import FONDJEPA, ConvDecoder, filter_forward, vijepa_forward
from vit_decoder import ViTDecoder
from latent import make_head


class StubPredictor(nn.Module):
    def __init__(self, p):
        super().__init__()
        self.net = nn.Linear(p, p)
        self.cond = nn.Linear(p, p)
    def forward(self, x, c):
        return self.net(x) + self.cond(c)


class StubActionEncoder(nn.Module):
    def __init__(self, a, p):
        super().__init__()
        self.lin = nn.Linear(a, p)
    def forward(self, x):
        return self.lin(x.float())


def build(family, D=192, img_hw=32, img_ch=3, grid=8, infer_backprop=False, k_inner=3):
    head = make_head(family)
    P = D * head.param_mult
    dec = ConvDecoder(D, img_ch=img_ch, img_hw=img_hw, grid=grid)
    return FONDJEPA(
        decoder=dec,
        predictor=StubPredictor(P),
        action_encoder=StubActionEncoder(a=4, p=P),
        latent_dim=D, head=head,
        k_inner=k_inner, tau=0.2, infer_lr=1.0, infer_backprop=infer_backprop,
        k_bptt=k_inner,
        img_ch=img_ch, img_hw=img_hw,
    )


def make_module(model):
    m = types.SimpleNamespace()
    m.model = model
    m.log_dict = lambda *a, **k: None
    return m


def make_cfg(pred_loss, **overrides):
    loss = {
        "beta": 1.0,
        "pred_loss": pred_loss,
        "infer_backprop": False,
        "k_bptt": 3,
    }
    loss.update(overrides)
    loss_obj = types.SimpleNamespace(**loss)
    loss_obj.get = lambda k, d=None: loss.get(k, d)
    return types.SimpleNamespace(
        history_size=3, num_preds=1,
        loss=loss_obj,
    )


def run_family(family, pred_loss):
    torch.manual_seed(0)
    # T = history_size + num_preds (the real data window): tgt=emb[:,n_preds:]
    # aligns with the ctx_len predictions, exactly as in lejepa_forward.
    B, T, D = 2, 4, 192
    model = build(family, D=D, img_hw=32, infer_backprop=False)
    P = D * model.head.param_mult
    batch = {"pixels": torch.rand(B, T, 3, 32, 32), "action": torch.randn(B, T, 4)}

    info = model.encode({k: v.clone() for k, v in batch.items()})
    assert info["emb"].shape == (B, T, P), info["emb"].shape
    assert info["act_emb"].shape == (B, T, P), info["act_emb"].shape

    pe = model.predict(info["emb"][:, :3], info["act_emb"][:, :3])
    assert pe.shape == (B, 3, P), pe.shape
    dec = model.decode(info["emb"])
    assert dec.shape == (B, T, 3, 32, 32), dec.shape
    assert (dec >= 0).all() and (dec <= 1).all(), "decoder not in [0,1]"

    m = make_module(model)
    out = vijepa_forward(m, {k: v.clone() for k, v in batch.items()}, "train", make_cfg(pred_loss))
    assert torch.isfinite(out["loss"]), "non-finite loss"
    print(f"[{family}/{pred_loss}] pred={out['pred_loss'].item():.4f} "
          f"recon={out['recon_loss'].item():.4f} kl={out['kl_exact'].item():.4f} "
          f"fq={out['fisher_quad'].item():.4f} noop={out['pred_noop'].item():.4f} "
          f"loss={out['loss'].item():.4f}")
    out["loss"].backward()

    dec_g = sum(p.grad.abs().sum().item() for p in model.decoder.parameters() if p.grad is not None)
    pred_g = sum(p.grad.abs().sum().item() for p in model.predictor.parameters() if p.grad is not None)
    prior_set = model.prior_param.grad is not None
    assert dec_g > 0, "decoder got no gradient (recon broken)"
    assert pred_g > 0, "predictor got no gradient (pred term broken)"
    assert prior_set, "prior_param got no gradient"

    # predictive-term-ONLY: predictor should get grad, decoder should be ~0
    model.zero_grad()
    info2 = model.encode({k: v.clone() for k, v in batch.items()})
    tgt = info2["emb"][:, 1:4].detach()
    pred = model.predict(info2["emb"][:, :3], info2["act_emb"][:, :3])
    pterm = model.head.pred_term(tgt, pred, pred_loss)
    pterm.backward()
    dec_g2 = sum(p.grad.abs().sum().item() for p in model.decoder.parameters() if p.grad is not None)
    pred_g2 = sum(p.grad.abs().sum().item() for p in model.predictor.parameters() if p.grad is not None)
    print(f"    [grads] full: dec={dec_g:.2f} pred={pred_g:.2f} prior={'set' if prior_set else 'None'}"
          f"  | pred-only: dec={dec_g2:.2e} (want ~0) pred={pred_g2:.2f} (want >0)")
    assert pred_g2 > 0, "predictor got no gradient from predictive term"
    assert dec_g2 < 1e-6, f"decoder got gradient from predictive term: {dec_g2}"


def test_vit_decoder_double_backward():
    torch.manual_seed(0)
    dec = ViTDecoder(latent_dim=24, dim=24, depth=1, heads=4, mlp_dim=48, img_hw=16, patch_size=8)
    code = torch.randn(2, 24, requires_grad=True)
    img = dec(code)
    g = torch.autograd.grad(img.pow(2).mean(), code, create_graph=True)[0]
    w = dec.to_pixels.weight
    gg = torch.autograd.grad(g.sum(), w, retain_graph=False)[0]
    assert torch.isfinite(gg).all(), "non-finite ViTDecoder double-backward grad"
    print(f"[vit_decoder] double-backward grad norm={gg.norm().item():.4e} OK")


def _online_cfg(infer_backprop):
    return make_cfg(
        "quadratic_fisher",
        target_scheme="online_filtering",
        infer_objective="free_energy",
        beta=1.0,
        infer_backprop=infer_backprop,
        k_bptt=2,
    )


def _decoder_grad_norm(model):
    total = 0.0
    for param in model.decoder.parameters():
        if param.grad is not None:
            total += float(param.grad.detach().pow(2).sum())
    return total ** 0.5


def test_decoder_grad_through_online_inference():
    torch.manual_seed(0)
    batch = {"pixels": torch.rand(2, 3, 3, 16, 16), "action": torch.randn(2, 3, 4)}
    on = build("gaussian", D=32, img_hw=16, grid=4, infer_backprop=True, k_inner=2)
    on.decoder = ViTDecoder(latent_dim=32, dim=32, depth=1, heads=4, mlp_dim=64, img_hw=16, patch_size=8)
    out_on = filter_forward(make_module(on), {k: v.clone() for k, v in batch.items()}, "train", _online_cfg(True))
    out_on["loss"].backward()
    grad_on = _decoder_grad_norm(on)

    off = build("gaussian", D=32, img_hw=16, grid=4, infer_backprop=False, k_inner=2)
    off.decoder = ViTDecoder(latent_dim=32, dim=32, depth=1, heads=4, mlp_dim=64, img_hw=16, patch_size=8)
    out_off = filter_forward(make_module(off), {k: v.clone() for k, v in batch.items()}, "train", _online_cfg(False))
    out_off["loss"].backward()
    grad_off = _decoder_grad_norm(off)

    print(f"[online_bptt] decoder grad norm: infer_backprop=True {grad_on:.4e} | False {grad_off:.4e}")
    assert torch.isfinite(torch.tensor(grad_on)) and grad_on > 0, "BPTT decoder grad missing"
    assert abs(grad_on - grad_off) > 1e-8, "BPTT and detached decoder grads did not differ"


def test_recon_target_assertion():
    torch.manual_seed(0)
    model = build("poisson", D=64, img_hw=16, grid=4, infer_backprop=False, k_inner=1)
    batch = {"pixels": torch.rand(2, 4, 3, 16, 16), "action": torch.randn(2, 4, 4)}
    out = vijepa_forward(make_module(model), {k: v.clone() for k, v in batch.items()}, "train", make_cfg("exact_kl"))
    assert torch.isfinite(out["loss"]), "raw [0,1] recon target check failed unexpectedly"
    bad = {"pixels": batch["pixels"] * 4.0 - 2.0, "action": batch["action"]}
    try:
        vijepa_forward(make_module(model), bad, "train", make_cfg("exact_kl"))
    except AssertionError as exc:
        assert "[0,1]" in str(exc)
        print("[recon_target] raw target passes and normalized-like target trips assert OK")
        return
    raise AssertionError("normalized-like recon target did not trip assertion")


def main():
    for family in ["poisson", "gaussian"]:
        for pred_loss in (["exact_kl", "quadratic_fisher"]):
            run_family(family, pred_loss)
    test_vit_decoder_double_backward()
    test_decoder_grad_through_online_inference()
    test_recon_target_assertion()
    print("\nALL FONDJEPA SMOKE TESTS PASSED")


if __name__ == "__main__":
    main()
