"""Explainability for ECGNet (implemented directly in PyTorch, no external services).

Methods
- integrated_gradients: path integral of gradients from a baseline (flat line, mean RR)
  to the input (Sundararajan et al., 2017). Satisfies completeness: attributions sum to
  f(x) - f(baseline).
- saliency: absolute input gradient of the class score.
- grad_cam: class activation map of the last convolutional layer (Selvaraju et al., 2017),
  adapted to 1D and upsampled to the input length.
- occlusion: drop in class probability when a sliding window is replaced by a flat line.

These are analytical visualisations of what influences the model's score. They are not
evidence of clinical reasoning.
"""
from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn.functional as F

METHODS = {
    "integrated_gradients": "Integrated Gradients (flat-line baseline, 64 steps)",
    "saliency": "Gradient saliency |d score / d input|",
    "grad_cam": "1D Grad-CAM on the last convolutional layer",
    "occlusion": "Occlusion sensitivity (40 ms flat window, 10 ms stride)",
}


def _norm(a: np.ndarray) -> np.ndarray:
    m = np.max(np.abs(a))
    return a / m if m > 0 else a


def integrated_gradients(model, x: torch.Tensor, rr: torch.Tensor, target: int, steps: int = 64):
    model.eval()
    base_x = torch.zeros_like(x)
    base_rr = torch.zeros_like(rr)          # scaled RR = training mean
    alphas = torch.linspace(0, 1, steps + 1)[1:].view(-1, 1, 1)
    xs = (base_x + alphas * (x - base_x)).requires_grad_(True)
    rrs = (base_rr + alphas.view(-1, 1) * (rr - base_rr)).requires_grad_(True)
    out = model(xs, rrs if model.use_rr else None)
    score = out[:, target].sum()
    gx, grr = torch.autograd.grad(score, [xs, rrs], allow_unused=True)
    attr_x = ((x - base_x) * gx.mean(0, keepdim=True)).squeeze().detach().numpy()
    attr_rr = None
    if model.use_rr and grr is not None:
        attr_rr = ((rr - base_rr) * grr.mean(0, keepdim=True)).squeeze(0).detach().numpy()
    with torch.no_grad():
        delta = float(model(x, rr if model.use_rr else None)[0, target]
                      - model(base_x, base_rr if model.use_rr else None)[0, target])
    total = float(attr_x.sum() + (attr_rr.sum() if attr_rr is not None else 0.0))
    return attr_x, attr_rr, {"completeness_gap": round(abs(total - delta), 5),
                             "score_delta": round(delta, 5)}


def saliency(model, x, rr, target):
    model.eval()
    xg = x.clone().requires_grad_(True)
    out = model(xg, rr if model.use_rr else None)
    out[0, target].backward()
    return xg.grad.abs().squeeze().numpy()


def grad_cam(model, x, rr, target):
    model.eval()
    fmap = model.feature_maps(x)
    fmap.retain_grad()
    out = model.classify(fmap, rr if model.use_rr else None)
    out[0, target].backward()
    w = fmap.grad.mean(dim=2, keepdim=True)
    cam = F.relu((w * fmap).sum(1, keepdim=True))
    cam = F.interpolate(cam, size=x.shape[-1], mode="linear", align_corners=False)
    return cam.squeeze().detach().numpy()


@torch.no_grad()
def occlusion(model, x, rr, target, fs: int = 360, window_ms: int = 40, stride_ms: int = 10):
    model.eval()
    n = x.shape[-1]
    w, st = max(int(window_ms * fs / 1000), 1), max(int(stride_ms * fs / 1000), 1)
    starts = list(range(0, n - w + 1, st))
    batch = x.repeat(len(starts), 1, 1)
    for i, s in enumerate(starts):
        batch[i, 0, s:s + w] = 0.0
    r = rr.repeat(len(starts), 1) if model.use_rr else None
    base = torch.softmax(model(x, rr if model.use_rr else None), 1)[0, target]
    probs = torch.softmax(model(batch, r), 1)[:, target]
    drop = (base - probs).numpy()
    heat = np.zeros(n)
    cnt = np.zeros(n)
    for d, s in zip(drop, starts):
        heat[s:s + w] += d
        cnt[s:s + w] += 1
    return heat / np.maximum(cnt, 1)


def explain(model, x: np.ndarray, rr: np.ndarray, target: int, method: str) -> dict:
    xt = torch.from_numpy(np.asarray(x, dtype=np.float32)).view(1, 1, -1)
    rt = torch.from_numpy(np.asarray(rr, dtype=np.float32)).view(1, -1)
    extra: dict = {}
    rr_attr = None
    if method == "integrated_gradients":
        a, rr_attr, extra = integrated_gradients(model, xt, rt, target)
    elif method == "saliency":
        a = saliency(model, xt, rt, target)
    elif method == "grad_cam":
        a = grad_cam(model, xt, rt, target)
    elif method == "occlusion":
        a = occlusion(model, xt, rt, target)
    else:
        raise ValueError(f"Unknown method {method}")
    model.zero_grad(set_to_none=True)
    return {"method": method, "description": METHODS[method],
            "attribution": [round(float(v), 5) for v in _norm(np.asarray(a, dtype=np.float64))],
            "rr_attribution": None if rr_attr is None else [round(float(v), 5) for v in rr_attr],
            **extra}


def randomization_sanity(model, x: np.ndarray, rr: np.ndarray, target: int, seed: int = 0) -> dict:
    """Model-parameter randomisation test (Adebayo et al., 2018).

    If explanations barely change when the trained weights are replaced by random ones,
    they reflect the input rather than what the model learned.
    """
    torch.manual_seed(seed)
    rand = copy.deepcopy(model)
    for m in rand.modules():
        if hasattr(m, "reset_parameters"):
            m.reset_parameters()
    rand.eval()
    out = {}
    for method in ("integrated_gradients", "grad_cam"):
        a = np.asarray(explain(model, x, rr, target, method)["attribution"])
        b = np.asarray(explain(rand, x, rr, target, method)["attribution"])
        if a.std() > 0 and b.std() > 0:
            out[method] = round(float(np.corrcoef(np.abs(a), np.abs(b))[0, 1]), 4)
        else:
            out[method] = None
    return {"correlation_trained_vs_random": out,
            "interpretation": "Low correlation means the explanation depends on the learned "
                              "weights (the desired outcome)."}
