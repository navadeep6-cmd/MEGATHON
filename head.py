"""Supervised multi-label head with asymmetric loss (ASL).

p_c = sigmoid(w_c^T z_x + b_c), trained on frozen embeddings with

  L = -sum_c [ y_c (1-p_c)^{g+} log p_c + (1-y_c) p_{c,m}^{g-} log(1-p_{c,m}) ],
  p_{c,m} = max(p_c - m, 0)

(Ridnik et al., "Asymmetric Loss For Multi-Label Classification", ICCV 2021;
implemented here from the formula). The margin m tolerates missing labels.
Editor *rejections* are reliable negatives, so for those entries the margin
and focusing are switched off (plain BCE negative term).

W is initialised from the category text prototypes (text-initialised
classifier), which shares parameters across categories and speeds training.
Inference is numpy-only so serving does not need torch for the head.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

log = logging.getLogger(__name__)


class LinearHead:
    """Numpy inference for the trained head. cat_ids[j] = category id of output j."""

    def __init__(self, W: np.ndarray, b: np.ndarray, cat_ids: np.ndarray):
        self.W = np.ascontiguousarray(W, dtype=np.float32)
        self.b = np.asarray(b, dtype=np.float32)
        self.cat_ids = np.asarray(cat_ids, dtype=np.int64)

    def probs(self, Z: np.ndarray) -> np.ndarray:
        logits = Z.astype(np.float32) @ self.W.T + self.b
        return 1.0 / (1.0 + np.exp(-np.clip(logits, -30.0, 30.0)))

    def save(self, path: str) -> None:
        np.savez(path, W=self.W, b=self.b, cat_ids=self.cat_ids)

    @classmethod
    def load(cls, path: str) -> "LinearHead":
        d = np.load(path)
        return cls(d["W"], d["b"], d["cat_ids"])


def sample_map_dense(scores: np.ndarray, Y: np.ndarray) -> float:
    """Mean average precision over rows of a dense score matrix (rows without positives skipped)."""
    keep = Y.sum(axis=1) > 0
    if not keep.any():
        return 0.0
    s, y = scores[keep], Y[keep].astype(np.float64)
    order = np.argsort(-s, axis=1)
    hits = np.take_along_axis(y, order, axis=1)
    cum = np.cumsum(hits, axis=1)
    ranks = np.arange(1, s.shape[1] + 1, dtype=np.float64)[None, :]
    ap = (cum / ranks * hits).sum(axis=1) / hits.sum(axis=1)
    return float(ap.mean())


def _dense(rows: Sequence[Sequence[int]], n_cols: int) -> np.ndarray:
    out = np.zeros((len(rows), n_cols), dtype=np.float32)
    for i, cols in enumerate(rows):
        if len(cols):
            out[i, list(cols)] = 1.0
    return out


def asl_loss(logits, targets, known_neg=None, gamma_pos: float = 0.0, gamma_neg: float = 4.0,
             margin: float = 0.05, eps: float = 1e-8):
    """Asymmetric loss (torch). targets/known_neg: float tensors of the logits' shape."""
    import torch

    p = torch.sigmoid(logits)
    p_m = (p - margin).clamp(min=0.0)
    loss_pos = targets * (1.0 - p).pow(gamma_pos) * torch.log(p.clamp(min=eps))
    neg = 1.0 - targets
    if known_neg is not None:
        unknown = neg * (1.0 - known_neg)
        loss_neg = unknown * p_m.pow(gamma_neg) * torch.log((1.0 - p_m).clamp(min=eps))
        loss_neg = loss_neg + known_neg * neg * torch.log((1.0 - p).clamp(min=eps))
    else:
        loss_neg = neg * p_m.pow(gamma_neg) * torch.log((1.0 - p_m).clamp(min=eps))
    return -(loss_pos + loss_neg).sum(dim=1).mean()


def train_head(
    X_train: np.ndarray,
    Y_train: Sequence[Sequence[int]],
    X_val: np.ndarray,
    Y_val: Sequence[Sequence[int]],
    T_head: np.ndarray,
    cfg: Dict,
    neg_train: Optional[Sequence[Sequence[int]]] = None,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
    """Train the head. Y_* hold head-column indices per sample. Returns (W, b, history)."""
    import torch

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    n, d = X_train.shape
    H = T_head.shape[0]
    alpha = float(cfg.get("text_init_scale", 20.0))
    bias_offset = float(cfg.get("init_bias_offset", 3.0))

    # text-initialised classifier: logit = alpha * (cos(z, t_c) - mean_cos_c) - offset
    sample = X_train[rng.choice(n, size=min(n, 5000), replace=False)]
    mean_cos = (sample @ T_head.T).mean(axis=0)
    W0 = (alpha * T_head).astype(np.float32)
    b0 = (-alpha * mean_cos - bias_offset).astype(np.float32)

    lin = torch.nn.Linear(d, H)
    with torch.no_grad():
        lin.weight.copy_(torch.from_numpy(W0))
        lin.bias.copy_(torch.from_numpy(b0))
    opt = torch.optim.AdamW(lin.parameters(), lr=float(cfg["lr"]), weight_decay=float(cfg["weight_decay"]))

    Xt = torch.from_numpy(np.ascontiguousarray(X_train, dtype=np.float32))
    Yv = _dense(Y_val, H)
    batch = int(cfg["batch_size"])
    gp, gn, m = float(cfg["gamma_pos"]), float(cfg["gamma_neg"]), float(cfg["margin"])
    epochs, patience = int(cfg["epochs"]), int(cfg["patience"])

    def val_map() -> float:
        with torch.no_grad():
            logits = lin(torch.from_numpy(np.ascontiguousarray(X_val, dtype=np.float32))).numpy()
        return sample_map_dense(logits, Yv)

    best = val_map()
    best_state = {k: v.clone() for k, v in lin.state_dict().items()}
    history = [{"epoch": 0, "loss": None, "val_mAP": best}]
    log.info("head init (text prototypes): val mAP %.4f", best)
    bad_epochs = 0
    for epoch in range(1, epochs + 1):
        perm = rng.permutation(n)
        total, steps = 0.0, 0
        for start in range(0, n, batch):
            idx = perm[start:start + batch]
            yb = torch.from_numpy(_dense([Y_train[i] for i in idx], H))
            kn = torch.from_numpy(_dense([neg_train[i] for i in idx], H)) if neg_train is not None else None
            loss = asl_loss(lin(Xt[torch.from_numpy(idx)]), yb, kn, gp, gn, m)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.item())
            steps += 1
        score = val_map()
        history.append({"epoch": epoch, "loss": total / max(steps, 1), "val_mAP": score})
        log.info("head epoch %d loss %.4f val mAP %.4f", epoch, total / max(steps, 1), score)
        if score > best + 1e-4:
            best, bad_epochs = score, 0
            best_state = {k: v.clone() for k, v in lin.state_dict().items()}
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break
    lin.load_state_dict(best_state)
    W = lin.weight.detach().numpy().astype(np.float32)
    b = lin.bias.detach().numpy().astype(np.float32)
    return W, b, history
