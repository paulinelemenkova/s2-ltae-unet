#!/usr/bin/env python3

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from unet_models import ConvLSTMNet, PlainUNet, TAEUNet

CHIP, BATCH, STEPS = 32, 32, 25
LR, LR_MIN, WD, DELTA, MAX_EPOCHS = 2e-3, 1e-5, 1e-4, 0.1, 24
WIDTHS = (32, 64, 128, 256)
I_NORTH, I_EAST = 2, 3
PRED_S = {"tae": 40.0, "plain": 20.0, "convlstm": 130.0}


def augment(
    x: torch.Tensor,
    s: torch.Tensor,
    y: torch.Tensor,
    m: torch.Tensor,
    g: torch.Generator,
) -> tuple[torch.Tensor, ...]:
    if torch.rand(1, generator=g) < 0.5:
        x, s, y, m = (t.flip(-1) for t in (x, s, y, m))
        s = s.clone()
        s[:, I_EAST] = -s[:, I_EAST]
    if torch.rand(1, generator=g) < 0.5:
        x, s, y, m = (t.flip(-2) for t in (x, s, y, m))
        s = s.clone()
        s[:, I_NORTH] = -s[:, I_NORTH]
    k = int(torch.randint(0, 4, (1,), generator=g))
    for _ in range(k):
        x, s, y, m = (torch.rot90(t, 1, (-2, -1)) for t in (x, s, y, m))
        s = s.clone()
        e, n = s[:, I_EAST].clone(), s[:, I_NORTH].clone()
        s[:, I_EAST], s[:, I_NORTH] = -n, e
    return x, s, y, m


def build(name: str, ns: int) -> torch.nn.Module:
    if name == "tae":
        return TAEUNet(4, ns, widths=WIDTHS)
    if name == "plain":
        return PlainUNet(4, ns, widths=WIDTHS)
    return ConvLSTMNet(4, ns, hid=16)


def predict(
    net: torch.nn.Module, x: torch.Tensor, s: torch.Tensor, want_att: bool
) -> tuple[np.ndarray, np.ndarray | None]:
    h, w = x.shape[-2:]
    st = CHIP // 2
    hann = torch.hann_window(CHIP + 2, periodic=False)[1:-1]
    wt = (hann[:, None] * hann[None, :]).clamp_min(1e-3)
    acc = torch.zeros(h, w)
    den = torch.zeros(h, w)
    att = torch.zeros(x.shape[1], h, w) if want_att else None
    rs = list(range(0, h - CHIP, st)) + [h - CHIP]
    cs = list(range(0, w - CHIP, st)) + [w - CHIP]
    net.eval()
    with torch.no_grad():
        for r in rs:
            xb = torch.stack([x[0, :, :, r : r + CHIP, c : c + CHIP] for c in cs])
            sb = torch.stack([s[0, :, r : r + CHIP, c : c + CHIP] for c in cs])
            p = net(xb, sb)[:, 0]
            for j, c in enumerate(cs):
                acc[r : r + CHIP, c : c + CHIP] += p[j] * wt
                den[r : r + CHIP, c : c + CHIP] += wt
                if att is not None:
                    a = net.last_attention[j].mean(0)
                    att[:, r : r + CHIP, c : c + CHIP] += a * wt
    pred = (acc / den).numpy()
    return pred, (att / den).numpy() if att is not None else None


def main() -> None:
    npz, name, k, out = sys.argv[1], sys.argv[2], int(sys.argv[3]), Path(sys.argv[4])
    budget = float(sys.argv[5])
    ckpt = sys.argv[6] if len(sys.argv) > 6 else None
    t0 = time.time()
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(k)
    torch.set_num_threads(2)
    g = torch.Generator().manual_seed(100 + k)
    d = np.load(npz)
    x = torch.from_numpy(d["x"].astype("float32") / 10000)[None]
    s = torch.from_numpy(d["s"].astype("float32"))[None]
    y = torch.from_numpy(d["y"])[None, None]
    valid, fold, blk = d["valid"], d["fold"], d["block"]

    train_blocks = np.unique(blk[valid & (fold >= 0) & (fold != k)])
    rng = np.random.default_rng(k)
    val_blocks = rng.choice(
        train_blocks, max(1, round(0.15 * len(train_blocks))), False
    )
    is_val = np.isin(blk, val_blocks)
    m_tr = torch.from_numpy(valid & (fold >= 0) & (fold != k) & ~is_val).float()[
        None, None
    ]
    m_va = torch.from_numpy(valid & is_val).float()[None, None]

    def anchors(mask: torch.Tensor, need: int) -> np.ndarray:
        cnt = F.avg_pool2d(mask, CHIP, stride=4)[0, 0].numpy() * CHIP * CHIP
        rr, cc = np.nonzero(cnt >= need)
        return np.c_[rr * 4, cc * 4]

    a_tr, a_va = anchors(m_tr, 100), anchors(m_va, 100)
    va_sel = a_va[np.linspace(0, len(a_va) - 1, min(64, len(a_va))).astype(int)]

    def chips(idx: np.ndarray, mask: torch.Tensor) -> tuple[torch.Tensor, ...]:
        xs = torch.stack([x[0, :, :, r : r + CHIP, c : c + CHIP] for r, c in idx])
        ss = torch.stack([s[0, :, r : r + CHIP, c : c + CHIP] for r, c in idx])
        ys = torch.stack([y[0, :, r : r + CHIP, c : c + CHIP] for r, c in idx])
        ms = torch.stack([mask[0, :, r : r + CHIP, c : c + CHIP] for r, c in idx])
        return xs, ss, ys, ms

    def loss_fn(p: torch.Tensor, t: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        e = F.huber_loss(p, t, reduction="none", delta=DELTA)
        return (e * m).sum() / m.sum().clamp_min(1)

    net = build(name, s.shape[1])
    opt = torch.optim.AdamW(net.parameters(), LR, weight_decay=WD)
    xv, sv, yv, mv = chips(va_sel, m_va)
    best, best_state, wait, log = np.inf, None, 0, []
    epoch = 0
    if ckpt:
        c = torch.load(ckpt, weights_only=False)
        net.load_state_dict(c["net"])
        opt.load_state_dict(c["opt"])
        best, best_state, wait, log = c["best"], c["best_state"], c["wait"], c["log"]
        epoch = c["epoch"]
        rng.bit_generator.state = c["rng"]
        g.set_state(c["g"])
    done = epoch >= MAX_EPOCHS
    while not done:
        te = time.time()
        net.train()
        tl = 0.0
        for i in range(STEPS):
            step = epoch * STEPS + i
            lr = LR_MIN + 0.5 * (LR - LR_MIN) * (
                1 + np.cos(np.pi * step / (MAX_EPOCHS * STEPS))
            )
            for pg in opt.param_groups:
                pg["lr"] = lr
            sel = a_tr[rng.integers(0, len(a_tr), BATCH)]
            xb, sb, yb, mb = chips(sel, m_tr)
            xb, sb, yb, mb = augment(xb, sb, yb, mb, g)
            opt.zero_grad()
            loss = loss_fn(net(xb, sb), yb, mb)
            loss.backward()
            opt.step()
            tl += loss.item() / STEPS
        net.eval()
        with torch.no_grad():
            vl = loss_fn(net(xv, sv), yv, mv).item()
        epoch += 1
        log.append([epoch, round(tl, 5), round(vl, 5)])
        if vl < best:
            best, wait = vl, 0
            best_state = {q: v.clone() for q, v in net.state_dict().items()}
        else:
            wait += 1
        dt = time.time() - te
        done = epoch >= MAX_EPOCHS
        late = time.time() - t0 + dt > budget
        if (not done and late) or (done and time.time() - t0 > budget - PRED_S[name]):
            torch.save(
                {
                    "net": net.state_dict(),
                    "opt": opt.state_dict(),
                    "best": best,
                    "best_state": best_state,
                    "wait": wait,
                    "log": log,
                    "epoch": epoch,
                    "rng": rng.bit_generator.state,
                    "g": g.get_state(),
                },
                out / f"ckpt_{name}_f{k}.pt",
            )
            print(
                json.dumps(
                    {
                        "model": name,
                        "fold": k,
                        "status": "continue",
                        "epoch": epoch,
                        "best_val": best,
                        "last": log[-1],
                    }
                )
            )
            return
    assert best_state is not None
    net.load_state_dict(best_state)
    pred, att = predict(net, x, s, want_att=name == "tae")
    held = valid & (fold == k)
    pred = np.where(held, pred, np.nan).astype("float32")
    np.save(out / f"pred_{name}_f{k}.npy", pred)
    if att is not None:
        np.save(
            out / f"att_tae_f{k}.npy", np.where(held, att, np.nan).astype("float16")
        )
    obs = d["y"][held]
    e = pred[held] - obs
    sc = {
        "r2": float(1 - (e**2).sum() / ((obs - obs.mean()) ** 2).sum()),
        "rmse": float(np.sqrt((e**2).mean())),
        "mae": float(np.abs(e).mean()),
        "bias": float(e.mean()),
        "n": int(held.sum()),
    }
    rec = {
        "model": name,
        "fold": k,
        "epochs": epoch,
        "best_val": best,
        "seconds": round(time.time() - t0),
        "scores": sc,
        "log": log,
        "params": sum(p.numel() for p in net.parameters()),
    }
    (out / f"log_{name}_f{k}.json").write_text(json.dumps(rec))
    print(json.dumps({q: v for q, v in rec.items() if q != "log"}))


if __name__ == "__main__":
    main()
