from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


def block(cin: int, cout: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
    )


class LTAE(nn.Module):
    def __init__(self, cin: int, d: int = 64, heads: int = 4, dk: int = 8, t: int = 12):
        super().__init__()
        self.heads, self.dk, self.d = heads, dk, d
        self.embed = nn.Sequential(nn.Linear(cin, d), nn.LayerNorm(d), nn.ReLU())
        pos = torch.arange(t, dtype=torch.float32)[:, None]
        div = torch.exp(
            torch.arange(0, d, 2, dtype=torch.float32) * (-math.log(1000.0) / d)
        )
        pe = torch.zeros(t, d)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe)
        self.key = nn.Linear(d, heads * dk)
        self.query = nn.Parameter(torch.randn(heads, dk) * 0.1)
        self.out = nn.Sequential(nn.Linear(d, d), nn.LayerNorm(d), nn.ReLU())

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        b, t, c, h, w = x.shape
        p = x.permute(0, 3, 4, 1, 2).reshape(b * h * w, t, c)
        e = self.embed(p) + self.pe[:t]
        k = self.key(e).view(-1, t, self.heads, self.dk)
        a = torch.einsum("nthk,hk->nht", k, self.query) / math.sqrt(self.dk)
        a = a.softmax(-1)
        v = e.view(-1, t, self.heads, self.d // self.heads)
        z = torch.einsum("nht,nthv->nhv", a, v).reshape(-1, self.d)
        z = self.out(z).view(b, h, w, self.d).permute(0, 3, 1, 2)
        return z, a.view(b, h, w, self.heads, t).permute(0, 3, 4, 1, 2)


class UNetCore(nn.Module):
    def __init__(
        self, cin: int, ns: int, widths: tuple[int, ...] = (64, 128, 256, 512)
    ):
        super().__init__()
        w1, w2, w3, wb = widths
        self.e1, self.e2, self.e3 = block(cin, w1), block(w1, w2), block(w2, w3)
        self.bott = block(w3 + ns, wb)
        self.u3 = nn.ConvTranspose2d(wb, w3, 2, 2)
        self.d3 = block(2 * w3, w3)
        self.u2 = nn.ConvTranspose2d(w3, w2, 2, 2)
        self.d2 = block(2 * w2, w2)
        self.u1 = nn.ConvTranspose2d(w2, w1, 2, 2)
        self.d1 = block(2 * w1, w1)
        self.head = nn.Conv2d(w1, 1, 1)

    def forward(self, z: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        x1 = self.e1(z)
        x2 = self.e2(F.max_pool2d(x1, 2))
        x3 = self.e3(F.max_pool2d(x2, 2))
        xb = F.max_pool2d(x3, 2)
        sb = F.adaptive_avg_pool2d(s, xb.shape[-2:])
        xb = self.bott(torch.cat([xb, sb], 1))
        y = self.d3(torch.cat([self.u3(xb), x3], 1))
        y = self.d2(torch.cat([self.u2(y), x2], 1))
        y = self.d1(torch.cat([self.u1(y), x1], 1))
        return self.head(y)


class TAEUNet(nn.Module):
    def __init__(
        self,
        c: int,
        ns: int,
        t: int = 12,
        widths: tuple[int, ...] = (64, 128, 256, 512),
    ):
        super().__init__()
        self.tae = LTAE(c, d=widths[0], t=t)
        self.unet = UNetCore(widths[0] + ns, ns, widths)
        self.last_attention: torch.Tensor | None = None

    def forward(self, x: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        z, a = self.tae(x)
        self.last_attention = a.detach()
        return self.unet(torch.cat([z, s], 1), s)


class PlainUNet(nn.Module):
    def __init__(
        self,
        c: int,
        ns: int,
        t: int = 12,
        widths: tuple[int, ...] = (64, 128, 256, 512),
    ):
        super().__init__()
        self.unet = UNetCore(c * t + ns, ns, widths)

    def forward(self, x: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        return self.unet(torch.cat([x.flatten(1, 2), s], 1), s)


class ConvLSTMCell(nn.Module):
    def __init__(self, cin: int, hid: int):
        super().__init__()
        self.hid = hid
        self.conv = nn.Conv2d(cin + hid, 4 * hid, 3, padding=1)

    def forward(
        self, x: torch.Tensor, hc: tuple[torch.Tensor, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h, c = hc
        i, f, o, g = self.conv(torch.cat([x, h], 1)).chunk(4, 1)
        c = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(g)
        h = torch.sigmoid(o) * torch.tanh(c)
        return h, c


class ConvLSTMNet(nn.Module):
    def __init__(self, c: int, ns: int, hid: int = 32):
        super().__init__()
        self.c1, self.c2 = ConvLSTMCell(c, hid), ConvLSTMCell(hid, hid)
        self.head = nn.Sequential(
            nn.Conv2d(hid + ns, hid, 3, padding=1), nn.ReLU(), nn.Conv2d(hid, 1, 1)
        )

    def forward(self, x: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        b, t, _, h, w = x.shape
        z = x.new_zeros(b, self.c1.hid, h, w)
        h1, c1, h2, c2 = z, z, z, z
        for k in range(t):
            h1, c1 = self.c1(x[:, k], (h1, c1))
            h2, c2 = self.c2(h1, (h2, c2))
        return self.head(torch.cat([h2, s], 1))
