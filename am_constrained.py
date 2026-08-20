import math
from typing import Dict, List, Optional, Tuple

import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F

from dataset_generator import CSPPInstance


# Hyperparameters (values cited in the revised paper):
D_HID = 64
N_HEADS = 4
N_LAYERS = 2
N_TRAIN_INSTANCES = 50
N_EPOCHS = 30
BATCH = 8
LR = 5e-4

class _AMDecoder(nn.Module):

    def __init__(self, d_hid: int, n_heads: int):
        super().__init__()
        # Tiny single-head dot-product; multi-head is unnecessary for
        # a 30-node pointer and the original MHA was buggy in this
        # code path. We keep n_heads as an arg to satisfy the API but
        # always operate on a single combined vector.
        self.q_proj = nn.Linear(d_hid, d_hid)
        self.k_proj = nn.Linear(d_hid, d_hid)
        self.d_hid = d_hid

    def forward(self, query: torch.Tensor, keys: torch.Tensor,
                mask: torch.Tensor) -> torch.Tensor:
        """Returns logits of shape (B, N)."""
        q = self.q_proj(query).unsqueeze(1)        # (B, 1, d)
        k = self.k_proj(keys)                      # (B, N, d)
        # (B, 1, d) x (B, d, N) -> (B, 1, N)
        logits = torch.bmm(q, k.transpose(1, 2)) / (self.d_hid ** 0.5)
        logits = logits.squeeze(1)                 # (B, N)
        logits = logits.masked_fill(mask, float("-inf"))
        return logits


class _AttentionModel(nn.Module):

    def __init__(self, n_nodes: int, d_hid: int = D_HID,
                 n_heads: int = N_HEADS, n_layers: int = N_LAYERS):
        super().__init__()
        self.embed = nn.Embedding(n_nodes, d_hid)
        enc_layer = nn.TransformerEncoderLayer(
            d_hid, n_heads, dim_feedforward=4 * d_hid, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, n_layers)
        self.dec = _AMDecoder(d_hid, n_heads)
        self.n_nodes = n_nodes
        self.d_hid = d_hid

    def encode(self, batch_indices: torch.Tensor) -> torch.Tensor:
        # batch_indices: (B,) -> one node id per batch element.
        # We embed all nodes once and index into them.
        all = self.embed.weight.unsqueeze(0).expand(
            batch_indices.size(0), -1, -1)
        return self.encoder(all)  # (B, N, d)

    def forward(self, h: torch.Tensor, query_idx: torch.Tensor,
                mask: torch.Tensor) -> torch.Tensor:
        """h: (B, N, d), query_idx: (B,), mask: (B, N) True where forbidden.
        Returns logits (B, N)."""
        q = h[torch.arange(h.size(0)), query_idx]   # (B, d)
        return self.dec(q, h, mask)                 # (B, N)

class ConstrainedAM:
    """Wrapper that trains (on the supplied instance) and then solves."""

    def __init__(self, instance: CSPPInstance, device: str = "cpu",
                 n_epochs: int = N_EPOCHS, lr: float = LR):
        self.instance = instance
        self.device = torch.device(device)
        self.n_nodes = instance.G.number_of_nodes()
        self.model = _AttentionModel(self.n_nodes).to(self.device)
        self.opt = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.s = instance.origin
        self.t = instance.destination
        self.T_max = instance.T_max
        # Tensor of edge times, shape (N, N) with -1 for non-edges.
        self.edge_time = torch.full((self.n_nodes, self.n_nodes), -1.0)
        self.edge_cost = torch.full((self.n_nodes, self.n_nodes), -1.0)
        for u, v, d in instance.G.edges(data=True):
            self.edge_time[u, v] = float(d["time"])
            self.edge_cost[u, v] = float(d["cost"])
        self.edge_time = self.edge_time.to(self.device)
        self.edge_cost = self.edge_cost.to(self.device)

    def _greedy_decode(self) -> Tuple[List[int], float, float]:
        with torch.no_grad():
            h = self.model.encode(torch.tensor([self.s],
                                               device=self.device))
            path = [self.s]
            visited = torch.zeros(self.n_nodes, dtype=torch.bool,
                                  device=self.device)
            visited[self.s] = True
            cum_t = 0.0
            cum_c = 0.0
            for _ in range(self.n_nodes):
                cur = path[-1]
                # Build feasibility mask: True where forbidden.
                mask = visited.clone()
                for v in range(self.n_nodes):
                    if mask[v]:
                        continue
                    t_uv = self.edge_time[cur, v].item()
                    if t_uv < 0:
                        mask[v] = True
                    elif cum_t + t_uv > self.T_max:
                        mask[v] = True
                if mask.all():
                    break
                logits = self.model(
                    h, torch.tensor([cur], device=self.device),
                    mask.unsqueeze(0))
                nxt = int(torch.argmax(logits, dim=-1).item())
                cum_t += self.edge_time[cur, nxt].item()
                cum_c += self.edge_cost[cur, nxt].item()
                path.append(nxt)
                visited[nxt] = True
                if nxt == self.t:
                    break
            return path, cum_c, cum_t

    def train_self_supervised(self, n_epochs: int = N_EPOCHS):
     
        rng = torch.Generator(device=self.device).manual_seed(0)
        for epoch in range(n_epochs):
            paths, log_probs, costs = [], [], []
            for _ in range(BATCH):
                h = self.model.encode(torch.tensor([self.s],
                                                   device=self.device))
                path = [self.s]
                visited = torch.zeros(self.n_nodes, dtype=torch.bool,
                                      device=self.device)
                visited[self.s] = True
                cum_t = 0.0
                cum_c = 0.0
                lp = 0.0
                for _ in range(self.n_nodes):
                    cur = path[-1]
                    mask = visited.clone()
                    for v in range(self.n_nodes):
                        if mask[v]:
                            continue
                        t_uv = self.edge_time[cur, v].item()
                        if t_uv < 0 or cum_t + t_uv > self.T_max:
                            mask[v] = True
                    if mask.all():
                        break
                    logits = self.model(
                        h, torch.tensor([cur], device=self.device),
                        mask.unsqueeze(0))
                    probs = F.softmax(logits, dim=-1)
                    a = torch.multinomial(probs, 1, generator=rng).item()
                    lp += torch.log(probs[0, a] + 1e-12).item()
                    cum_t += self.edge_time[cur, a].item()
                    cum_c += self.edge_cost[cur, a].item()
                    path.append(a)
                    visited[a] = True
                    if a == self.t:
                        break
                feasible = (path[-1] == self.t) and (cum_t <= self.T_max)
                paths.append((path, cum_c if feasible else float("inf"),
                              feasible))
                log_probs.append(lp)

            # Pick best feasible path as baseline.
            feasibles = [p for p in paths if p[2]]
            if not feasibles:
                continue
            best = min(feasibles, key=lambda x: x[1])[1]
            # REINFORCE: minimise expected cost.
            loss = 0.0
            for (p, c, f), lp in zip(paths, log_probs):
                if not f:
                    continue
                advantage = c - best
                loss = loss + advantage * lp
            loss = torch.tensor(loss / max(len(feasibles), 1),
                                requires_grad=True, device=self.device)
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()

    def solve(self) -> Dict:
        path, cost, time = self._greedy_decode()
        feasible = (path[-1] == self.t) and (time <= self.T_max)
        return {"path": path if feasible else None,
                "cost": float(cost) if feasible else float("inf"),
                "time": float(time),
                "feasible": bool(feasible)}


def solve(instance: CSPPInstance, n_epochs: int = N_EPOCHS,
          device: str = "cpu") -> Dict:
    """Train+infer AM on a single instance."""
    am = ConstrainedAM(instance, device=device, n_epochs=n_epochs)
    am.train_self_supervised(n_epochs)
    return am.solve()
