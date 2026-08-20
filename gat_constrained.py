from typing import Dict, List, Optional, Tuple
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F

from dataset_generator import CSPPInstance


D_HID = 64
N_HEADS = 8
N_LAYERS = 3
BATCH = 8
N_EPOCHS = 30
LR = 5e-4


class _GATLayer(nn.Module):
 
    def __init__(self, d_in: int, d_out: int, n_heads: int):
        super().__init__()
        assert d_out % n_heads == 0
        self.W = nn.Linear(d_in, d_out, bias=False)
        self.a_src = nn.Parameter(torch.zeros(n_heads, d_out // n_heads))
        self.a_dst = nn.Parameter(torch.zeros(n_heads, d_out // n_heads))
        nn.init.xavier_uniform_(self.a_src)
        nn.init.xavier_uniform_(self.a_dst)
        self.n_heads = n_heads
        self.d_head = d_out // n_heads

    def forward(self, x: torch.Tensor,
                edge_index: torch.Tensor) -> torch.Tensor:
        """x: (N, d_in), edge_index: (2, E)."""
        h = self.W(x)                       # (N, d_out)
        N = h.size(0)
        H, Dh = self.n_heads, self.d_head
        h = h.view(N, H, Dh)
        src, dst = edge_index
        alpha_src = (h[src] * self.a_src).sum(-1)   # (E, H)
        alpha_dst = (h[dst] * self.a_dst).sum(-1)
        alpha = F.leaky_relu(alpha_src + alpha_dst, negative_slope=0.2)
        alpha = torch.exp(alpha - alpha.max())
        # Softmax over dst of each src.
        out = torch.zeros_like(h)
        denom = torch.zeros(N, H, device=x.device).scatter_add_(
            0, dst.unsqueeze(-1).expand(-1, H), alpha)
        msg = alpha.unsqueeze(-1) * h[src]  # (E, H, Dh)
        out.index_add_(0, dst, msg)
        out = out / (denom.unsqueeze(-1) + 1e-12)
        return out.reshape(N, H * Dh)


class _GATPolicy(nn.Module):
 

    def __init__(self, n_nodes: int, d_in: int = 2, d_hid: int = D_HID,
                 n_heads: int = N_HEADS, n_layers: int = N_LAYERS):
        super().__init__()
        self.embed = nn.Linear(d_in, d_hid)
        self.layers = nn.ModuleList(
            [_GATLayer(d_hid, d_hid, n_heads) for _ in range(n_layers)]
        )
        self.head = nn.Linear(3 * d_hid, 1)
        self.n_nodes = n_nodes

    def encode(self, node_feats: torch.Tensor,
               edge_index: torch.Tensor) -> torch.Tensor:
        h = self.embed(node_feats)
        for layer in self.layers:
            h = F.elu(layer(h, edge_index))
        return h

    def forward(self, h: torch.Tensor, cur: int,
                candidates: List[int], mask: torch.Tensor) -> torch.Tensor:
 
        cand_idx = torch.tensor(candidates, device=h.device)
        feats = torch.cat([
            h[cur].unsqueeze(0).expand(len(candidates), -1),
            h[cand_idx],
            h[cur].unsqueeze(0).expand(len(candidates), -1) - h[cand_idx],
        ], dim=-1)
        logits = self.head(feats).squeeze(-1)
        logits = logits.masked_fill(mask, float("-inf"))
        return logits


class ConstrainedGAT:
    def __init__(self, instance: CSPPInstance, device: str = "cpu",
                 lr: float = LR):
        self.instance = instance
        self.device = torch.device(device)
        self.n_nodes = instance.G.number_of_nodes()
        self.s = instance.origin
        self.t = instance.destination
        self.T_max = instance.T_max
        # Node features: [in_degree_norm, out_degree_norm].
        indeg = [instance.G.in_degree(v) / self.n_nodes
                 for v in range(self.n_nodes)]
        outdeg = [instance.G.out_degree(v) / self.n_nodes
                  for v in range(self.n_nodes)]
        self.node_feats = torch.tensor(
            [[indeg[v], outdeg[v]] for v in range(self.n_nodes)],
            dtype=torch.float32, device=self.device)
        # Edge index.
        edges = list(instance.G.edges())
        self.edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous().to(self.device)
        # Edge times/costs.
        self.edge_time = torch.full((self.n_nodes, self.n_nodes), -1.0)
        for u, v, d in instance.G.edges(data=True):
            self.edge_time[u, v] = float(d["time"])
        self.edge_time = self.edge_time.to(self.device)
        self.model = _GATPolicy(self.n_nodes).to(self.device)
        self.opt = torch.optim.Adam(self.model.parameters(), lr=lr)

    def _step(self, h, cur, visited, cum_t):
        successors = [v for v in range(self.n_nodes)
                      if self.edge_time[cur, v] >= 0 and v not in visited]
        if not successors:
            return None
        cand_t = torch.tensor(
            [self.edge_time[cur, v].item() + cum_t for v in successors],
            device=self.device)
        mask = cand_t > self.T_max
        if mask.all():
            return None
        logits = self.model(h, cur, successors, mask)
        return successors, logits, mask

    def _greedy_decode(self, h) -> Tuple[List[int], float, float]:
        path = [self.s]
        visited = {self.s}
        cum_t = 0.0
        cum_c = 0.0
        for _ in range(self.n_nodes):
            cur = path[-1]
            out = self._step(h, cur, visited, cum_t)
            if out is None:
                break
            successors, logits, mask = out
            nxt = successors[int(torch.argmax(logits).item())]
            cum_t += self.edge_time[cur, nxt].item()
            cum_c += instance_get_cost(self.instance, cur, nxt)
            path.append(nxt)
            visited.add(nxt)
            if nxt == self.t:
                break
        return path, cum_c, cum_t

    def train_self_supervised(self, n_epochs: int = N_EPOCHS):
        rng = torch.Generator(device=self.device).manual_seed(0)
        for _ in range(n_epochs):
            self.opt.zero_grad()
            loss = 0.0
            feas = []
            for _b in range(BATCH):
                h = self.model.encode(self.node_feats, self.edge_index)
                path = [self.s]
                visited = {self.s}
                cum_t = 0.0
                cum_c = 0.0
                log_probs = []
                for _ in range(self.n_nodes):
                    cur = path[-1]
                    out = self._step(h, cur, visited, cum_t)
                    if out is None:
                        break
                    successors, logits, mask = out
                    probs = F.softmax(logits, dim=-1)
                    a = int(torch.multinomial(
                        probs, 1, generator=rng).item())
                    log_probs.append(torch.log(probs[a] + 1e-12))
                    cum_t += self.edge_time[cur, successors[a]].item()
                    cum_c += instance_get_cost(self.instance, cur,
                                               successors[a])
                    path.append(successors[a])
                    visited.add(successors[a])
                    if successors[a] == self.t:
                        break
                feasible = (path[-1] == self.t) and (cum_t <= self.T_max)
                if feasible:
                    feas.append((cum_c, log_probs))
            if not feas:
                continue
            best = min(c[0] for c in feas)
            batch_loss = torch.tensor(0.0, device=self.device,
                                      requires_grad=True)
            for c, lps in feas:
                adv = c - best
                lp = torch.stack(lps).sum()
                batch_loss = batch_loss + adv * lp
            batch_loss = batch_loss / len(feas)
            batch_loss.backward()
            self.opt.step()

    def solve(self) -> Dict:
        with torch.no_grad():
            h = self.model.encode(self.node_feats, self.edge_index)
            path, cost, time = self._greedy_decode(h)
        feasible = (path[-1] == self.t) and (time <= self.T_max)
        return {"path": path if feasible else None,
                "cost": float(cost) if feasible else float("inf"),
                "time": float(time),
                "feasible": bool(feasible)}


def instance_get_cost(instance: CSPPInstance, u: int, v: int) -> float:
    return float(instance.G[u][v]["cost"])


def solve(instance: CSPPInstance, n_epochs: int = N_EPOCHS,
          device: str = "cpu") -> Dict:
    gat = ConstrainedGAT(instance, device=device)
    gat.train_self_supervised(n_epochs)
    return gat.solve()
