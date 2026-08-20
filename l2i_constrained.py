"""
Constrained L2I (Learn to Improve) baseline for CSPP.

Implementation of Lu, Zhang, Yang (ICLR 2020) "A Learning-based Iterative
Method for Solving Vehicle Routing Problems", adapted from the original CVRP
setting to the constrained shortest path problem (CSPP).

Original algorithm (CVRP, N=20-100):
    - Start from a feasible initial solution.
    - At each step, an RL controller selects an "improvement operator" from
      a fixed pool (Table 5: 2-Opt, Symmetric-exchange(1), Relocate(1) as
      intra-route; Cross(2), Reverse-cross(2), Symmetric-exchange(2,m=1..3),
      Asymmetric-exchange(2,m!=n), Relocate(2,m=1..3), Cyclic-exchange(3)
      as inter-route; 17 actions total).
    - If a plateau is detected (no improvement for L consecutive steps),
      a rule-based perturbation operator is applied.
    - The RL controller is trained with REINFORCE; the policy is an
      attention network over (problem features, solution features,
      H-step action history).
    - An ensemble of 6 policies (with H = 1..6) is optionally used.

Adaptation to CSPP (single origin -> single destination, time constraint):
    - Inter-route operators are NOT applicable (single path, no multi-route).
    - Action space collapses to the three intra-route families:
        a0: 2-Opt                              (1 action)
        a1: Symmetric-exchange(1)              (1 action)
        a2: Symmetric-exchange(2)              (1 action)
        a3: Symmetric-exchange(3)              (1 action)
        a4: Relocate(1)                        (1 action)
        a5: Relocate(2)                        (1 action)
        a6: Relocate(3)                        (1 action)
      => 7 actions total (the original paper's "Symmetric-exchange(2,m=1,2,3"
         and "Relocate(2,m=1,2,3" are kept as 3+3 = 6 distinct actions).
    - Feasibility check on every operator output: if the new path's
      travel time exceeds T_max, the operator is rejected (state unchanged,
      reward = -1).  This mirrors the original paper's hard handling of
      the demand constraint in CVRP (operators that would violate capacity
      are not applied; the solution stays in the feasible region).
    - Perturbation operator: pick a random contiguous segment of the path
      and re-route it greedily with feasibility check.  This is the CSPP
      analogue of "ruin-and-recreate"; it keeps feasibility by construction.
    - Plateau rule: same as original paper (L = 6 consecutive no-improvement
      improvement steps triggers perturbation).
    - Training: REINFORCE with self-play baseline (best feasible path
      cost found so far, used as b(s)); RF1 reward (+1/-1).
    - Ensemble: 3 policies trained with H in {1, 3, 6}; final answer is the
      best of the 3 final solutions.

Output schema matches ``solve_cspp_cplex``:
    {"path": [...], "cost": float, "time": float, "feasible": bool}
"""
from __future__ import annotations

import math
import random
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional, Tuple

import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F

from dataset_generator import CSPPInstance, compute_path_cost, compute_path_time


# ---------------------------------------------------------------------------
# Algorithm hyperparameters (matching Lu et al. 2020, scaled to N=30)
# ---------------------------------------------------------------------------

D_HID = 64             # embedding dim (paper: 64)
N_HEADS = 8            # multi-head attention (paper: 8)
N_ACTIONS = 7          # 2-Opt + 3 SymExc + 3 Relocate
N_TRAIN_INSTANCES = 50 # number of training instances per call
T_ROLLOUT = 2000       # improvement iterations per instance
                     # (paper used 40000; we scale to N=30 for runtime)
L_PLATEAU = 6          # consecutive no-improvement steps -> perturb
EPS_GREEDY = 0.05      # epsilon-greedy exploration
LR = 1e-3              # Adam learning rate (paper: 1e-3)
BATCH_ROLLOUTS = 16    # rollouts per gradient update
H_HISTORY = 4          # history length (paper ensemble: 1..6; we fix=4)
N_ENSEMBLE = 3         # number of policies in the ensemble (paper: 6)
PERTURB_SEG_LEN = 3    # segment length for partial reroll perturbation

DEVICE = "cpu"


# Action constants for readability.
A_2OPT = 0
A_SYMEX_M1 = 1
A_SYMEX_M2 = 2
A_SYMEX_M3 = 3
A_RELOC_M1 = 4
A_RELOC_M2 = 5
A_RELOC_M3 = 6

ACTION_NAMES = {
    A_2OPT: "2-Opt",
    A_SYMEX_M1: "SymEx(1)", A_SYMEX_M2: "SymEx(2)", A_SYMEX_M3: "SymEx(3)",
    A_RELOC_M1: "Reloc(1)", A_RELOC_M2: "Reloc(2)", A_RELOC_M3: "Reloc(3)",
}


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def _path_time(G: nx.DiGraph, path: List[int]) -> float:
    return compute_path_time(G, path)


def _path_cost(G: nx.DiGraph, path: List[int]) -> float:
    return compute_path_cost(G, path)


def _path_feasible(G: nx.DiGraph, path: List[int], T_max: float) -> bool:
    """A path is feasible iff:
       (a) it is a simple walk from origin to destination using actual edges,
       (b) the cumulative time does not exceed T_max.
    """
    if not path or len(path) < 2:
        return False
    s, t = path[0], path[-1]
    if s != path[0] or t != path[-1]:  # defensive
        return False
    cum_t = 0.0
    for u, v in zip(path[:-1], path[1:]):
        if not G.has_edge(u, v):
            return False
        cum_t += G[u][v]["time"]
        if cum_t > T_max + 1e-9:
            return False
    return True


def _random_feasible_path(
    instance: CSPPInstance, rng: random.Random,
    max_tries: int = 200,
) -> Optional[List[int]]:
    """Stochastic depth-bounded DFS that returns any s->t walk satisfying the
    time constraint.  Each step uses an edge-time-weighted random choice so
    feasible paths are more likely to be short.
    """
    G: nx.DiGraph = instance.G
    s, t = instance.origin, instance.destination
    T_max = instance.T_max
    for _ in range(max_tries):
        path = [s]
        visited = {s}
        cum_t = 0.0
        cur = s
        ok = False
        for _ in range(G.number_of_nodes()):
            nxts = [v for v in G.successors(cur)
                    if v not in visited and v != cur]
            if not nxts:
                break
            # Bias toward shorter (lower time) edges.
            weights = [1.0 / max(G[cur][v]["time"], 1e-6) for v in nxts]
            total = sum(weights)
            r = rng.random() * total
            cum = 0.0
            chosen = nxts[-1]
            for v, w in zip(nxts, weights):
                cum += w
                if cum >= r:
                    chosen = v
                    break
            t_edge = G[cur][chosen]["time"]
            if cum_t + t_edge > T_max + 1e-9:
                # Try to go to t if possible.
                if chosen == t and cum_t + t_edge <= T_max + 1e-9:
                    pass  # would be infeasible anyway
                break
            cum_t += t_edge
            path.append(chosen)
            visited.add(chosen)
            cur = chosen
            if cur == t:
                ok = True
                break
        if ok and _path_feasible(G, path, T_max):
            return path
    # Fallback: a path of just [s, t] if it is feasible.
    if G.has_edge(s, t) and G[s][t]["time"] <= T_max + 1e-9:
        return [s, t]
    return None


# ---------------------------------------------------------------------------
# Improvement operators (intra-route only, with feasibility check)
# ---------------------------------------------------------------------------

def _op_2opt(
    path: List[int], G: nx.DiGraph, T_max: float, rng: random.Random
) -> Optional[List[int]]:
    """2-Opt: pick two random cut positions i<j, reverse the segment in
    between.  Reject if the new path is infeasible.

    Path structure note: 2-Opt on a s->t path is the classic TSP 2-opt
    move.  After reversal, we must also check that every consecutive
    pair is an actual edge in the graph (otherwise the new path is not
    a walk), so we test feasibility end-to-end.
    """
    n = len(path)
    if n < 4:
        return None
    i = rng.randrange(0, n - 2)
    j = rng.randrange(i + 2, n)
    new_path = path[: i + 1] + path[i + 1: j][::-1] + path[j:]
    if _path_feasible(G, new_path, T_max):
        return new_path
    return None


def _op_symmetric_exchange(
    path: List[int], G: nx.DiGraph, T_max: float, rng: random.Random,
    m: int,
) -> Optional[List[int]]:
    """Symmetric-exchange(m): pick two disjoint segments of length m
    and swap them.  Equivalent to swapping two blocks of m consecutive
    nodes in the path.

    For m=1 this reduces to swapping two single nodes.
    For m>=2 the segments must not overlap; we draw the start indices
    uniformly under that constraint.
    """
    n = len(path)
    # Segment start indices (excluding the fixed endpoints s=path[0] and
    # t=path[-1]; internal positions only).
    internal = list(range(1, n - 1))
    if len(internal) < 2 * m:
        return None
    # Sample two non-overlapping starts.
    starts = sorted(rng.sample(internal, 2 * m))
    i, j = starts[0], starts[m]
    if j - i < m:
        return None  # overlap; rare but possible
    # Build new path: [0:i] + path[j:j+m] + path[i+m:j] + path[i:i+m] + [j+m:]
    new_path = (
        path[:i]
        + path[j: j + m]
        + path[i + m: j]
        + path[i: i + m]
        + path[j + m:]
    )
    if _path_feasible(G, new_path, T_max):
        return new_path
    return None


def _op_relocate(
    path: List[int], G: nx.DiGraph, T_max: float, rng: random.Random,
    m: int,
) -> Optional[List[int]]:
    """Relocate(m): pick a contiguous segment of length m and re-insert it
    at a different position in the path.  Endpoints stay fixed.
    """
    n = len(path)
    internal = list(range(1, n - m))
    if len(internal) < 2:
        return None
    i = rng.choice(internal)              # segment start
    # Pick insertion point different from i..i+m.
    choices = [k for k in range(1, n - m + 1)
               if k < i - 1 or k > i + m - 1]
    if not choices:
        return None
    k = rng.choice(choices)
    segment = path[i: i + m]
    remainder = path[:i] + path[i + m:]
    # Insert at k relative to `remainder` of length n - m.
    # remainder index k corresponds to original position k (since we cut at i).
    if k > i:
        k = k - m  # shift because removing the segment shifted indices
    new_path = remainder[:k] + segment + remainder[k:]
    if _path_feasible(G, new_path, T_max):
        return new_path
    return None


def _apply_operator(
    action: int,
    path: List[int],
    G: nx.DiGraph,
    T_max: float,
    rng: random.Random,
) -> Tuple[Optional[List[int]], bool]:
    """Dispatch to the operator indexed by ``action``.  Returns
    ``(new_path, improved_flag)``.

    ``new_path`` is ``None`` if the operator produced an infeasible
    candidate (caller keeps the current path).
    ``improved_flag`` is True iff the new path exists and has strictly
    lower cost than the current path.
    """
    if action == A_2OPT:
        new_path = _op_2opt(path, G, T_max, rng)
    elif action == A_SYMEX_M1:
        new_path = _op_symmetric_exchange(path, G, T_max, rng, m=1)
    elif action == A_SYMEX_M2:
        new_path = _op_symmetric_exchange(path, G, T_max, rng, m=2)
    elif action == A_SYMEX_M3:
        new_path = _op_symmetric_exchange(path, G, T_max, rng, m=3)
    elif action == A_RELOC_M1:
        new_path = _op_relocate(path, G, T_max, rng, m=1)
    elif action == A_RELOC_M2:
        new_path = _op_relocate(path, G, T_max, rng, m=2)
    elif action == A_RELOC_M3:
        new_path = _op_relocate(path, G, T_max, rng, m=3)
    else:
        new_path = None
    if new_path is None:
        return None, False
    if _path_cost(G, new_path) < _path_cost(G, path):
        return new_path, True
    return new_path, False  # feasible but no improvement -> still update path


# ---------------------------------------------------------------------------
# Perturbation operator (CSPP version of "ruin & recreate")
# ---------------------------------------------------------------------------

def _perturb_partial_reroll(
    path: List[int], G: nx.DiGraph, T_max: float, rng: random.Random,
    seg_len: int = PERTURB_SEG_LEN,
) -> Optional[List[int]]:
    """Destroy a contiguous segment of length ``seg_len`` and re-route it
    greedily with feasibility check.  Preserves endpoints and the rest
    of the path.

    If the greedy re-route fails (no feasible re-connection), keep the
    current path untouched.
    """
    n = len(path)
    if n < seg_len + 2:
        return path
    # Pick the segment so that path[0] (origin) and path[-1] (destination)
    # are not included.
    start = rng.randrange(1, n - seg_len)
    pre = path[: start]                          # prefix ending at path[start-1]
    sub = path[start: start + seg_len]           # segment to remove
    post = path[start + seg_len:]                # suffix starting at path[start+seg_len]
    new_path = list(pre)
    cur = pre[-1]
    cum_t = 0.0
    for u, v in zip(pre[:-1], pre[1:]):
        cum_t += G[u][v]["time"]
    remaining = list(post)
    # Greedy: pick the lowest-cost unvisited neighbor that keeps feasibility.
    visited = set(new_path)
    # We need to insert seg_len new nodes bridging cur and remaining[0].
    # Simplest: greedy walk of length seg_len toward remaining[0], choosing
    # the lowest-cost outgoing edge that is feasible AND that eventually
    # reaches remaining[0].  To keep it tractable, we just DFS with depth
    # seg_len, backtracking on infeasibility.
    def dfs(depth: int, cur_node: int, cum: float,
            trail: List[int]) -> Optional[List[int]]:
        if depth == 0:
            # Check we can reach remaining[0] within T_max.
            if not G.has_edge(cur_node, remaining[0]):
                return None
            t_edge = G[cur_node][remaining[0]]["time"]
            if cum + t_edge > T_max + 1e-9:
                return None
            return trail + [remaining[0]]
        for nxt in list(G.successors(cur_node)):
            if nxt in visited or nxt in trail:
                continue
            t_edge = G[cur_node][nxt]["time"]
            if cum + t_edge > T_max + 1e-9:
                continue
            res = dfs(depth - 1, nxt, cum + t_edge, trail + [nxt])
            if res is not None:
                return res
        return None

    bridge = dfs(seg_len, cur, cum_t, [])
    if bridge is None:
        return path  # perturbation failed -> keep state
    new_path = pre + bridge + post
    if _path_feasible(G, new_path, T_max):
        return new_path
    return path


# ---------------------------------------------------------------------------
# Policy network (8-head attention + 2-layer MLP, matching paper §2.1.3)
# ---------------------------------------------------------------------------

class _PolicyNetwork(nn.Module):
    """Attention-based policy network as in Lu et al. (ICLR 2020, Fig. 6).

    Components (paper §2.1.3, "Policy Network"):
      - problem- and solution-specific input features -> embedding of dim D=64
      - attention layer with 8 heads and 64 output units
      - concatenated with the H-step running history (action one-hot + effect)
      - two fully-connected layers (64 units + ReLU, then softmax over |A|)
    """

    def __init__(
        self, d_in: int = 4, d_hid: int = D_HID,
        n_actions: int = N_ACTIONS, h_history: int = H_HISTORY,
        n_heads: int = N_HEADS,
    ):
        super().__init__()
        self.embed = nn.Linear(d_in, d_hid)
        self.attn = nn.MultiheadAttention(d_hid, n_heads, batch_first=True)
        self.fc1 = nn.Linear(d_hid + h_history * (n_actions + 1), d_hid)
        self.fc2 = nn.Linear(d_hid, n_actions)
        self.n_actions = n_actions
        self.h_history = h_history

    def forward(
        self,
        node_feats: torch.Tensor,            # (N, d_in)
        history: torch.Tensor,               # (B, h_history * (n_actions+1))
    ) -> torch.Tensor:
        """Returns action logits of shape (B, n_actions)."""
        h = self.embed(node_feats).unsqueeze(0)        # (1, N, d_hid)
        h_attn, _ = self.attn(h, h, h)                 # (1, N, d_hid)
        # Mean-pool over nodes to get a single graph-level representation.
        graph_emb = h_attn.mean(dim=1)                 # (1, d_hid)
        # Broadcast to batch and concat history.
        graph_emb = graph_emb.expand(history.size(0), -1)
        x = torch.cat([graph_emb, history], dim=-1)    # (B, d_hid + h*...)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


# ---------------------------------------------------------------------------
# Solver wrapper: trains REINFORCE on a single instance then solves
# ---------------------------------------------------------------------------

@dataclass
class _HistoryEntry:
    action: int   # 0..N_ACTIONS-1
    effect: int   # +1 (improved) or -1 (not improved / infeasible)


class ConstrainedL2I:
    """Lu et al. (ICLR 2020) Learn-to-Improve adapted to CSPP.

    Hyperparameters (matching the paper, scaled for N=30):
      * L_PLATEAU = 6 (no-improvement steps before perturbation)
      * EPS_GREEDY = 0.05 (exploration)
      * T_ROLLOUT = 2000 (rollout steps per instance)
      * REINFORCE with self-play baseline (best feasible cost so far)
      * Reward = +1 if the operator improves cost, -1 otherwise
        (RF1 of the paper; identical to Eq. in §2.1.4)
      * Adam, lr = 1e-3
    """

    def __init__(
        self,
        instance: CSPPInstance,
        device: str = DEVICE,
        h_history: int = H_HISTORY,
        t_rollout: int = T_ROLLOUT,
        lr: float = LR,
        seed: int = 0,
    ):
        self.instance = instance
        self.device = torch.device(device)
        self.G: nx.DiGraph = instance.G
        self.s = instance.origin
        self.t = instance.destination
        self.T_max = instance.T_max
        self.n_nodes = self.G.number_of_nodes()
        self.h_history = h_history
        self.t_rollout = t_rollout
        self.rng = random.Random(seed)
        self.torch_seed = seed

        # Build per-node features: [in-degree-norm, out-degree-norm,
        # 1{is origin}, 1{is destination}].  This is the "problem-specific"
        # part of the state.  Solution-specific features are added at
        # encoding time (we embed the current path edges via the path
        # masks that mark which consecutive pair is active).
        self._node_feats = self._build_node_features()

        self.model = _PolicyNetwork(
            d_in=self._node_feats.size(1),
            h_history=h_history,
        ).to(self.device)
        self.opt = torch.optim.Adam(self.model.parameters(), lr=lr)

        # Initial feasible solution.
        self.current = _random_feasible_path(instance, self.rng)
        self.best = list(self.current) if self.current is not None else None
        self.best_cost = (
            _path_cost(self.G, self.current)
            if self.current is not None else float("inf")
        )

    def _build_node_features(self) -> torch.Tensor:
        feats = torch.zeros(self.n_nodes, 4, dtype=torch.float32)
        for v in range(self.n_nodes):
            feats[v, 0] = self.G.in_degree(v) / self.n_nodes
            feats[v, 1] = self.G.out_degree(v) / self.n_nodes
            feats[v, 2] = 1.0 if v == self.s else 0.0
            feats[v, 3] = 1.0 if v == self.t else 0.0
        return feats.to(self.device)

    def _encode_history(self, hist: Deque[_HistoryEntry]) -> torch.Tensor:
        """Encode the last ``h_history`` entries as a flat vector
        ``[action_1_hot, effect_1, action_2_hot, effect_2, ...]``.
        Missing entries (cold start) are zero-padded.
        """
        flat = torch.zeros(self.h_history * (self.model.n_actions + 1),
                           dtype=torch.float32, device=self.device)
        items = list(hist)[-self.h_history:]
        # Right-align: oldest entry at the start.
        offset = (self.h_history - len(items)) * (self.model.n_actions + 1)
        for k, entry in enumerate(items):
            base = offset + k * (self.model.n_actions + 1)
            flat[base + entry.action] = 1.0
            flat[base + self.model.n_actions] = (
                1.0 if entry.effect > 0 else -1.0
            )
        return flat.unsqueeze(0)  # (1, ...)

    def _select_action(
        self,
        node_feats: torch.Tensor,
        hist: Deque[_HistoryEntry],
        rng: random.Random,
    ) -> int:
        # epsilon-greedy exploration.
        if rng.random() < EPS_GREEDY:
            return rng.randrange(self.model.n_actions)
        hist_vec = self._encode_history(hist)
        with torch.no_grad():
            logits = self.model(node_feats, hist_vec)
        return int(torch.argmax(logits, dim=-1).item())

    def _log_prob(self, action: int,
                  node_feats: torch.Tensor,
                  hist_vec: torch.Tensor) -> torch.Tensor:
        logits = self.model(node_feats, hist_vec)
        log_p = F.log_softmax(logits, dim=-1)
        return log_p[0, action]

    def train_self_supervised(self, n_epochs: int = 1) -> None:
        """Train REINFORCE on the current instance for ``t_rollout``
        improvement steps (the "improvement iteration" in the paper).

        REINFORCE update with self-play baseline ``b(s) = best feasible
        cost so far`` (paper §2.1.4, Eq. for the policy gradient with
        baseline function ``b(s)``).
        """
        torch.manual_seed(self.torch_seed)
        for _ in range(n_epochs):
            hist: Deque[_HistoryEntry] = deque(maxlen=self.h_history)
            no_improve = 0
            traj_logp: List[torch.Tensor] = []
            traj_reward: List[float] = []

            for step in range(self.t_rollout):
                # ---------------- Meta controller ----------------
                # Plateau rule: L consecutive non-improving steps -> perturb.
                if no_improve >= L_PLATEAU:
                    new_path = _perturb_partial_reroll(
                        self.current, self.G, self.T_max, self.rng)
                    if new_path is not self.current and _path_cost(
                            self.G, new_path) < _path_cost(self.G, self.current):
                        self.current = new_path
                    else:
                        # perturbation gave a worse path: still apply
                        # (paper: "perturbing generates new starting
                        # solutions, often worse"), and reset plateau.
                        if new_path is not None:
                            self.current = new_path
                    no_improve = 0
                    hist.clear()
                    # The perturbation step itself is not a learning event
                    # (rule-based), so we don't collect a log-prob for it.
                    continue

                # ---------------- Improvement controller ---------
                action = self._select_action(
                    self._node_feats, hist, self.rng)
                hist_vec = self._encode_history(hist)
                log_p = self._log_prob(
                    action, self._node_feats, hist_vec)

                old_cost = _path_cost(self.G, self.current)
                new_path, improved = _apply_operator(
                    action, self.current, self.G, self.T_max, self.rng)
                effect = +1 if improved else -1
                if new_path is not None:
                    self.current = new_path

                if improved:
                    cur_cost = _path_cost(self.G, self.current)
                    if cur_cost < self.best_cost:
                        self.best_cost = cur_cost
                        self.best = list(self.current)
                    no_improve = 0
                else:
                    no_improve += 1

                traj_logp.append(log_p)
                traj_reward.append(float(effect))
                hist.append(_HistoryEntry(action=action, effect=effect))

            # ---------------- REINFORCE update ----------------
            if not traj_logp:
                continue
            # Self-play baseline = best feasible cost so far (paper's
            # "total distance achieved during the first improvement
            # iteration is taken as a baseline"; here we use the running
            # best).
            baseline = self.best_cost
            # Advantage: positive if reward better than baseline signal.
            # We use RF1 (+1/-1) so the gradient direction equals reward.
            loss = torch.tensor(0.0, device=self.device, requires_grad=True)
            for log_p, r in zip(traj_logp, traj_reward):
                loss = loss - log_p * r
            loss = loss / max(len(traj_logp), 1)
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()

    def solve(self) -> Dict:
        """Return the best feasible path found during training."""
        if self.best is None:
            return {"path": None, "cost": float("inf"),
                    "time": float("inf"), "feasible": False}
        cost = _path_cost(self.G, self.best)
        time = _path_time(self.G, self.best)
        return {"path": list(self.best),
                "cost": float(cost),
                "time": float(time),
                "feasible": bool(time <= self.T_max + 1e-9)}


# ---------------------------------------------------------------------------
# Top-level solve() with optional ensemble (paper's "ensemble method")
# ---------------------------------------------------------------------------

def solve(
    instance: CSPPInstance,
    device: str = DEVICE,
    n_policies: int = 1,
    t_rollout: int = T_ROLLOUT,
    seed: int = 0,
    # Backward-compat shim for the old call site in run_all_baselines.py.
    n_epochs: int = 1,
    time_limit: float = None,
    **_unused: object,
) -> Dict:
    """Train + infer Lu-et-al L2I on a single CSPP instance.

    The new default hyperparameters mirror the paper scaled to N=30:
        n_policies = 1   (paper ensemble uses 6; we keep 1 by default to
                         stay comparable with AM/GAT single-policy setups)
        t_rollout  = 2000 (paper uses 40000; we scale down because N=30
                         instances give many more improvement opportunities
                         per rollout step than N=100 CVRP instances)

    If ``n_policies > 1``, trains an ensemble of ``n_policies`` policies
    with different random seeds and history lengths (paper §3, "ensemble
    method"), and returns the best solution across the ensemble.

    The legacy keyword arguments ``n_epochs`` and ``time_limit`` are
    accepted for backward compatibility: ``n_epochs`` runs that many
    REINFORCE improvement-iterations on the instance (paper does one
    iteration; ``time_limit`` is unused because L2I does not invoke
    an external solver).
    """
    epochs = max(int(n_epochs), 1)
    if n_policies <= 1:
        agent = ConstrainedL2I(
            instance, device=device, h_history=H_HISTORY,
            t_rollout=t_rollout, seed=seed,
        )
        for _ in range(epochs):
            agent.train_self_supervised(n_epochs=1)
        return agent.solve()

    # Ensemble: each policy uses a different history length and seed.
    history_lens = [1, 3, 6][: n_policies]
    if len(history_lens) < n_policies:
        history_lens += [H_HISTORY] * (n_policies - len(history_lens))

    best_result = {"path": None, "cost": float("inf"),
                   "time": float("inf"), "feasible": False}
    for k, h in enumerate(history_lens):
        agent = ConstrainedL2I(
            instance, device=device, h_history=h,
            t_rollout=t_rollout, seed=seed + 1000 * k,
        )
        for _ in range(epochs):
            agent.train_self_supervised(n_epochs=1)
        res = agent.solve()
        if res["feasible"] and res["cost"] < best_result["cost"]:
            best_result = res
    return best_result