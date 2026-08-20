import random
from typing import Dict, List, Optional

import networkx as nx

from dataset_generator import CSPPInstance


ALPHA = 1.0
BETA = 2.0
RHO = 0.1
N_ANTS = 50
N_ITER = 100
EPS = 1e-6


def _select_next(
    rng: random.Random,
    current: int,
    successors: List[int],
    visited: set,
    pheromone: Dict[tuple, float],
    heuristic: Dict[tuple, float],
) -> Optional[int]:

    candidates = [v for v in successors if v not in visited]
    if not candidates:
        return None
    weights = []
    for v in candidates:
        tau = pheromone.get((current, v), 1.0)
        eta = heuristic.get((current, v), 0.0)
        weights.append((tau ** ALPHA) * (eta ** BETA))
    total = sum(weights)
    if total <= 0:
        return rng.choice(candidates)
    r = rng.random() * total
    cum = 0.0
    for v, w in zip(candidates, weights):
        cum += w
        if cum >= r:
            return v
    return candidates[-1]


def solve(
    instance: CSPPInstance,
    n_ants: int = N_ANTS,
    n_iter: int = N_ITER,
    seed: Optional[int] = None,
) -> Dict:

    rng = random.Random(seed)
    G: nx.DiGraph = instance.G
    s, t = instance.origin, instance.destination
    T_max = instance.T_max

    pheromone = {(u, v): 1.0 for (u, v) in G.edges()}
    heuristic = {(u, v): 1.0 / max(d["time"], EPS)
                 for (u, v, d) in G.edges(data=True)}

    # Precompute successor lists for speed.
    succ = {u: list(G.successors(u)) for u in G.nodes()}

    best_path: Optional[List[int]] = None
    best_cost = float("inf")

    for _ in range(n_iter):
        iter_paths: List[List[int]] = []
        iter_costs: List[float] = []

        for _ in range(n_ants):
            path = [s]
            visited = {s}
            cur = s
            cum_t = 0.0
            cum_c = 0.0

            while cur != t and len(path) < G.number_of_nodes():
                nxt = _select_next(rng, cur, succ[cur], visited,
                                   pheromone, heuristic)
                if nxt is None:
                    break
                cum_t += G[cur][nxt]["time"]
                cum_c += G[cur][nxt]["cost"]
                # Feasibility pruning.
                if cum_t > T_max:
                    cum_c = float("inf")
                    break
                path.append(nxt)
                visited.add(nxt)
                cur = nxt

            if cur == t and cum_c < float("inf"):
                iter_paths.append(path)
                iter_costs.append(cum_c)

        # Evaporation.
        for k in pheromone:
            pheromone[k] *= (1.0 - RHO)
        # Deposit (only feasible ants).
        for path, cost in zip(iter_paths, iter_costs):
            if cost <= 0:
                continue
            deposit = 1.0 / cost
            for u, v in zip(path[:-1], path[1:]):
                pheromone[(u, v)] += deposit

        for path, cost in zip(iter_paths, iter_costs):
            if cost < best_cost:
                best_cost = cost
                best_path = path

    if best_path is None:
        return {"path": None, "cost": float("inf"),
                "time": float("inf"), "feasible": False}

    cost = sum(G[u][v]["cost"] for u, v in zip(best_path[:-1], best_path[1:]))
    time = sum(G[u][v]["time"] for u, v in zip(best_path[:-1], best_path[1:]))
    return {"path": best_path, "cost": float(cost),
            "time": float(time), "feasible": bool(time <= T_max)}
