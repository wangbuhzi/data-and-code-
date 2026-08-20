"""
Agent for E2E-GERL.

Implements the ε-greedy policy and path construction logic
as described in Sections 3.6 and 4.3 of the paper.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass

from environment import CSPEnvironment, State, Transition
from structure2vec import Structure2Vec, build_node_features, build_edge_index_and_features


@dataclass
class ActionSelectionResult:
    """Result of an action selection."""
    action: int
    q_value: float
    is_exploratory: bool
    q_values: Optional[List[float]] = None


class E2EGERLAgent:
    """
    End-to-End Graph Embedding Reinforcement Learning Agent.
    
    Implements:
    - Structure2Vec-based graph embedding
    - Q-value computation for candidate actions
    - ε-greedy action selection
    - Path construction with feasibility masking
    """
    
    def __init__(
        self,
        env: CSPEnvironment,
        embedding_dim: int = 64,
        hidden_dims: List[int] = [128, 128, 64],
        num_propagations: int = 3,
        device: str = 'cpu'
    ):
        """
        Args:
            env: The CSP environment
            embedding_dim: Dimension of node embeddings
            hidden_dims: Hidden layer dimensions for Q-network
            num_propagations: Number of Structure2Vec propagation iterations
            device: Device for computation ('cpu' or 'cuda')
        """
        self.env = env
        self.device = torch.device(device)
        
        self.embedding_dim = embedding_dim
        self.num_propagations = num_propagations
        
        self.structure2vec = Structure2Vec(
            node_feature_dim=6,
            embedding_dim=embedding_dim,
            num_propagations=num_propagations
        ).to(self.device)
        
        state_action_dim = embedding_dim * 3 + embedding_dim + 4
        
        layers = []
        prev_dim = state_action_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.ReLU())
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, 1))
        
        self.q_network = nn.Sequential(*layers).to(self.device)
        
        target_layers = []
        prev_dim = state_action_dim
        for h_dim in hidden_dims:
            target_layers.append(nn.Linear(prev_dim, h_dim))
            target_layers.append(nn.ReLU())
            prev_dim = h_dim
        target_layers.append(nn.Linear(prev_dim, 1))
        
        self.target_q_network = nn.Sequential(*target_layers).to(self.device)
        self.target_q_network.load_state_dict(self.q_network.state_dict())
        
        self.cached_edge_index: Optional[torch.Tensor] = None
        self.cached_edge_features: Optional[Dict] = None
        self.cached_num_nodes: Optional[int] = None
        self.cached_avg_cost: Optional[float] = None
        self.cached_avg_time: Optional[float] = None
    
    def _get_or_compute_embeddings(self, state: State) -> Tuple[torch.Tensor, Dict]:
        """
        Get cached node embeddings or compute them if state changed.

        --------------------------------------------------------------------------
        Eq. 15-16: when does Structure2Vec actually re-run?

        Two scenarios trigger a full re-computation:

          1. The graph instance has changed (new node count, new edge
             structure, new avg_edge_cost / time).  In that case the
             edge index AND the cached edge feature dictionary are
             invalidated and rebuilt -- this is what ``ensure_path``
             does when a new instance is loaded.

          2. The cumulative cost / time / visited set has changed.
             Then the node features x_v^(t) (Eq. 13) change for every
             node, so the W_x · x_v^(t) projection and the K
             Structure2Vec propagation steps (Eq. 15) MUST re-run.
             This is what the inner ``self.structure2vec(...)`` call
             does at every decision step.

        Crucially, the *edge* part of the update (W_e · Σ φ_e(e_vu))
        is shared across the whole episode: edge features do not
        depend on cumulative cost / time, so we re-use the cached
        ``edge_feat_dict`` and ``edge_index``.  The cost is therefore

            O(K · |E| · d)  per decision step,

        not O(K · |V|² · d) as a naïve GAT-style implementation would
        pay.  See ``profile_embedding_cost.py`` for the empirical
        numbers.
        --------------------------------------------------------------------------

        Args:
            state: Current state

        Returns:
            Tuple of (node_embeddings, edge_features_dict)
        """
        G = state.graph
        
        if (self.cached_edge_index is None or
            self.cached_num_nodes != len(G.G.nodes()) or
            self.cached_avg_cost != G.avg_edge_cost):
            self.cached_num_nodes = len(G.G.nodes())
            self.cached_avg_cost = G.avg_edge_cost
            self.cached_avg_time = G.avg_edge_time
            self.cached_edge_index = None
            self.cached_edge_features = None
        
        if self.cached_edge_index is None:
            edge_index, edge_features, edge_feat_dict = build_edge_index_and_features(
                G.G, G.avg_edge_cost, G.avg_edge_time, self.device
            )
            self.cached_edge_index = edge_index
            self.cached_edge_features = edge_feat_dict
        else:
            edge_index = self.cached_edge_index
            edge_feat_dict = self.cached_edge_features
        
        node_features = build_node_features(
            current_node=state.current_node,
            destination_node=state.destination,
            visited_nodes=state.visited_nodes,
            cumulative_cost=state.cumulative_cost,
            cumulative_time=state.cumulative_time,
            T_max=state.T_max,
            avg_cost=G.avg_edge_cost,
            num_nodes=len(G.G.nodes()),
            device=self.device
        )
        
        dummy_edge_features = torch.zeros(
            (edge_index.shape[1], 2), device=self.device, dtype=torch.float32
        )
        
        embeddings = self.structure2vec(
            node_features=node_features,
            edge_index=edge_index,
            edge_features=dummy_edge_features,
            num_nodes=len(G.G.nodes())
        )
        
        return embeddings, edge_feat_dict
    
    def _build_state_action_repr(
        self,
        state: State,
        candidate: int,
        embeddings: torch.Tensor,
        edge_features: Dict
    ) -> torch.Tensor:
        """
        Build state-action representation h_Θ(s_t, u) from Eq. (17).
        
        Args:
            state: Current state
            candidate: Candidate action node
            embeddings: Node embeddings from Structure2Vec
            edge_features: Edge feature dictionary
        
        Returns:
            State-action representation tensor
        """
        G = state.graph
        
        mu_vt = embeddings[state.current_node]
        mu_u = embeddings[candidate]
        mu_d = embeddings[state.destination]
        
        edge_key = (state.current_node, candidate)
        if edge_key in edge_features:
            phi_e = self.structure2vec.edge_net(edge_features[edge_key].unsqueeze(0).float()).squeeze(0)
        else:
            phi_e = torch.zeros(self.embedding_dim, device=self.device, dtype=torch.float32)
        
        cost_norm = torch.tensor(
            [state.cumulative_cost / G.avg_edge_cost if G.avg_edge_cost > 0 else 0.0],
            device=self.device,
            dtype=torch.float32
        )
        time_ratio = torch.tensor(
            [state.cumulative_time / state.T_max if state.T_max > 0 else 0.0],
            device=self.device,
            dtype=torch.float32
        )
        time_remaining = torch.tensor(
            [(state.T_max - state.cumulative_time) / state.T_max if state.T_max > 0 else 0.0],
            device=self.device,
            dtype=torch.float32
        )
        is_dest = torch.tensor(
            [1.0 if candidate == state.destination else 0.0],
            device=self.device,
            dtype=torch.float32
        )
        
        h = torch.cat([mu_vt, mu_u, mu_d, phi_e, cost_norm, time_ratio, time_remaining, is_dest])
        
        return h
    
    def compute_q_values(
        self,
        state: State,
        candidates: List[int]
    ) -> Tuple[List[float], List[torch.Tensor]]:
        """
        Compute Q-values for all candidate actions.
        
        Args:
            state: Current state
            candidates: List of candidate nodes
        
        Returns:
            Tuple of (q_values_list, state_action_reprs)
        """
        if not candidates:
            return [], []
        
        embeddings, edge_features = self._get_or_compute_embeddings(state)
        
        state_action_reprs = []
        for candidate in candidates:
            repr_tensor = self._build_state_action_repr(
                state, candidate, embeddings, edge_features
            )
            state_action_reprs.append(repr_tensor)
        
        batch_reprs = torch.stack(state_action_reprs).to(self.device)
        
        q_values_tensor = self.q_network(batch_reprs).squeeze(-1)
        
        q_values_list = q_values_tensor.detach().cpu().tolist()
        
        return q_values_list, state_action_reprs
    
    def compute_q_values_target(
        self,
        state: State,
        candidates: List[int]
    ) -> Tuple[List[float], List[torch.Tensor]]:
        """
        Compute Q-values using the target network.
        
        Used for bootstrapping in Q-learning targets.
        
        Args:
            state: Current state
            candidates: List of candidate nodes
        
        Returns:
            Tuple of (q_values_list, state_action_reprs)
        """
        if not candidates:
            return [], []
        
        embeddings, edge_features = self._get_or_compute_embeddings(state)
        
        state_action_reprs = []
        for candidate in candidates:
            repr_tensor = self._build_state_action_repr(
                state, candidate, embeddings, edge_features
            )
            state_action_reprs.append(repr_tensor)
        
        batch_reprs = torch.stack(state_action_reprs).to(self.device)
        
        with torch.no_grad():
            q_values_tensor = self.target_q_network(batch_reprs).squeeze(-1)
        
        q_values_list = q_values_tensor.detach().cpu().tolist()
        
        return q_values_list, state_action_reprs
    
    def compute_q_values_train(
        self,
        state: State,
        candidates: List[int],
        action_idx: int
    ) -> torch.Tensor:
        """
        Compute Q-value for a specific action during training (with gradients).
        
        Args:
            state: Current state
            candidates: List of candidate nodes
            action_idx: Index of the action to compute Q-value for
        
        Returns:
            Q-value scalar tensor (with gradients)
        """
        if not candidates:
            return torch.zeros(1, device=self.device, requires_grad=True)
        
        embeddings, edge_features = self._get_or_compute_embeddings(state)
        
        repr_tensor = self._build_state_action_repr(
            state, candidates[action_idx], embeddings, edge_features
        )
        
        repr_batch = repr_tensor.unsqueeze(0)
        q_value = self.q_network(repr_batch).squeeze(-1)
        
        return q_value
    
    def select_action(
        self,
        state: State,
        epsilon: float = 0.0,
        candidates: Optional[List[int]] = None
    ) -> ActionSelectionResult:
        """
        Select action using ε-greedy policy from Eq. (29).
        
        Args:
            state: Current state
            epsilon: Exploration probability
            candidates: List of valid candidates (computed if None)
        
        Returns:
            ActionSelectionResult with chosen action and metadata
        """
        if candidates is None:
            candidates = state.get_available_actions()
        
        if not candidates:
            raise ValueError("No valid actions available")
        
        q_values, _ = self.compute_q_values(state, candidates)
        
        is_exploratory = np.random.random() < epsilon
        
        if is_exploratory:
            action = int(np.random.choice(candidates))
            action_idx = candidates.index(action)
            q_value = q_values[action_idx]
        else:
            max_q = max(q_values)
            max_indices = [i for i, q in enumerate(q_values) if abs(q - max_q) < 1e-6]
            action_idx = np.random.choice(max_indices)
            action = candidates[action_idx]
            q_value = q_values[action_idx]
        
        return ActionSelectionResult(
            action=action,
            q_value=q_value,
            is_exploratory=is_exploratory,
            q_values=q_values
        )
    
    def get_best_action(self, state: State) -> int:
        """
        Get the best action according to current Q-values (greedy).
        
        Args:
            state: Current state
        
        Returns:
            Best action (no exploration)
        """
        candidates = state.get_available_actions()
        return self.select_action(state, epsilon=0.0, candidates=candidates).action
    
    def construct_path(
        self,
        state: State,
        max_steps: int = 100
    ) -> Tuple[List[int], float, float, bool]:
        """
        Construct a path from current state to destination.
        
        Args:
            state: Starting state
            max_steps: Maximum number of steps
        
        Returns:
            Tuple of (path, total_cost, total_time, success)
        """
        path = [state.current_node]
        current_state = state
        steps = 0
        
        while current_state.current_node != state.destination and steps < max_steps:
            candidates = current_state.get_available_actions()
            
            if not candidates:
                return path, current_state.cumulative_cost, current_state.cumulative_time, False
            
            action = self.get_best_action(current_state)
            
            next_state, _, done, info = self.env.step(action)
            path.append(action)
            current_state = next_state
            steps += 1
            
            if done:
                break
        
        success = current_state.current_node == state.destination
        
        return path, current_state.cumulative_cost, current_state.cumulative_time, success
    
    def update_target_network(self, tau: float = 0.005):
        """
        Soft update of target network from Eq. (28).
        
        Ψ̄ ← τΨ + (1 - τ)Ψ̄
        
        Args:
            tau: Soft update coefficient
        """
        for target_param, param in zip(
            self.target_q_network.parameters(),
            self.q_network.parameters()
        ):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
    
    def hard_update_target_network(self):
        """Hard update: copy Q-network to target network."""
        self.target_q_network.load_state_dict(self.q_network.state_dict())
    
    def get_parameters(self) -> Dict[str, torch.Tensor]:
        """Get all trainable parameters."""
        return {
            'structure2vec': self.structure2vec.state_dict(),
            'q_network': self.q_network.state_dict()
        }
    
    def set_parameters(self, params: Dict[str, Dict]):
        """Set parameters from a dictionary."""
        if 'structure2vec' in params:
            self.structure2vec.load_state_dict(params['structure2vec'])
        if 'q_network' in params:
            self.q_network.load_state_dict(params['q_network'])
    
    def save(self, filepath: str):
        """Save model parameters."""
        torch.save({
            'structure2vec': self.structure2vec.state_dict(),
            'q_network': self.q_network.state_dict(),
            'target_q_network': self.target_q_network.state_dict()
        }, filepath)
    
    def load(self, filepath: str):
        """Load model parameters."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.structure2vec.load_state_dict(checkpoint['structure2vec'])
        self.q_network.load_state_dict(checkpoint['q_network'])
        self.target_q_network.load_state_dict(checkpoint['target_q_network'])


def update_lambda(
    lambda_current: float,
    T_P: float,
    T_max: float,
    eta_lambda: float = 0.1
) -> float:
    """
    Update λ using projected subgradient rule from Eq. (6).
    
    λ^{(l+1)} = [λ^{(l)} + η_λ(T(P^{(l)}) - T_max)]_+
    
    Args:
        lambda_current: Current λ value
        T_P: Path time
        T_max: Maximum allowed time
        eta_lambda: Step size η_λ
    
    Returns:
        Updated λ value
    """
    violation = T_P - T_max
    new_lambda = lambda_current + eta_lambda * violation
    return max(0.0, new_lambda)
