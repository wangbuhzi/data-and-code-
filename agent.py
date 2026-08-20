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
    
    def __init__(
        self,
        env: CSPEnvironment,
        embedding_dim: int = 64,
        hidden_dims: List[int] = [128, 128, 64],
        num_propagations: int = 3,
        device: str = 'cpu'
    ):
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
        
        candidates = state.get_available_actions()
        return self.select_action(state, epsilon=0.0, candidates=candidates).action
    
    def construct_path(
        self,
        state: State,
        max_steps: int = 100
    ) -> Tuple[List[int], float, float, bool]:
    
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
    
        for target_param, param in zip(
            self.target_q_network.parameters(),
            self.q_network.parameters()
        ):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
    
    def hard_update_target_network(self):
        
        self.target_q_network.load_state_dict(self.q_network.state_dict())
    
    def get_parameters(self) -> Dict[str, torch.Tensor]:
      
        return {
            'structure2vec': self.structure2vec.state_dict(),
            'q_network': self.q_network.state_dict()
        }
    
    def set_parameters(self, params: Dict[str, Dict]):
     
        if 'structure2vec' in params:
            self.structure2vec.load_state_dict(params['structure2vec'])
        if 'q_network' in params:
            self.q_network.load_state_dict(params['q_network'])
    
    def save(self, filepath: str):
   
        torch.save({
            'structure2vec': self.structure2vec.state_dict(),
            'q_network': self.q_network.state_dict(),
            'target_q_network': self.target_q_network.state_dict()
        }, filepath)
    
    def load(self, filepath: str):
    
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
    violation = T_P - T_max
    new_lambda = lambda_current + eta_lambda * violation
    return max(0.0, new_lambda)
