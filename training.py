import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
from collections import deque
import copy

from environment import CSPEnvironment, State, Transition
from agent import E2EGERLAgent, update_lambda
from graph_utils import GraphInstance, create_instance


@dataclass
class TrainingStats:
    episode: int
    episode_reward: float
    path_cost: float
    path_time: float
    path_feasible: bool
    num_steps: int
    epsilon: float
    loss: Optional[float] = None
    lambda_value: float = 0.0


class NStepReturnBuffer:
   

    def __init__(self, n_step: int = 5, gamma: float = 1.0):
        self.n_step = n_step
        self.gamma = gamma
        self.buffer: deque = deque(maxlen=n_step)

    def add(self, state: State, action: int, reward: float, done: bool):
        self.buffer.append({
            'state': state,
            'action': action,
            'reward': reward,
            'done': done
        })

    def get_n_step_return(
        self,
        bootstrap_state: Optional[State] = None,
        bootstrap_done: bool = False
    ) -> Tuple[Optional[State], Optional[int], Optional[float], Optional[State], bool]:
        if not self.buffer:
            return None, None, None, None, False

        first = self.buffer[0]
        last = self.buffer[-1]

        n_step_return = 0.0
        for k, entry in enumerate(self.buffer):
            n_step_return += (self.gamma ** k) * entry['reward']

        return (
            first['state'],
            first['action'],
            n_step_return,
            last['state'],
            last['done']
        )

    def clear(self):
        self.buffer.clear()

    def __len__(self) -> int:
        return len(self.buffer)


class E2EGERLTrainer:

    def __init__(
        self,
        agent: E2EGERLAgent,
        env: CSPEnvironment,
        learning_rate: float = 1e-4,
        gamma: float = 1.0,
        n_step: int = 2,
        batch_size: int = 64,
        replay_capacity: int = 10000,
        epsilon_start: float = 1.0,
        epsilon_min: float = 0.1,
        epsilon_decay: float = 0.995,
        target_update_freq: int = 10,
        tau: float = 0.005,
        max_steps_per_episode: int = 1000,
        device: str = 'gpu'
    ):
      
        self.agent = agent
        self.env = env

        self.gamma = gamma
        self.n_step = n_step
        self.batch_size = batch_size
        self.epsilon = epsilon_start
        self._epsilon_start = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.target_update_freq = target_update_freq
        self.tau = tau
        self.max_steps = max_steps_per_episode
        self.device = torch.device(device)

        self.lambda_penalty = lambda_init
        self.eta_lambda = eta_lambda

        self.replay_memory: List[Transition] = []
        self.replay_capacity = replay_capacity

        self.n_step_buffer = NStepReturnBuffer(n_step=n_step, gamma=gamma)

        params = list(agent.structure2vec.parameters()) + list(agent.q_network.parameters())
        self.optimizer = optim.Adam(params, lr=learning_rate)

        self.loss_fn = nn.MSELoss()

        self.training_stats: List[TrainingStats] = []
        self.episode_count = 0

    def set_lambda(self, value: float):
      
        self.lambda_penalty = float(value)
        self.env.lambda_penalty = float(value)

    def _compute_epsilon(self, episode: int) -> float:
       
        return max(self.epsilon_min, self._epsilon_start * (self.epsilon_decay ** episode))

    def _store_transition(self, transition: Transition):
       
        if len(self.replay_memory) >= self.replay_capacity:
            self.replay_memory.pop(0)
        self.replay_memory.append(transition)

    def _sample_batch(self) -> List[Transition]:
      
        if len(self.replay_memory) < self.batch_size:
            return self.replay_memory.copy()
        return list(np.random.choice(self.replay_memory, self.batch_size, replace=False))

    def _compute_target(
        self,
        transitions: List[Transition]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        current_q_values = []
        target_values = []

        for trans in transitions:
            state = trans.state
            action = trans.action

            candidates = state.get_available_actions()

            if action in candidates and candidates:
                action_idx = candidates.index(action)
                q_val = self.agent.compute_q_values_train(state, candidates, action_idx)
                current_q_values.append(q_val)
            else:
                current_q_values.append(
                    torch.tensor(0.0, device=self.device, requires_grad=True)
                )

            n_return = trans.n_step_return

            if trans.done or trans.next_state is None:
                target = torch.tensor(n_return, device=self.device, dtype=torch.float32)
            else:
                next_state = trans.next_state
                next_candidates = next_state.get_available_actions()

                if next_candidates:
                    with torch.no_grad():
                        next_q_values, _ = self.agent.compute_q_values_target(
                            next_state, next_candidates
                        )
                        max_next_q = max(next_q_values)
                    bootstrap = (self.gamma ** self.n_step) * max_next_q
                    target = torch.tensor(
                        n_return + bootstrap, device=self.device, dtype=torch.float32
                    )
                else:
                    target = torch.tensor(n_return, device=self.device, dtype=torch.float32)

            target_values.append(target)

        current_q_tensor = torch.stack([q.squeeze() for q in current_q_values])
        target_tensor = torch.stack(target_values)

        return current_q_tensor, target_tensor

    def train_step(self) -> Optional[float]:
      
        if len(self.replay_memory) < self.batch_size:
            return None

        batch = self._sample_batch()

        current_q, target_q = self._compute_target(batch)

        loss = self.loss_fn(current_q.float(), target_q.float().detach())

        self.optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            list(self.agent.structure2vec.parameters()) +
            list(self.agent.q_network.parameters()),
            max_norm=1.0
        )

        self.optimizer.step()

        return loss.item()

    def run_episode(
        self,
        instance: GraphInstance,
        training: bool = True,
        update_lambda_flag: bool = True
    ) -> TrainingStats:
       
        self.env.set_instance(instance)
        state = self.env.reset()

        self.n_step_buffer.clear()

        episode_reward = 0.0
        step = 0
        done = False
        reached_destination = False

        while not done and step < self.max_steps:
            candidates = state.get_available_actions()

            if not candidates:
                break

            epsilon = self.epsilon if training else 0.0
            action_result = self.agent.select_action(state, epsilon=epsilon, candidates=candidates)
            action = action_result.action

            next_state, reward, done, info = self.env.step(action)

            episode_reward += reward

            self.n_step_buffer.add(state, action, reward, done)

            if done:
                while len(self.n_step_buffer) > 0:
                    n_result = self.n_step_buffer.get_n_step_return()
                    if n_result[0] is None:
                        break
                    store_state, store_action, n_return, store_next_state, store_done = n_result

                    transition = Transition(
                        state=store_state,
                        action=store_action,
                        reward=n_return,
                        next_state=store_next_state,
                        done=store_done,
                        n_step_return=n_return
                    )
                    self._store_transition(transition)
                    self.n_step_buffer.buffer.popleft()

                if next_state.current_node == instance.destination:
                    reached_destination = True
            elif len(self.n_step_buffer) >= self.n_step:
                n_result = self.n_step_buffer.get_n_step_return()
                if n_result[0] is not None:
                    store_state, store_action, n_return, store_next_state, store_done = n_result

                    transition = Transition(
                        state=store_state,
                        action=store_action,
                        reward=n_return,
                        next_state=store_next_state,
                        done=store_done,
                        n_step_return=n_return
                    )
                    self._store_transition(transition)
                    self.n_step_buffer.buffer.popleft()

            state = next_state
            step += 1

        loss = None
        if training and len(self.replay_memory) >= self.batch_size:
            loss = self.train_step()

            if self.episode_count % self.target_update_freq == 0:
                self.agent.hard_update_target_network()
            else:
                self.agent.update_target_network(tau=self.tau)

            self.epsilon = max(
                self.epsilon_min,
                self.epsilon * self.epsilon_decay
            )

        if (training
                and update_lambda_flag
                and reached_destination
                and state.T_max > 0.0):
            self.lambda_penalty = update_lambda(
                self.lambda_penalty,
                T_P=state.cumulative_time,
                T_max=state.T_max,
                eta_lambda=self.eta_lambda,
            )
            self.env.lambda_penalty = self.lambda_penalty

        path_feasible = (state.cumulative_time <= state.T_max) if reached_destination else False

        stats = TrainingStats(
            episode=self.episode_count,
            episode_reward=episode_reward,
            path_cost=state.cumulative_cost,
            path_time=state.cumulative_time,
            path_feasible=path_feasible,
            num_steps=step,
            epsilon=self.epsilon,
            loss=loss,
            lambda_value=self.lambda_penalty
        )

        self.training_stats.append(stats)
        self.episode_count += 1

        return stats

    def train(
        self,
        num_episodes: int,
        instances: Optional[List[GraphInstance]] = None,
        instance_generator: Optional[callable] = None,
        val_instances: Optional[List[GraphInstance]] = None,
        val_freq: int = 100,
        verbose: bool = True
    ) -> List[TrainingStats]:
     
        best_val_cost = float('inf')

        for episode in range(num_episodes):
            if instances and episode < len(instances):
                instance = instances[episode % len(instances)]
            elif instance_generator:
                instance = instance_generator()
            else:
                instance = create_instance(num_nodes=20, edge_prob=0.3, seed=episode)

            self.env.lambda_penalty = self.lambda_penalty

            stats = self.run_episode(instance, training=True)

            if val_instances and episode % val_freq == 0:
                val_stats = self.validate(val_instances)
                if verbose:
                    print(f"Episode {episode}: Train Cost={stats.path_cost:.2f}, "
                          f"Val Cost={val_stats['avg_cost']:.2f}, "
                          f"Val Feasibility={val_stats['feasibility_rate']:.2%}, "
                          f"ε={stats.epsilon:.4f}, Loss={stats.loss:.4f if stats.loss else 'N/A'}")

                if val_stats['avg_cost'] < best_val_cost and val_stats['feasibility_rate'] >= 0.5:
                    best_val_cost = val_stats['avg_cost']
            elif verbose and episode % 20 == 0:
                print(f"Episode {episode}: Cost={stats.path_cost:.2f}, "
                      f"Time={stats.path_time:.2f}/{stats.path_feasible}, "
                      f"Steps={stats.num_steps}, ε={stats.epsilon:.4f}")

        return self.training_stats

    def validate(self, instances: List[GraphInstance]) -> Dict[str, float]:
       
        costs = []
        times = []
        feasible_count = 0

        for instance in instances:
            self.env.set_instance(instance)
            state = self.env.reset()

            path, cost, time, success = self.agent.construct_path(state)

            costs.append(cost)
            times.append(time)

            if success and time <= instance.T_max:
                feasible_count += 1

        return {
            'avg_cost': np.mean(costs),
            'std_cost': np.std(costs),
            'avg_time': np.mean(times),
            'feasibility_rate': feasible_count / len(instances) if instances else 0.0
        }


def generate_training_instances(
    num_instances: int,
    num_nodes_range: Tuple[int, int] = (15, 30),
    edge_prob_range: Tuple[float, float] = (0.2, 0.4),
    seed: int = 42
) -> List[GraphInstance]:
  
    np.random.seed(seed)

    instances = []
    for i in range(num_instances):
        num_nodes = np.random.randint(num_nodes_range[0], num_nodes_range[1] + 1)
        edge_prob = np.random.uniform(edge_prob_range[0], edge_prob_range[1])

        instance = create_instance(
            num_nodes=num_nodes,
            edge_prob=edge_prob,
            seed=seed + i,
            ensure_path=True
        )
        instances.append(instance)

    return instances
