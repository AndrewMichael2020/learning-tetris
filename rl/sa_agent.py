"""
Simulated Annealing Agent - "Coffee Break"  
Uses Metropolis acceptance criterion with cooling temperature.
"""
import random
import statistics
from typing import Dict, Any, Optional
from .agents.base import Agent
from .utils.search import enumerate_actions, score_afterstate, metropolis_accept


class SimulatedAnnealingAgent(Agent):
    """
    Simulated annealing agent that accepts worse moves with decreasing probability.
    Uses Metropolis acceptance criterion and exponential cooling schedule.
    """
    
    def __init__(self, T0: Optional[float] = None, alpha: float = 0.99, 
                 steps_per_T: int = 500, proposal_top_k: int = 10, 
                 rng_seed: int = 42, **weights):
        """
        Initialize simulated annealing agent.
        
        Args:
            T0: Initial temperature (None = auto-calculate)
            alpha: Cooling rate (T_new = alpha * T_old)
            steps_per_T: Steps per temperature level (unused in Tetris)
            proposal_top_k: Number of top moves to sample from
            rng_seed: Random seed
            **weights: Feature weights
        """
        self.T0 = T0
        self.alpha = alpha
        self.steps_per_T = steps_per_T
        self.k = proposal_top_k
        self.weights = weights
        
        random.seed(rng_seed)
        
        self.T = None  # Current temperature
        self.prev_cost = None
        self.step_count = 0
        self.acceptance_count = 0
        self.total_proposals = 0
        
    def reset(self):
        """Reset agent state for new episode."""
        self.T = self.T0
        self.prev_cost = None
        self.step_count = 0
        self.acceptance_count = 0
        self.total_proposals = 0
        
    def select_action(self, env) -> tuple[int, int]:
        """
        Select action using simulated annealing with Metropolis acceptance.
        
        Args:
            env: TetrisEnv instance
            
        Returns:
            Tuple of (column, rotation) for selected placement
        """
        actions, afterstates = enumerate_actions(env)
        
        if not actions:
            return 0, 0
            
        # Evaluate all afterstates
        costs = [score_afterstate(afterstate, **self.weights) 
                for afterstate in afterstates]
        
        # Sort by cost to get neighborhood
        sorted_indices = sorted(range(len(costs)), key=lambda i: costs[i])
        top_k_indices = sorted_indices[:min(self.k, len(costs))]
        
        # Auto-calculate initial temperature if not set
        if self.T is None:
            if len(costs) > 1:
                cost_sample = [costs[i] for i in top_k_indices]
                cost_range = max(cost_sample) - min(cost_sample)
                self.T = self.T0 if self.T0 is not None else max(10.0 * cost_range, 1.0)
            else:
                self.T = self.T0 if self.T0 is not None else 1.0
        
        # Propose move: select randomly from top-k
        proposal_idx = random.choice(top_k_indices)
        proposal_cost = costs[proposal_idx]
        proposal_action = actions[proposal_idx]
        
        # Current best as fallback
        best_idx = sorted_indices[0]
        current_cost = costs[best_idx] if self.prev_cost is None else self.prev_cost
        
        # Metropolis acceptance
        delta = proposal_cost - current_cost
        accept = metropolis_accept(delta, self.T)
        
        self.total_proposals += 1
        
        if accept:
            chosen_idx = proposal_idx
            chosen_cost = proposal_cost
            chosen_action = proposal_action
            self.acceptance_count += 1
        else:
            chosen_idx = best_idx
            chosen_cost = costs[best_idx] 
            chosen_action = actions[best_idx]
        
        # Update state
        self.prev_cost = chosen_cost
        self.step_count += 1
        
        # Cool temperature
        self.T *= self.alpha
        self.T = max(self.T, 1e-6)  # Prevent temperature from going to zero
        
        return chosen_action
    
    def get_stats(self) -> Dict[str, Any]:
        """Get current agent statistics."""
        acceptance_rate = (self.acceptance_count / max(1, self.total_proposals)) * 100
        
        return {
            'algorithm': 'Simulated Annealing (Coffee Break)',
            'current_cost': self.prev_cost,
            'temperature': self.T,
            'step_count': self.step_count,
            'acceptance_rate': f"{acceptance_rate:.1f}%",
            'cooling_rate': self.alpha
        }