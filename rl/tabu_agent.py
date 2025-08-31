"""
Tabu Search Agent - "Nurse Gossip"
Uses tabu memory to avoid recently explored moves and aspiration criteria.
"""
from typing import Deque, List, Dict, Any, Tuple
from collections import deque
import random
from .agents.base import Agent
from .utils.search import enumerate_actions, score_afterstate


class TabuAgent(Agent):
    """
    Tabu search agent that maintains memory of recent moves to diversify search.
    Uses aspiration criteria to override tabu restrictions when improvements are found.
    """
    
    def __init__(self, tenure: int = 25, max_iters: int = 500, 
                 neighborhood_top_k: int = 10, aspiration: bool = True, 
                 rng_seed: int = 42, **weights):
        """
        Initialize tabu search agent.
        
        Args:
            tenure: Length of tabu list (how long moves stay forbidden)
            max_iters: Maximum iterations (unused in single-step Tetris)
            neighborhood_top_k: Number of best moves to consider
            aspiration: Whether to use aspiration criteria
            rng_seed: Random seed
            **weights: Feature weights
        """
        self.tenure = tenure
        self.max_iters = max_iters
        self.k = neighborhood_top_k
        self.aspiration = aspiration
        self.weights = weights
        
        random.seed(rng_seed)
        
        # Tabu memory stores action indices (not actual actions since they change each step)
        self._tabu: Deque[Tuple[int, int]] = deque(maxlen=self.tenure)  # (col, rotation) pairs
        self.best_cost = float("inf")
        self.current_cost = None
        self.iteration_count = 0
        
    def reset(self):
        """Reset agent state for new episode."""
        self._tabu.clear()
        self.best_cost = float("inf")
        self.current_cost = None
        self.iteration_count = 0
    
    def select_action(self, env) -> tuple[int, int]:
        """
        Select action using tabu search with aspiration criteria.
        
        Args:
            env: TetrisEnv instance
            
        Returns:
            Tuple of (column, rotation) for best non-tabu placement
        """
        actions, afterstates = enumerate_actions(env)
        
        if not actions:
            return 0, 0
        
        # Evaluate all afterstates
        costs = [score_afterstate(afterstate, **self.weights) 
                for afterstate in afterstates]
        
        # Sort by cost for neighborhood selection
        sorted_indices = sorted(range(len(costs)), key=lambda i: costs[i])
        top_k_indices = sorted_indices[:min(self.k, len(costs))]
        
        # Find best non-tabu move (or best overall if aspiration applies)
        best_candidate = None
        best_candidate_cost = float("inf")
        best_candidate_action = (0, 0)
        
        for i in top_k_indices:
            action = actions[i]
            cost = costs[i]
            
            # Check if this action is tabu
            is_tabu = action in self._tabu
            
            # Apply aspiration criteria: override tabu if better than best known
            aspirate = self.aspiration and cost < self.best_cost
            
            if not is_tabu or aspirate:
                if cost < best_candidate_cost:
                    best_candidate = i
                    best_candidate_cost = cost
                    best_candidate_action = action
        
        # If no non-tabu moves found, pick the best overall (emergency fallback)
        if best_candidate is None:
            best_candidate = sorted_indices[0]
            best_candidate_cost = costs[best_candidate]
            best_candidate_action = actions[best_candidate]
        
        # Update tabu memory and best solution
        self._tabu.append(best_candidate_action)
        
        if best_candidate_cost < self.best_cost:
            self.best_cost = best_candidate_cost
        
        self.current_cost = best_candidate_cost
        self.iteration_count += 1
        
        return best_candidate_action
    
    def get_stats(self) -> Dict[str, Any]:
        """Get current agent statistics."""
        return {
            'algorithm': 'Tabu Search (Nurse Gossip)', 
            'current_cost': self.current_cost,
            'best_cost': self.best_cost,
            'tabu_list_size': len(self._tabu),
            'iteration_count': self.iteration_count,
            'tabu_tenure': self.tenure
        }