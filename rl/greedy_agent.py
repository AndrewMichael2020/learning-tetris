"""
Greedy Heuristic Agent - "Nurse Dictator"
Simply picks the action with the lowest cost according to feature weights.
"""
from typing import Dict, Any
from .agent_base import Agent
from .search_utils import enumerate_actions, score_afterstate, argmin_with_index


class GreedyAgent(Agent):
    """
    Greedy heuristic agent that always picks the action with minimum cost.
    No lookahead, no learning - just direct greedy evaluation.
    """
    
    def __init__(self, **weights):
        """
        Initialize greedy agent with feature weights.
        
        Args:
            **weights: Feature weights (w_holes, w_max_height, etc.)
        """
        self.weights = weights
        self.last_cost = None
        
    def select_action(self, env) -> tuple[int, int]:
        """
        Select action with minimum cost according to feature evaluation.
        
        Args:
            env: TetrisEnv instance
            
        Returns:
            Tuple of (column, rotation) for best placement
        """
        actions, afterstates = enumerate_actions(env)
        
        if not actions:
            return 0, 0  # Default action if no valid moves
        
        # Evaluate all afterstates
        costs = [score_afterstate(afterstate, **self.weights) 
                for afterstate in afterstates]
        
        # Find minimum cost action
        best_idx, best_cost = argmin_with_index(costs)
        self.last_cost = best_cost
        
        return actions[best_idx]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get current agent statistics."""
        return {
            'algorithm': 'Greedy (Nurse Dictator)',
            'current_cost': self.last_cost,
            'weights': dict(self.weights)
        }