"""
Ant Colony Optimization Agent - "Night Shift Ant March"
Uses pheromone trails and heuristic information to guide action selection.
"""
import math
import random
from typing import List, Dict, Any
from .agents.base import Agent
from .utils.search import enumerate_actions, score_afterstate, softmax


class ACOAgent(Agent):
    """
    Ant Colony Optimization agent that maintains pheromone trails.
    Uses probabilistic action selection based on pheromone and heuristic information.
    """
    
    def __init__(self, alpha: float = 1.0, beta: float = 2.0, rho: float = 0.10, 
                 ants: int = 20, elite: int = 1, Q: float = 1.0, 
                 rng_seed: int = 42, **weights):
        """
        Initialize ACO agent.
        
        Args:
            alpha: Pheromone influence parameter
            beta: Heuristic influence parameter  
            rho: Pheromone evaporation rate
            ants: Number of ants per iteration
            elite: Number of elite ants for pheromone update
            Q: Pheromone deposit scaling factor
            rng_seed: Random seed
            **weights: Feature weights
        """
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.ants = ants
        self.elite = elite
        self.Q = Q
        self.weights = weights
        
        random.seed(rng_seed)
        
        self.pheromone: List[float] = []  # Pheromone levels for each action
        self.iteration_count = 0
        self.best_cost_ever = float('inf')
        self.pheromone_stats = {'min': 1.0, 'max': 1.0, 'mean': 1.0}
        
    def reset(self):
        """Reset agent state for new episode."""
        self.pheromone = []
        self.iteration_count = 0
        self.best_cost_ever = float('inf')
        self.pheromone_stats = {'min': 1.0, 'max': 1.0, 'mean': 1.0}
        
    def select_action(self, env) -> tuple[int, int]:
        """
        Select action using ACO probabilistic selection.
        
        Args:
            env: TetrisEnv instance
            
        Returns:
            Tuple of (column, rotation) for selected placement
        """
        actions, afterstates = enumerate_actions(env)
        
        if not actions:
            return 0, 0
        
        n_actions = len(actions)
        costs = [score_afterstate(afterstate, **self.weights) 
                for afterstate in afterstates]
        
        # Initialize or resize pheromone trails
        if len(self.pheromone) != n_actions:
            self.pheromone = [1.0] * n_actions
        
        # Heuristic information (inverse of cost)
        eta = [1.0 / (1.0 + cost) for cost in costs]
        
        # Generate solutions for all ants
        ant_solutions = []
        for ant in range(self.ants):
            # Build probability distribution
            tau_alpha = [math.pow(max(tau, 1e-10), self.alpha) for tau in self.pheromone]
            eta_beta = [math.pow(e, self.beta) for e in eta]
            
            # Combined attractiveness
            attractiveness = [tau_alpha[i] * eta_beta[i] for i in range(n_actions)]
            
            # Handle edge case of all zero attractiveness
            if sum(attractiveness) == 0:
                probs = [1.0 / n_actions] * n_actions
            else:
                total = sum(attractiveness)
                probs = [a / total for a in attractiveness]
            
            # Roulette wheel selection
            r = random.random()
            cumulative = 0.0
            selected_idx = 0
            
            for i, prob in enumerate(probs):
                cumulative += prob
                if r <= cumulative:
                    selected_idx = i
                    break
            
            ant_solutions.append((selected_idx, costs[selected_idx]))
        
        # Sort ants by solution quality (cost)
        ant_solutions.sort(key=lambda x: x[1])
        elite_ants = ant_solutions[:self.elite]
        
        # Pheromone evaporation
        self.pheromone = [(1.0 - self.rho) * tau for tau in self.pheromone]
        
        # Pheromone deposition by elite ants
        for ant_idx, ant_cost in elite_ants:
            deposit = self.Q / (1.0 + ant_cost)
            self.pheromone[ant_idx] += deposit
            
            # Track best solution
            if ant_cost < self.best_cost_ever:
                self.best_cost_ever = ant_cost
        
        # Update pheromone statistics
        if self.pheromone:
            self.pheromone_stats = {
                'min': min(self.pheromone),
                'max': max(self.pheromone),
                'mean': sum(self.pheromone) / len(self.pheromone)
            }
        
        self.iteration_count += 1
        
        # Return the best solution found by elite ants
        best_ant_idx, _ = elite_ants[0]
        return actions[best_ant_idx]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get current agent statistics."""
        return {
            'algorithm': 'Ant Colony Optimization (Night Shift Ant March)',
            'iteration_count': self.iteration_count,
            'best_cost': self.best_cost_ever,
            'pheromone_min': f"{self.pheromone_stats['min']:.3f}",
            'pheromone_max': f"{self.pheromone_stats['max']:.3f}",
            'pheromone_mean': f"{self.pheromone_stats['mean']:.3f}",
            'num_ants': self.ants,
            'elite_ants': self.elite
        }