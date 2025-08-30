"""
Shared utility functions for search-based algorithms.
"""
import math
import random
from typing import List, Tuple, Dict


def softmax(xs: List[float]) -> List[float]:
    """
    Compute softmax probabilities from log-values.
    
    Args:
        xs: List of log-values
        
    Returns:
        List of probabilities that sum to 1
    """
    if not xs:
        return []
    
    # Subtract max for numerical stability
    m = max(xs)
    exps = [math.exp(x - m) for x in xs]
    s = sum(exps)
    
    if s == 0:
        return [1.0 / len(xs)] * len(xs)
    
    return [e / s for e in exps]


def argmin_with_index(vals: List[float]) -> Tuple[int, float]:
    """
    Find index and value of minimum element.
    
    Args:
        vals: List of values
        
    Returns:
        Tuple of (index, value) of minimum element
    """
    if not vals:
        return -1, float('inf')
    
    i = min(range(len(vals)), key=lambda k: vals[k])
    return i, vals[i]


def metropolis_accept(delta: float, temperature: float) -> bool:
    """
    Metropolis acceptance criterion for simulated annealing.
    
    Args:
        delta: Change in objective (new - current)
        temperature: Current temperature
        
    Returns:
        True if move should be accepted
    """
    if delta <= 0:
        return True  # Always accept improvements
    
    if temperature <= 1e-12:
        return False  # No randomness at zero temperature
    
    prob = math.exp(-delta / temperature)
    return random.random() < prob


def enumerate_actions(env):
    """
    Enumerate all legal actions (afterstates) for the current piece.
    
    Args:
        env: TetrisEnv instance
        
    Returns:
        Tuple of (actions, afterstates) where:
        - actions: List of (column, rotation) tuples
        - afterstates: List of board states after each action
    """
    try:
        from .afterstate import enumerate_afterstates
        afterstates_data = enumerate_afterstates(env)
        
        actions = []
        afterstates = []
        
        for afterstate_board, action_info in afterstates_data:
            actions.append((action_info['col'], action_info['rotation']))
            afterstates.append(afterstate_board)
            
        return actions, afterstates
        
    except ImportError:
        # Fallback for testing without full dependencies
        # Use center positions instead of left-biased defaults
        return [(3, 0), (4, 0), (5, 0)], [None, None, None]


def compute_cost_from_features(features: List[float], weights: Dict[str, float]) -> float:
    """
    Compute total cost from board features and weights.
    Uses a large penalty (1e6) for hard violations like game over.
    
    Args:
        features: List of feature values [holes, max_height, bumpiness, etc.]
        weights: Dict mapping feature names to weights
        
    Returns:
        Total cost (minimize)
    """
    if len(features) == 0:
        return 1e6  # Invalid state penalty
    
    # Standard feature names (must match feature extraction order)
    feature_names = [
        'w_holes', 'w_max_height', 'w_bumpiness', 
        'w_line_potential', 'w_aggregate_height'
    ]
    
    cost = 0.0
    for i, feature_val in enumerate(features):
        if i < len(feature_names):
            weight = weights.get(feature_names[i], 1.0)
            cost += weight * feature_val
    
    return cost


def score_afterstate(afterstate_board, **weights) -> float:
    """
    Score an afterstate using feature weights. 
    This is a simplified version that will work with the existing feature extraction.
    
    Args:
        afterstate_board: Board state after piece placement
        **weights: Feature weights as keyword arguments
        
    Returns:
        Cost score (lower is better)
    """
    try:
        # Try to use the full feature extraction if available
        from .features import board_to_features, holes, max_height, bumpiness, aggregate_height
        
        features_array = board_to_features(afterstate_board)
        
        # Extract key features for cost calculation
        if len(features_array) >= 17:  # Full feature vector
            # Use indices from board_to_features: holes(10), agg_height(11), bumpiness(12), max_height(14)
            holes_val = features_array[10] if len(features_array) > 10 else 0
            agg_height_val = features_array[11] if len(features_array) > 11 else 0
            bumpiness_val = features_array[12] if len(features_array) > 12 else 0
            max_height_val = features_array[14] if len(features_array) > 14 else 0
        else:
            # Fallback to direct computation
            holes_val = holes(afterstate_board) / 100.0  # Normalize
            agg_height_val = aggregate_height(afterstate_board) / 200.0
            bumpiness_val = bumpiness(afterstate_board) / 100.0
            max_height_val = max_height(afterstate_board) / 20.0
        
        # Compute weighted cost
        cost = (weights.get('w_holes', 1.0) * holes_val +
                weights.get('w_max_height', 1.0) * max_height_val +
                weights.get('w_bumpiness', 1.0) * bumpiness_val +
                weights.get('w_aggregate_height', 0.5) * agg_height_val)
        
        return cost
        
    except ImportError:
        # Fallback for when numpy/features are not available
        return random.uniform(0, 10)  # Random score for testing