"""
Consolidated utility functions for search-based Tetris algorithms.
Combines search utilities with afterstate functionality to eliminate redundancy.
"""
import math
import random
from typing import List, Tuple, Dict
import numpy as np


def softmax(xs: List[float]) -> List[float]:
    """Compute softmax probabilities from log-values."""
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
    """Find index and value of minimum element."""
    if not vals:
        return -1, float('inf')
    
    i = min(range(len(vals)), key=lambda k: vals[k])
    return i, vals[i]


def metropolis_accept(delta: float, temperature: float) -> bool:
    """Metropolis acceptance criterion for simulated annealing."""
    if delta <= 0:
        return True  # Always accept improvements
    
    if temperature <= 1e-12:
        return False  # No randomness at zero temperature
    
    prob = math.exp(-delta / temperature)
    return random.random() < prob


def enumerate_afterstates(env):
    """
    Enumerate all legal actions and their resulting board states.
    Consolidated from afterstate.py to eliminate redundancy.
    
    Args:
        env: TetrisEnv instance
        
    Returns:
        List of (afterstate_board, action_info) tuples
    """
    afterstates = []
    
    if not hasattr(env, 'current_piece') or env.current_piece is None:
        return afterstates
    
    if not hasattr(env, 'current_piece_name') or env.current_piece_name is None:
        return afterstates
    
    from ..tetris_env import TETRIS_PIECES
    
    # Try all rotations (0-3)
    for rotation in range(4):
        # Get piece shape for this rotation using pre-rotated pieces
        piece_matrix = TETRIS_PIECES[env.current_piece_name][rotation]
        
        # Find the piece bounds to calculate valid positions
        piece_rows, piece_cols = np.where(piece_matrix > 0)
        if len(piece_rows) == 0:  # Empty piece, skip
            continue
            
        piece_min_col = np.min(piece_cols)
        piece_max_col = np.max(piece_cols)
        piece_width = piece_max_col - piece_min_col + 1
        
        # Calculate valid column range
        min_col = -piece_min_col  # Allow piece to align its leftmost part
        max_col = env.board.shape[1] - piece_max_col - 1
        
        # Try all valid column positions
        for col in range(min_col, max_col + 1):
            # Check if placement is valid
            env_copy = env.copy()
            action = (col, rotation)
            
            try:
                # Test if this action is valid
                env_copy.step(action)
                if not env_copy.game_over:
                    afterstate_info = {
                        'col': col,
                        'rotation': rotation,
                        'lines_cleared': env_copy.lines_cleared - env.lines_cleared,
                        'reward': env_copy.score - env.score
                    }
                    afterstates.append((env_copy.board.copy(), afterstate_info))
            except:
                # Invalid placement, skip
                continue
    
    return afterstates


def score_afterstate_features(env, action, weights=None):
    """Score action using full feature vector."""
    if weights is None:
        # Default feature weights
        weights = np.array([
            -0.510066, 0.760666, -0.35663, -0.184483,
            -0.5, -0.3, -0.2, -0.1,
            -0.4, -0.6, -0.8, -0.9,
            -0.1, -0.2, -0.3, -0.4, -0.5
        ])
    
    try:
        from .features import compute_features
        env_copy = env.copy()
        env_copy.step(action)
        features = compute_features(env_copy.board)
        return np.dot(features, weights)
    except ImportError:
        return score_afterstate_simple(env, action)


def score_afterstate_simple(env, action):
    """Score action using simple hole-based heuristic."""
    env_copy = env.copy()
    env_copy.step(action)
    
    # Use simple hole-based scoring
    board = env_copy.board
    holes = count_holes(board)
    
    # More holes = worse score
    return -holes


def count_holes(board):
    """Count holes in the board (empty cells below filled cells)."""
    holes = 0
    for col in range(board.shape[1]):
        for row in range(board.shape[0]):
            if board[row, col] == 0:
                # Check if there's a filled cell above
                if np.any(board[:row, col] > 0):
                    holes += 1
    return holes


# Legacy compatibility functions
def enumerate_actions(env):
    """Legacy function - use enumerate_afterstates instead."""
    afterstates_data = enumerate_afterstates(env)
    
    actions = []
    afterstates = []
    
    for afterstate_board, action_info in afterstates_data:
        actions.append((action_info['col'], action_info['rotation']))
        afterstates.append(afterstate_board)
        
    return actions, afterstates


def score_afterstate(afterstate_board, **weights) -> float:
    """
    Legacy compatibility function for scoring afterstates.
    Use score_afterstate_features or score_afterstate_simple instead.
    """
    try:
        from .features import board_to_features
        
        features_array = board_to_features(afterstate_board)
        
        if len(features_array) >= 17:
            holes_val = features_array[10] if len(features_array) > 10 else 0
            agg_height_val = features_array[11] if len(features_array) > 11 else 0
            bumpiness_val = features_array[12] if len(features_array) > 12 else 0
            max_height_val = features_array[14] if len(features_array) > 14 else 0
        else:
            holes_val = count_holes(afterstate_board) / 100.0
            agg_height_val = np.sum(np.max(np.where(afterstate_board > 0, 20 - np.arange(20)[:, None], 0), axis=0)) / 200.0
            max_height_val = np.max(np.sum(afterstate_board > 0, axis=0)) / 20.0
            bumpiness_val = 0  # Simplified
        
        cost = (weights.get('w_holes', 1.0) * holes_val +
                weights.get('w_max_height', 1.0) * max_height_val +
                weights.get('w_bumpiness', 1.0) * bumpiness_val +
                weights.get('w_aggregate_height', 0.5) * agg_height_val)
        
        return cost
        
    except ImportError:
        return random.uniform(0, 10)