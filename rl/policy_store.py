"""
Policy storage utilities for saving and loading trained models.
"""
import numpy as np
import os
from typing import Dict, Any, Optional


def save_policy(weights: Dict[str, np.ndarray], path: str, 
                metadata: Optional[Dict[str, Any]] = None) -> None:
    """
    Save policy weights to .npz file.
    
    Args:
        weights: Dictionary of numpy arrays (policy parameters)
        path: File path to save to
        metadata: Optional metadata dict to include
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    # Prepare data for saving
    save_dict = weights.copy()
    
    # Add metadata as a special entry
    if metadata is not None:
        # Convert metadata to saveable format
        meta_arrays = {}
        for key, value in metadata.items():
            if isinstance(value, (int, float)):
                meta_arrays[f'meta_{key}'] = np.array([value])
            elif isinstance(value, str):
                meta_arrays[f'meta_{key}'] = np.array([value], dtype='U')
            elif isinstance(value, (list, tuple)):
                meta_arrays[f'meta_{key}'] = np.array(value)
            else:
                # Convert to string representation
                meta_arrays[f'meta_{key}'] = np.array([str(value)], dtype='U')
        
        save_dict.update(meta_arrays)
    
    # Save to file
    np.savez_compressed(path, **save_dict)


def load_policy(path: str) -> Dict[str, np.ndarray]:
    """
    Load policy weights from .npz file.
    
    Args:
        path: File path to load from
        
    Returns:
        Dictionary of numpy arrays (policy parameters)
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Policy file not found: {path}")
    
    # Load from file
    data = np.load(path)
    
    # Convert back to dict and separate metadata
    weights = {}
    metadata = {}
    
    for key in data.files:
        if key.startswith('meta_'):
            # This is metadata
            meta_key = key[5:]  # Remove 'meta_' prefix
            array = data[key]
            
            # Convert back to appropriate type
            if array.dtype.kind == 'U':  # Unicode string
                metadata[meta_key] = str(array[0])
            elif len(array) == 1:
                metadata[meta_key] = float(array[0]) if '.' in str(array[0]) else int(array[0])
            else:
                metadata[meta_key] = array.tolist()
        else:
            # This is a weight array
            weights[key] = data[key]
    
    # Add metadata as special key if it exists
    if metadata:
        weights['_metadata'] = metadata
    
    return weights


def create_random_policy(feature_dim: int, seed: Optional[int] = None) -> Dict[str, np.ndarray]:
    """
    Create a balanced heuristic policy instead of random.
    
    Args:
        feature_dim: Dimension of feature vector
        seed: Random seed (for compatibility, but we use heuristic weights)
        
    Returns:
        Dictionary with balanced heuristic policy weights
    """
    # Use balanced heuristic weights instead of random
    # This creates unbiased gameplay with good Tetris strategy
    if feature_dim == 17:
        weights = np.array([
            # Column heights - all zero (no bias toward any column)
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            # Strategic feature weights
            -0.8,  # holes (very bad, minimize)
            -0.2,  # agg_height (minimize total height)
            -0.3,  # bumpiness (bad, creates holes)
            -0.15, # wells (mildly bad)
            -0.4,  # max_height (bad, minimize max height)
            1.0,   # completed_lines (very good, maximize)
            -0.1   # row_transitions (mildly bad)
        ], dtype=np.float32)
    else:
        # Fallback to small random weights for other dimensions
        rng = np.random.default_rng(seed)
        weights = rng.normal(0, 0.05, size=feature_dim).astype(np.float32)
    
    return {
        'linear_weights': weights
    }


def ensure_policies_dir() -> str:
    """Ensure policies directory exists and return path."""
    policies_dir = "policies"
    os.makedirs(policies_dir, exist_ok=True)
    return policies_dir


def get_default_policy_path() -> str:
    """Get default policy save path."""
    return os.path.join(ensure_policies_dir(), "best.npz")


def load_or_create_policy(path: Optional[str] = None, 
                         feature_dim: int = 17,
                         seed: Optional[int] = None) -> Dict[str, np.ndarray]:
    """
    Load policy from file or create random one if file doesn't exist.
    
    Args:
        path: Policy file path (uses default if None)
        feature_dim: Feature dimension for random policy creation
        seed: Random seed for policy creation
        
    Returns:
        Dictionary with policy weights
    """
    if path is None:
        path = get_default_policy_path()
    
    try:
        return load_policy(path)
    except FileNotFoundError:
        # Create balanced heuristic policy
        policy = create_random_policy(feature_dim, seed)
        
        # Save it for future use
        metadata = {
            'created': 'balanced_heuristic',
            'feature_dim': feature_dim,
            'algorithm': 'heuristic',
            'version': '1.0'
        }
        save_policy(policy, path, metadata)
        
        return policy