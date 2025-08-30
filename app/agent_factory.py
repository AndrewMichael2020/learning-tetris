"""
Agent factory for creating different algorithm instances.
"""
from typing import Dict, Any
from .algo_schemas import ALGORITHM_PARAMS


def make_agent(algo: str, params: Dict[str, Any] = None):
    """
    Create an agent instance based on algorithm name and parameters.
    
    Args:
        algo: Algorithm name ('cem', 'reinforce', 'greedy', 'tabu', 'anneal', 'aco')
        params: Algorithm-specific parameters
        
    Returns:
        Agent instance
        
    Raises:
        ValueError: If algorithm is unknown or parameters are invalid
    """
    if params is None:
        params = {}
    
    # Handle existing algorithms that aren't converted to new agent system yet
    if algo in ["cem", "reinforce"]:
        # For now, return None and let existing code handle these
        return None
    
    # Create new algorithm agents
    if algo == "greedy":
        from rl.greedy_agent import GreedyAgent
        validated_params = ALGORITHM_PARAMS['greedy'](**params)
        return GreedyAgent(**validated_params.model_dump())
        
    elif algo == "tabu":
        from rl.tabu_agent import TabuAgent
        validated_params = ALGORITHM_PARAMS['tabu'](**params)
        return TabuAgent(**validated_params.model_dump())
        
    elif algo == "anneal":
        from rl.sa_agent import SimulatedAnnealingAgent
        validated_params = ALGORITHM_PARAMS['anneal'](**params)
        return SimulatedAnnealingAgent(**validated_params.model_dump())
        
    elif algo == "aco":
        from rl.aco_agent import ACOAgent
        validated_params = ALGORITHM_PARAMS['aco'](**params)
        return ACOAgent(**validated_params.model_dump())
    
    else:
        raise ValueError(f"Unknown algorithm: {algo}")


def get_default_params(algo: str) -> Dict[str, Any]:
    """
    Get default parameters for an algorithm.
    
    Args:
        algo: Algorithm name
        
    Returns:
        Dictionary of default parameters
    """
    if algo in ALGORITHM_PARAMS:
        param_class = ALGORITHM_PARAMS[algo]
        return param_class().model_dump()
    
    return {}