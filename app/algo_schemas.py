"""
Pydantic schemas for new algorithm parameters.
"""
from pydantic import BaseModel, Field
from typing import Optional


class BaseWeights(BaseModel):
    """Base class for feature weights common to all algorithms."""
    w_holes: float = Field(default=8.0, description="Weight for holes penalty")
    w_max_height: float = Field(default=1.0, description="Weight for max height penalty")
    w_bumpiness: float = Field(default=1.0, description="Weight for bumpiness penalty")
    w_line_potential: float = Field(default=2.0, description="Weight for line clearing potential")
    w_aggregate_height: float = Field(default=0.5, description="Weight for aggregate height penalty")


class GreedyParams(BaseWeights):
    """Parameters for Greedy Heuristic agent."""
    pass  # Only uses the base weights


class TabuParams(BaseWeights):
    """Parameters for Tabu Search agent."""
    tenure: int = Field(default=25, ge=5, le=100, description="Tabu list tenure (memory length)")
    max_iters: int = Field(default=500, ge=100, le=1000, description="Maximum iterations (unused in Tetris)")
    neighborhood_top_k: int = Field(default=10, ge=3, le=20, description="Neighborhood size for exploration")
    aspiration: bool = Field(default=True, description="Enable aspiration criteria")
    rng_seed: int = Field(default=42, description="Random seed")


class AnnealParams(BaseWeights):
    """Parameters for Simulated Annealing agent."""
    T0: Optional[float] = Field(default=None, ge=0.1, description="Initial temperature (None = auto)")
    alpha: float = Field(default=0.99, gt=0.5, lt=1.0, description="Cooling rate")
    steps_per_T: int = Field(default=500, ge=100, le=1000, description="Steps per temperature (unused)")
    proposal_top_k: int = Field(default=10, ge=3, le=20, description="Proposal neighborhood size")
    rng_seed: int = Field(default=42, description="Random seed")


class ACOParams(BaseWeights):
    """Parameters for Ant Colony Optimization agent."""
    alpha: float = Field(default=1.0, ge=0.1, le=5.0, description="Pheromone influence")
    beta: float = Field(default=2.0, ge=0.1, le=5.0, description="Heuristic influence")
    rho: float = Field(default=0.10, gt=0.01, lt=0.5, description="Evaporation rate")
    ants: int = Field(default=20, ge=5, le=50, description="Number of ants per iteration")
    elite: int = Field(default=1, ge=1, le=10, description="Number of elite ants")
    Q: float = Field(default=1.0, gt=0.1, le=10.0, description="Pheromone deposit scaling")
    rng_seed: int = Field(default=42, description="Random seed")


# Parameter mapping for easy access
ALGORITHM_PARAMS = {
    'greedy': GreedyParams,
    'tabu': TabuParams, 
    'anneal': AnnealParams,
    'aco': ACOParams
}