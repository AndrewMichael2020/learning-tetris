"""
Pydantic schemas for API requests and responses.
"""
from pydantic import BaseModel, Field
from typing import Optional, List, Literal, Dict, Any


class PlayRequest(BaseModel):
    """Request schema for /api/play endpoint."""
    episodes: int = Field(default=1, ge=1, le=100, description="Number of episodes to play")
    seed: Optional[int] = Field(default=None, description="Random seed for reproducibility")
    algo: Literal["cem", "reinforce", "greedy", "tabu", "anneal", "aco"] = Field(default="cem", description="Algorithm to use")


class PlayResponse(BaseModel):
    """Response schema for /api/play endpoint."""
    total_lines: int = Field(description="Total lines cleared across all episodes")
    avg_score: float = Field(description="Average score across episodes")
    episodes: int = Field(description="Number of episodes played")
    scores: List[int] = Field(description="Individual episode scores")
    lines_cleared: List[int] = Field(description="Lines cleared per episode")
    episode_lengths: List[int] = Field(description="Episode lengths (steps)")


class TrainRequest(BaseModel):
    """Request schema for /api/train endpoint."""
    algo: Literal["cem", "reinforce", "greedy", "tabu", "anneal", "aco"] = Field(default="cem", description="Algorithm to train")
    seed: int = Field(default=42, description="Random seed")
    
    # CEM specific parameters
    generations: Optional[int] = Field(default=5, ge=1, le=10, description="Generations for CEM")
    population_size: Optional[int] = Field(default=20, ge=10, le=50, description="Population size for CEM")
    episodes_per_candidate: Optional[int] = Field(default=2, ge=1, le=5, description="Episodes per candidate for CEM")
    
    # REINFORCE specific parameters  
    episodes: Optional[int] = Field(default=200, ge=50, le=500, description="Episodes for REINFORCE")
    learning_rate: Optional[float] = Field(default=0.01, gt=0, le=1, description="Learning rate for REINFORCE")
    
    # New algorithm parameters (will be validated by specific schemas)
    params: Optional[Dict[str, Any]] = Field(default=None, description="Algorithm-specific parameters")


class TrainResponse(BaseModel):
    """Response schema for /api/train endpoint."""
    success: bool = Field(description="Whether training completed successfully")
    message: str = Field(description="Training completion message")
    algo: str = Field(description="Algorithm that was trained")
    best_performance: float = Field(description="Best performance achieved")
    training_time: float = Field(description="Training time in seconds")


class HealthResponse(BaseModel):
    """Response schema for /api/health endpoint."""
    status: Literal["ok"] = Field(default="ok", description="Health status")
    train_enabled: bool = Field(description="Whether training is enabled")
    policy_loaded: bool = Field(description="Whether a policy is loaded")


class StreamFrame(BaseModel):
    """Schema for WebSocket stream frames."""
    frame: str = Field(description="Base64-encoded PNG image")
    lines: int = Field(description="Lines cleared so far")
    score: float = Field(description="Current score")
    step: int = Field(description="Current step number")
    done: bool = Field(default=False, description="Whether episode is finished")