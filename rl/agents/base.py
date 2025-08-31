"""
Abstract base class for all Tetris agents.
Provides a common interface for different algorithmic approaches.
"""
from abc import ABC, abstractmethod
from typing import Any, Dict


class Agent(ABC):
    """
    Abstract base class for Tetris agents.
    All agents must implement select_action to choose piece placements.
    """
    
    @abstractmethod
    def select_action(self, env) -> tuple[int, int]:
        """
        Select the best action (column, rotation) for the current piece.
        
        Args:
            env: TetrisEnv instance with current game state
            
        Returns:
            Tuple of (column, rotation) for piece placement
        """
        pass
    
    def notify_step(self, prev_state, action, new_state, reward: float, info: Dict[str, Any]):
        """
        Optional hook for learning-based agents to receive step feedback.
        
        Args:
            prev_state: Previous board state
            action: Action taken (column, rotation)
            new_state: New board state after action
            reward: Reward received for the action
            info: Additional information dict
        """
        pass
    
    def reset(self):
        """
        Optional episodic reset for agents with internal state.
        Called at the beginning of each new episode.
        """
        pass
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get agent-specific statistics for UI display.
        
        Returns:
            Dictionary with agent state information
        """
        return {}