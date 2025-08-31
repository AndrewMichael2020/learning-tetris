"""
Agent package for different Tetris playing strategies.
Provides clean separation between learning agents and heuristic agents.
"""

from .base import Agent
from .learning import REINFORCEAgent, CEMAgent
from .heuristic import GreedyAgent, TabuAgent, SimulatedAnnealingAgent, ACOAgent

__all__ = [
    'Agent',
    'REINFORCEAgent', 'CEMAgent',
    'GreedyAgent', 'TabuAgent', 'SimulatedAnnealingAgent', 'ACOAgent'
]
