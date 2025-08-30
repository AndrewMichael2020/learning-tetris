"""
Configuration management for the Tetris app.
"""
import os
from typing import Optional


class Config:
    """Application configuration."""
    
    def __init__(self):
        self.port: int = int(os.getenv('PORT', '8080'))
        self.train_enabled: bool = os.getenv('TRAIN_ENABLED', 'false').lower() == 'true'
        self.demo_seed: Optional[int] = self._parse_int_env('DEMO_SEED')
        self.policy_path: str = os.getenv('POLICY_PATH', 'policies/best.npz')
        self.max_episodes: int = int(os.getenv('MAX_EPISODES', '100'))
        self.max_generations: int = int(os.getenv('MAX_GENERATIONS', '5'))
        self.stream_fps: int = int(os.getenv('STREAM_FPS', '10'))
    
    def _parse_int_env(self, var_name: str) -> Optional[int]:
        """Parse integer environment variable, return None if not set or invalid."""
        value = os.getenv(var_name)
        if value is None:
            return None
        try:
            return int(value)
        except ValueError:
            return None


# Global config instance
config = Config()