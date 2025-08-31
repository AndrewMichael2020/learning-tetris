"""
Training CLI for Tetris RL agents. -
"""
import argparse
import json
import os
from typing import Dict, Any
from .tetris_env import TetrisEnv
from .cem_agent import evolve as cem_evolve
from .reinforce_agent import train as reinforce_train


def create_env_factory():
    """Create factory function for TetrisEnv instances."""
    def factory():
        return TetrisEnv(width=10, height=20)
    return factory


def save_metrics(metrics: Dict[str, Any], out_path: str) -> None:
    """Save training metrics to JSON file."""
    metrics_path = out_path.replace('.npz', '_metrics.json')
    
    # Convert numpy arrays to lists for JSON serialization
    json_metrics = {}
    for key, value in metrics.items():
        if hasattr(value, 'tolist'):  # numpy array
            json_metrics[key] = value.tolist()
        elif isinstance(value, list) and len(value) > 0 and hasattr(value[0], 'tolist'):
            json_metrics[key] = [v.tolist() if hasattr(v, 'tolist') else v for v in value]
        else:
            json_metrics[key] = value
    
    with open(metrics_path, 'w') as f:
        json.dump(json_metrics, f, indent=2)
    
    print(f"Saved metrics to {metrics_path}")


def main():
    """Main training CLI."""
    parser = argparse.ArgumentParser(description="Train Tetris RL agents")
    
    parser.add_argument('--algo', type=str, choices=['cem', 'reinforce'], 
                       default='cem', help='Algorithm to use')
    parser.add_argument('--generations', type=int, default=10,
                       help='Number of generations (CEM only)')
    parser.add_argument('--episodes', type=int, default=1000,
                       help='Number of episodes (REINFORCE only)')
    parser.add_argument('--episodes-per-candidate', type=int, default=3,
                       help='Episodes per candidate (CEM only)')
    parser.add_argument('--population-size', type=int, default=50,
                       help='Population size (CEM only)')
    parser.add_argument('--learning-rate', type=float, default=0.01,
                       help='Learning rate (REINFORCE only)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--out', type=str, default='policies/best.npz',
                       help='Output path for trained policy')
    
    args = parser.parse_args()
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    
    # Create environment factory
    env_factory = create_env_factory()
    
    print(f"Training {args.algo.upper()} agent with seed {args.seed}")
    print(f"Output: {args.out}")
    
    if args.algo == 'cem':
        metrics = cem_evolve(
            env_factory=env_factory,
            generations=args.generations,
            seed=args.seed,
            out_path=args.out,
            episodes_per_candidate=args.episodes_per_candidate,
            population_size=args.population_size
        )
        
        print(f"\nCEM Training completed!")
        print(f"Best fitness: {metrics['best_fitness']:.1f}")
        
    elif args.algo == 'reinforce':
        metrics = reinforce_train(
            env_factory=env_factory,
            episodes=args.episodes,
            seed=args.seed,
            out_path=args.out,
            learning_rate=args.learning_rate
        )
        
        print(f"\nREINFORCE Training completed!")
        print(f"Best reward: {metrics['best_reward']:.1f}")
        print(f"Final baseline: {metrics['final_baseline']:.1f}")
    
    # Save metrics
    save_metrics(metrics, args.out)
    
    print("Training completed successfully!")


if __name__ == '__main__':
    main()