"""
Evaluation CLI for trained Tetris policies.
"""
import argparse
import numpy as np
import json
import os
from typing import List, Dict, Any
from .tetris_env import TetrisEnv
from .policy_store import load_policy
from .afterstate import get_best_placement, execute_placement


def evaluate_policy(policy_path: str, episodes: int = 10, seed: int = 42,
                   render_best: bool = False, out_dir: str = None) -> Dict[str, Any]:
    """
    Evaluate a trained policy.
    
    Args:
        policy_path: Path to policy file
        episodes: Number of episodes to evaluate
        seed: Random seed
        render_best: Whether to render the best episode
        out_dir: Output directory for rendered frames
        
    Returns:
        Dictionary with evaluation metrics
    """
    # Load policy
    policy_dict = load_policy(policy_path)
    weights = policy_dict['linear_weights']
    
    rng = np.random.default_rng(seed)
    
    # Run episodes
    episode_scores = []
    episode_lines = []
    episode_lengths = []
    best_episode_score = -1
    best_episode_frames = []
    
    print(f"Evaluating policy: {policy_path}")
    print(f"Policy weights shape: {weights.shape}")
    
    if '_metadata' in policy_dict:
        print(f"Policy metadata: {policy_dict['_metadata']}")
    
    for episode in range(episodes):
        env = TetrisEnv()
        env.reset(seed=rng.integers(0, 1000000))
        
        episode_score = 0
        episode_steps = 0
        max_steps = 1000
        frames = []
        
        while not env.game_over and episode_steps < max_steps:
            # Render frame if needed
            if render_best:
                frame = env.render(mode="rgb_array")
                frames.append(frame)
            
            # Get best placement
            best_col, best_rotation, best_score = get_best_placement(env, weights)
            
            # Execute placement
            prev_score = env.score
            success = execute_placement(env, best_col, best_rotation)
            
            if not success:
                break
            
            # Track score
            episode_score = env.score
            episode_steps += 1
            
            # Spawn new piece
            if not env.game_over:
                env._spawn_piece()
        
        episode_scores.append(episode_score)
        episode_lines.append(env.lines_cleared)
        episode_lengths.append(episode_steps)
        
        # Track best episode for rendering
        if render_best and episode_score > best_episode_score:
            best_episode_score = episode_score
            best_episode_frames = frames
        
        print(f"Episode {episode+1}: Score={episode_score}, Lines={env.lines_cleared}, Steps={episode_steps}")
    
    # Calculate statistics
    metrics = {
        'episodes': episodes,
        'mean_score': float(np.mean(episode_scores)),
        'std_score': float(np.std(episode_scores)),
        'max_score': float(np.max(episode_scores)),
        'min_score': float(np.min(episode_scores)),
        'mean_lines': float(np.mean(episode_lines)),
        'std_lines': float(np.std(episode_lines)),
        'max_lines': int(np.max(episode_lines)),
        'mean_length': float(np.mean(episode_lengths)),
        'episode_scores': [float(s) for s in episode_scores],
        'episode_lines': [int(l) for l in episode_lines],
        'episode_lengths': [int(l) for l in episode_lengths]
    }
    
    # Save best episode frames if requested
    if render_best and out_dir and best_episode_frames:
        os.makedirs(out_dir, exist_ok=True)
        
        # Save frames as PNG files (would need PIL/imageio for actual PNG saving)
        # For now, save as numpy arrays
        for i, frame in enumerate(best_episode_frames[:100]):  # Limit to first 100 frames
            frame_path = os.path.join(out_dir, f'frame_{i:03d}.npy')
            np.save(frame_path, frame)
        
        print(f"Saved {len(best_episode_frames[:100])} frames to {out_dir}")
    
    return metrics


def main():
    """Main evaluation CLI."""
    parser = argparse.ArgumentParser(description="Evaluate trained Tetris policies")
    
    parser.add_argument('--policy', type=str, default='policies/best.npz',
                       help='Path to policy file')
    parser.add_argument('--episodes', type=int, default=10,
                       help='Number of episodes to evaluate')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--render', action='store_true',
                       help='Render best episode frames')
    parser.add_argument('--out-dir', type=str, default='out',
                       help='Output directory for rendered frames')
    parser.add_argument('--save-metrics', action='store_true',
                       help='Save evaluation metrics to JSON')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.policy):
        print(f"Error: Policy file not found: {args.policy}")
        return
    
    # Run evaluation
    metrics = evaluate_policy(
        policy_path=args.policy,
        episodes=args.episodes,
        seed=args.seed,
        render_best=args.render,
        out_dir=args.out_dir if args.render else None
    )
    
    # Print summary
    print(f"\nEvaluation Results ({args.episodes} episodes):")
    print(f"Mean Score: {metrics['mean_score']:.1f} ± {metrics['std_score']:.1f}")
    print(f"Max Score: {metrics['max_score']:.1f}")
    print(f"Mean Lines: {metrics['mean_lines']:.1f} ± {metrics['std_lines']:.1f}")
    print(f"Max Lines: {metrics['max_lines']}")
    print(f"Mean Episode Length: {metrics['mean_length']:.1f}")
    
    # Save metrics if requested
    if args.save_metrics:
        metrics_path = args.policy.replace('.npz', '_eval_metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"Saved evaluation metrics to {metrics_path}")


if __name__ == '__main__':
    main()