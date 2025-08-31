#!/usr/bin/env python3
"""
Test script to demonstrate the distribution range fixes.
This shows how the improved CEM and REINFORCE agents now play more strategically.
"""
import numpy as np
from rl.tetris_env import TetrisEnv
from rl.cem_agent import CEMPolicy, evolve
from rl.reinforce_agent import REINFORCEPolicy, train
from rl.features import board_to_features


def test_placement_distribution(agent_name, policy, env, num_tests=10):
    """Test the distribution of placement choices."""
    env.reset(seed=42)
    placement_cols = []
    
    for _ in range(num_tests):
        if hasattr(policy, 'get_action_probabilities'):
            # REINFORCE policy
            rng = np.random.default_rng(42)
            afterstate_info, probs = policy.get_action_probabilities(env, rng)
            if len(afterstate_info) > 0:
                # Sample from probabilities
                action_idx = rng.choice(len(afterstate_info), p=probs)
                col = afterstate_info[action_idx]['col']
                placement_cols.append(col)
        else:
            # CEM policy
            col, rotation, score = policy.get_best_placement(env)
            placement_cols.append(col)
        
        # Reset for consistent testing
        env.reset(seed=42)
    
    if placement_cols:
        col_counts = np.bincount(placement_cols, minlength=env.width)
        print(f"{agent_name} placement distribution (columns 0-{env.width-1}):")
        for i, count in enumerate(col_counts):
            bar = '█' * int(count * 20 / max(col_counts)) if max(col_counts) > 0 else ''
            print(f"  Col {i}: {count:2d} {bar}")
        
        # Check for wall-hugging (too many placements in columns 0, 1, 8, 9)
        wall_cols = col_counts[0] + col_counts[1] + col_counts[-2] + col_counts[-1]
        center_cols = sum(col_counts[2:-2])
        wall_ratio = wall_cols / (wall_cols + center_cols) if (wall_cols + center_cols) > 0 else 0
        print(f"  Wall-hugging ratio: {wall_ratio:.2f} (lower is better)")
        return wall_ratio
    return 1.0


def main():
    print("Testing Distribution Range Fixes")
    print("=" * 50)
    
    # Create small environment for faster testing
    env = TetrisEnv(width=10, height=8)
    
    print("\n1. Testing CEM Agent (Quick Train)...")
    def env_factory():
        return TetrisEnv(width=10, height=8)
    
    # Train CEM with new distribution ranges
    cem_result = evolve(
        env_factory=env_factory,
        generations=3,
        seed=42,
        episodes_per_candidate=2,
        population_size=10,
        feature_dim=17,
        out_path="policies/test_cem.npz"
    )
    
    cem_policy = CEMPolicy(cem_result['best_weights'])
    cem_wall_ratio = test_placement_distribution("CEM", cem_policy, env)
    
    print("\n2. Testing REINFORCE Agent (Quick Train)...")
    
    # Train REINFORCE with new distribution ranges
    reinforce_result = train(
        env_factory=env_factory,
        episodes=20,
        seed=42,
        learning_rate=0.01,
        temperature=1.0,
        feature_dim=17,
        out_path="policies/test_reinforce.npz"
    )
    
    reinforce_policy = REINFORCEPolicy(reinforce_result['best_weights'], temperature=1.0)
    reinforce_wall_ratio = test_placement_distribution("REINFORCE", reinforce_policy, env)
    
    print("\n3. Performance Analysis:")
    print(f"CEM Best Fitness: {cem_result['best_fitness']:.1f}")
    print(f"REINFORCE Best Reward: {reinforce_result['best_reward']:.1f}")
    print(f"CEM Wall-Hugging: {cem_wall_ratio:.2f}")
    print(f"REINFORCE Wall-Hugging: {reinforce_wall_ratio:.2f}")
    
    print("\n4. Improvements Summary:")
    if cem_wall_ratio < 0.6:
        print("✓ CEM shows reduced wall-hugging behavior")
    else:
        print("⚠ CEM still shows some wall-hugging")
        
    if reinforce_wall_ratio < 0.6:
        print("✓ REINFORCE shows reduced wall-hugging behavior")
    else:
        print("⚠ REINFORCE still shows some wall-hugging")
    
    print(f"\nThe fixes applied:")
    print("- CEM: Wider initial exploration (std=1.2), candidate clipping, std bounds")
    print("- REINFORCE: Standardized logits, stable softmax, smaller weight init")
    print("- Both: Better numerical stability and exploration-exploitation balance")


if __name__ == "__main__":
    main()
