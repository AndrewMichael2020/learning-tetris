"""
REINFORCE agent with baseline for Tetris.
Uses stochastic policy with softmax over linear feature preferences.
"""
import numpy as np
from typing import Callable, Dict, List, Any, Optional, Tuple
from .tetris_env import TetrisEnv
from .features import board_to_features
from .afterstate import enumerate_afterstates
from .policy_store import save_policy


class REINFORCEPolicy:
    """Stochastic policy using softmax over linear feature weights."""
    
    def __init__(self, weights: np.ndarray, temperature: float = 1.0):
        """
        Initialize REINFORCE policy.
        
        Args:
            weights: Linear weights for features
            temperature: Softmax temperature for exploration
        """
        self.weights = weights.copy()
        self.temperature = temperature
        
    def get_action_probabilities(self, env: TetrisEnv, rng: np.random.Generator) -> Tuple[List, np.ndarray]:
        """
        Get action probabilities for current state using afterstate evaluation.
        
        Returns:
            (afterstates_info, probabilities) where probabilities[i] corresponds to afterstates_info[i]
        """
        afterstates = enumerate_afterstates(env)
        
        if not afterstates:
            return [], np.array([])
        
        # Calculate scores for each afterstate
        scores = []
        afterstate_info = []
        
        for afterstate_board, action_info in afterstates:
            features = board_to_features(afterstate_board)
            score = np.dot(self.weights, features) + action_info['reward']
            scores.append(score)
            afterstate_info.append(action_info)
        
        # Convert to softmax probabilities
        scores = np.array(scores)
        exp_scores = np.exp(scores / self.temperature)
        probabilities = exp_scores / np.sum(exp_scores)
        
        return afterstate_info, probabilities
    
    def predict(self, board: np.ndarray, legal_actions: List[int]) -> int:
        """Predict action (deterministic version for evaluation)."""
        # For evaluation, just use the highest scoring afterstate
        env = TetrisEnv()
        env.board = board.copy()
        env.game_over = False
        
        # This is a simplified version - in practice we'd need full env state
        # For now, return first legal action
        return legal_actions[0] if legal_actions else 5


def collect_episode(policy: REINFORCEPolicy, env: TetrisEnv, rng: np.random.Generator,
                   max_steps: int = 1000) -> Tuple[List[Dict], float]:
    """
    Collect one episode using stochastic policy.
    
    Returns:
        (episode_data, total_reward) where episode_data contains step information
    """
    episode_data = []
    env.reset(seed=rng.integers(0, 1000000))
    
    total_reward = 0.0
    steps = 0
    
    while not env.game_over and steps < max_steps:
        # Get state features (before action)
        state_features = board_to_features(env.board)
        
        # Get action probabilities using afterstates
        afterstate_info, action_probs = policy.get_action_probabilities(env, rng)
        
        if len(afterstate_info) == 0:
            break  # No valid actions
        
        # Sample action according to probabilities
        action_idx = rng.choice(len(action_probs), p=action_probs)
        chosen_action = afterstate_info[action_idx]
        
        # Execute action and get reward
        prev_score = env.score
        prev_lines = env.lines_cleared
        
        # Execute the placement
        from .afterstate import execute_placement
        success = execute_placement(env, chosen_action['col'], chosen_action['rotation'])
        
        if not success:
            break
        
        # Calculate reward
        reward = (env.score - prev_score) + (env.lines_cleared - prev_lines) * 10
        reward += chosen_action['reward']  # Include shaping reward
        
        total_reward += reward
        
        # Store step data
        step_data = {
            'state_features': state_features,
            'action_idx': action_idx,
            'action_probs': action_probs.copy(),
            'afterstate_info': afterstate_info,
            'reward': reward,
            'log_prob': np.log(action_probs[action_idx] + 1e-8)
        }
        episode_data.append(step_data)
        
        steps += 1
        
        # Spawn new piece for next iteration
        if not env.game_over:
            env._spawn_piece()
    
    return episode_data, total_reward


def train(env_factory: Callable[[], TetrisEnv], episodes: int = 1000,
          seed: int = 42, out_path: str = "policies/best.npz",
          learning_rate: float = 0.01, baseline_alpha: float = 0.1,
          temperature: float = 1.0, feature_dim: int = 17) -> Dict[str, Any]:
    """
    Train REINFORCE policy with baseline.
    
    Args:
        env_factory: Function that creates TetrisEnv instances
        episodes: Number of episodes to train
        seed: Random seed
        out_path: Path to save best policy
        learning_rate: Learning rate for policy updates
        baseline_alpha: Learning rate for baseline (EMA of returns)
        temperature: Softmax temperature
        feature_dim: Dimension of feature vector
        
    Returns:
        Dictionary with training metrics
    """
    rng = np.random.default_rng(seed)
    
    # Initialize policy with small random weights
    weights = rng.normal(0, 0.1, size=feature_dim).astype(np.float32)
    policy = REINFORCEPolicy(weights, temperature)
    
    # Initialize baseline (exponential moving average of returns)
    baseline = 0.0
    
    # Training metrics
    episode_rewards = []
    episode_lengths = []
    baseline_history = []
    best_reward = float('-inf')
    best_weights = None
    
    print(f"Starting REINFORCE training: {episodes} episodes")
    
    for episode in range(episodes):
        env = env_factory()
        
        # Collect episode
        episode_data, total_reward = collect_episode(policy, env, rng)
        
        if len(episode_data) == 0:
            continue  # Skip empty episodes
        
        # Update baseline (exponential moving average)
        if episode == 0:
            baseline = total_reward
        else:
            baseline = (1 - baseline_alpha) * baseline + baseline_alpha * total_reward
        
        # Calculate advantages and policy gradient
        advantage = total_reward - baseline
        
        # Policy gradient update
        gradient = np.zeros_like(weights)
        
        for step_data in episode_data:
            # Calculate gradient for this step
            state_features = step_data['state_features']
            log_prob = step_data['log_prob']
            
            # REINFORCE gradient: advantage * grad log π(a|s)
            # For softmax policy: grad log π(a|s) = features * (1 - π(a|s)) for chosen action
            action_idx = step_data['action_idx']
            action_probs = step_data['action_probs']
            
            # Simplified gradient calculation
            gradient += advantage * state_features * log_prob
        
        # Apply gradient update
        weights += learning_rate * gradient / len(episode_data)
        policy.weights = weights.copy()
        
        # Track metrics
        episode_rewards.append(total_reward)
        episode_lengths.append(len(episode_data))
        baseline_history.append(baseline)
        
        # Track best policy
        if total_reward > best_reward:
            best_reward = total_reward
            best_weights = weights.copy()
        
        # Print progress
        if (episode + 1) % 100 == 0 or episode == 0:
            recent_rewards = episode_rewards[-100:]
            mean_reward = np.mean(recent_rewards)
            print(f"Episode {episode+1}/{episodes}: "
                  f"Reward={total_reward:.1f}, "
                  f"Mean(100)={mean_reward:.1f}, "
                  f"Baseline={baseline:.1f}")
    
    # Save best policy
    if best_weights is not None:
        policy_dict = {'linear_weights': best_weights}
        metadata = {
            'algorithm': 'REINFORCE',
            'episodes': episodes,
            'learning_rate': learning_rate,
            'baseline_alpha': baseline_alpha,
            'temperature': temperature,
            'best_reward': float(best_reward),
            'final_baseline': float(baseline),
            'seed': seed
        }
        save_policy(policy_dict, out_path, metadata)
        print(f"Saved best policy to {out_path} (reward: {best_reward:.1f})")
    
    return {
        'episode_rewards': episode_rewards,
        'episode_lengths': episode_lengths,
        'baseline_history': baseline_history,
        'best_reward': best_reward,
        'best_weights': best_weights,
        'final_weights': weights,
        'final_baseline': baseline
    }