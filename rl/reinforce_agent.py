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
        
    def get_action_probabilities(self, env: TetrisEnv, rng: np.random.Generator) -> Tuple[List, np.ndarray, np.ndarray]:
        """
        Get action probabilities for current state using afterstate evaluation.
        
        Returns:
            (afterstates_info, probabilities, phi_list) where:
            - probabilities[i] corresponds to afterstates_info[i]
            - phi_list[i] contains afterstate features for action i
        """
        afterstates = enumerate_afterstates(env)
        
        if not afterstates:
            return [], np.array([]), np.array([])
        
        # Calculate scores for each afterstate
        scores = []
        afterstate_info = []
        phis = []
        
        for afterstate_board, action_info in afterstates:
            phi = board_to_features(afterstate_board)
            phis.append(phi)
            # Policy scores = w·phi_afterstate only (no reward shaping here)
            score = np.dot(self.weights, phi)
            scores.append(score)
            afterstate_info.append(action_info)
        
        # Convert to softmax probabilities with numerical stability
        scores = np.array(scores)
        phis = np.array(phis)
        
        # Numerical stability: subtract max and clip extreme values
        scores = np.clip(scores - np.max(scores), -500, 500)
        
        exp_scores = np.exp(scores / self.temperature)
        
        # Additional safety check
        exp_scores = np.clip(exp_scores, 1e-10, 1e10)
        
        probabilities = exp_scores / (np.sum(exp_scores) + 1e-10)
        
        # Ensure probabilities are valid
        probabilities = np.clip(probabilities, 1e-8, 1.0)
        probabilities = probabilities / np.sum(probabilities)  # Renormalize
        
        return afterstate_info, probabilities, phis
    
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
                   max_steps: int = 5000) -> Tuple[List[Dict], float]:
    """
    Collect one episode using stochastic policy.
    Returns step data for Actor-Critic updates.
    
    Returns:
        (episode_data, total_reward) where episode_data contains step information
    """
    episode_data = []
    env.reset(seed=rng.integers(0, 1000000))
    
    # Ensure there is a piece at the start only
    if getattr(env, "current_piece", None) is None and not env.game_over:
        env._spawn_piece()
    
    total_reward = 0.0
    steps = 0
    empty_afterstate_count = 0
    
    while not env.game_over and steps < max_steps:
        # Get state features BEFORE action (for critic)
        psi_s = board_to_features(env.board)
        
        # Get action probabilities using afterstates with phi_list
        afterstates = enumerate_afterstates(env)
        
        if not afterstates:
            empty_afterstate_count += 1
            print(f"Warning: No afterstates available at step {steps}, empty count: {empty_afterstate_count}")
            break
            
        afterstate_info, action_probs, phi_list = policy.get_action_probabilities(env, rng)
        
        # Validate probabilities
        if (len(action_probs) == 0 or np.any(np.isnan(action_probs)) or 
            np.any(np.isinf(action_probs)) or np.sum(action_probs) <= 1e-10):
            print(f"Warning: Invalid probabilities detected, terminating episode at step {steps}")
            break
        
        if len(afterstate_info) == 0:
            print(f"Warning: No afterstate info at step {steps}")
            break  # No valid actions
        
        # Sample action according to probabilities
        action_idx = rng.choice(len(action_probs), p=action_probs)
        chosen_action = afterstate_info[action_idx]
        
        # Execute action using pre-computed afterstate (single source of truth)
        prev_score = env.score
        prev_lines = env.lines_cleared
        
        # Use the exact afterstate that was scored by the policy
        from .afterstate import apply_afterstate
        success = apply_afterstate(env, chosen_action['after_board'], chosen_action)
        
        if not success:
            raise RuntimeError("apply_afterstate failed for enumerated afterstate")
        
        # Get state features AFTER action (for critic bootstrap)
        psi_sp = board_to_features(env.board) if not env.game_over else np.zeros_like(psi_s)
        
        # Calculate reward using native game signals only
        reward = env.score - prev_score  # Native Tetris points only
        total_reward += reward
        
        # Store step data for Actor-Critic updates
        step_data = {
            'phi_list': phi_list.copy(),      # Afterstate features [A, D]
            'action_idx': action_idx,         # Chosen action index  
            'action_probs': action_probs.copy(),  # Action probabilities [A]
            'psi_s': psi_s.copy(),           # State features before action [D]
            'psi_sp': psi_sp.copy(),         # State features after action [D]
            'reward': reward
        }
        episode_data.append(step_data)
        
        steps += 1
    
    # Sanity check for episode length
    if len(episode_data) == 0:
        print("⚠️  Empty episode: no actions taken")
    elif len(episode_data) < 5:
        print(f"⚠️  Short episode: {len(episode_data)} steps")
    
    return episode_data, total_reward


def train(env_factory: Callable[[], TetrisEnv], episodes: int = 1000,
          seed: int = 42, out_path: str = "policies/best.npz",
          learning_rate: float = 0.002, baseline_alpha: float = 0.1,
          temperature: float = 1.4, feature_dim: int = 17) -> Dict[str, Any]:
    """
    Train Actor-Critic policy (policy gradient with learned baseline).
    
    Args:
        env_factory: Function that creates TetrisEnv instances
        episodes: Number of episodes to train
        seed: Random seed
        out_path: Path to save best policy
        learning_rate: Learning rate for actor updates
        baseline_alpha: Not used (kept for compatibility)
        temperature: Softmax temperature
        feature_dim: Dimension of feature vector
        
    Returns:
        Dictionary with training metrics
    """
    rng = np.random.default_rng(seed)
    
    # Initialize policy with better starting weights (standard Tetris heuristics)
    weights = np.zeros(feature_dim, dtype=np.float32)
    weights[0] = -2.0 + rng.normal(0, 0.5)   # Holes penalty (with noise)
    weights[1] = -0.5 + rng.normal(0, 0.2)   # Height penalty  
    weights[2] = -0.2 + rng.normal(0, 0.1)   # Bumpiness penalty
    weights[3] = 1.0 + rng.normal(0, 0.3)    # Lines cleared bonus
    weights[4:] = rng.normal(0, 0.1, size=feature_dim-4)  # Other features small random
    
    # Clip to reasonable range
    weights = np.clip(weights, -5.0, 5.0)
    policy = REINFORCEPolicy(weights, temperature)
    
    # Actor-Critic parameters
    gamma = 0.995
    lr_actor = learning_rate  # e.g., 0.001-0.003
    lr_critic = 0.003        # Critic learns faster
    entropy_beta = 0.01      # Entropy regularization
    
    # Initialize critic (value function weights on state features)
    v = np.zeros(feature_dim, dtype=np.float32)
    
    # Feature normalization
    feature_means = np.zeros(feature_dim)
    feature_vars = np.ones(feature_dim)
    feature_count = 0
    
    # Batch gradient accumulation for variance reduction
    batch_size = 8
    actor_gradient_batch = np.zeros_like(weights)
    critic_gradient_batch = np.zeros_like(v)
    batch_count = 0
    
    # Training metrics
    episode_rewards = []
    episode_lengths = []
    td_errors = []
    best_reward = float('-inf')
    best_weights = None
    
    print(f"Starting Actor-Critic training: {episodes} episodes (batch_size={batch_size})")
    
    for episode in range(episodes):
        env = env_factory()
        
        # Collect episode
        episode_data, total_reward = collect_episode(policy, env, rng)
        
        # Assert episodes are long enough to be meaningful
        assert len(episode_data) > 0, "Empty episode: no actions taken"
        
        if len(episode_data) == 0:
            continue  # Skip empty episodes
        
        # Actor-Critic updates (per step, online)
        episode_td_errors = []
        
        for step in episode_data:
            phi_list = step['phi_list']        # [A, D] afterstate features
            probs = step['action_probs']       # [A] action probabilities
            k = step['action_idx']             # Chosen action index
            psi_s = step['psi_s']              # [D] state features before action
            psi_sp = step['psi_sp']            # [D] state features after action
            r = step['reward']                 # Native reward
            
            # Update feature normalization statistics
            for phi in phi_list:
                feature_count += 1
                delta = phi - feature_means
                feature_means += delta / feature_count
                feature_vars += delta * (phi - feature_means)
            
            # Normalize features using running statistics
            if feature_count > 50:  # Wait for stable statistics
                phi_list_norm = (phi_list - feature_means) / np.sqrt(feature_vars / feature_count + 1e-8)
                psi_s_norm = (psi_s - feature_means) / np.sqrt(feature_vars / feature_count + 1e-8)
                psi_sp_norm = (psi_sp - feature_means) / np.sqrt(feature_vars / feature_count + 1e-8)
            else:
                phi_list_norm = phi_list
                psi_s_norm = psi_s
                psi_sp_norm = psi_sp
            
            # --- CRITIC UPDATE (TD(0)) ---
            V_s = float(np.dot(v, psi_s_norm))                    # V(s_t)
            V_sp = float(np.dot(v, psi_sp_norm)) if not env.game_over else 0.0  # V(s_{t+1})
            delta = r + gamma * V_sp - V_s                        # TD error
            
            episode_td_errors.append(abs(delta))
            
            # Accumulate critic gradient
            critic_gradient_batch += delta * psi_s_norm
            
            # --- ACTOR UPDATE ---
            phi_k = phi_list_norm[k]                             # Chosen afterstate features
            phi_bar = probs @ phi_list_norm                      # Expected features Σ p_j φ_j
            actor_grad = phi_k - phi_bar                         # Policy gradient
            
            # Accumulate actor gradient
            actor_gradient_batch += delta * actor_grad
            
            # Add entropy regularization
            log_probs = np.log(probs + 1e-8)
            entropy_weights = probs * (-(log_probs + 1.0))       # [A]
            entropy_features = (phi_list_norm - phi_bar[None, :]).sum(axis=0)  # Simplified entropy grad
            actor_gradient_batch += entropy_beta * entropy_features
        
        # Update batch count
        batch_count += 1
        
        # Apply batch updates
        if batch_count >= batch_size or episode == episodes - 1:
            # Apply accumulated gradients
            if batch_count > 0:
                # Actor update
                avg_actor_grad = actor_gradient_batch / batch_count
                weights += lr_actor * np.clip(avg_actor_grad, -2, 2)
                weights = np.clip(weights, -10, 10)
                
                # Critic update
                avg_critic_grad = critic_gradient_batch / batch_count
                v += lr_critic * np.clip(avg_critic_grad, -2, 2)
                v = np.clip(v, -10, 10)
                
                # Update policy
                policy.weights = weights.copy()
            
            # Reset batch accumulators
            actor_gradient_batch = np.zeros_like(weights)
            critic_gradient_batch = np.zeros_like(v)
            batch_count = 0
        
        # Check for NaN in weights
        if np.any(np.isnan(weights)) or np.any(np.isnan(v)):
            print(f"Warning: NaN detected at episode {episode+1}, resetting")
            weights = rng.normal(0, 0.01, size=feature_dim).astype(np.float32)
            weights = np.clip(weights, -0.1, 0.1)
            v = np.zeros(feature_dim, dtype=np.float32)
            policy.weights = weights.copy()
        
        # Temperature annealing for exploration control (1.4 → 0.9)
        if episode > 0 and episode % 100 == 0:
            policy.temperature = max(0.9, policy.temperature * 0.95)
        
        # Track metrics
        episode_rewards.append(total_reward)
        episode_lengths.append(len(episode_data))
        if episode_td_errors:
            td_errors.append(np.mean(episode_td_errors))
        
        # Track best policy
        if total_reward > best_reward:
            best_reward = total_reward
            best_weights = weights.copy()
        
        # Print progress with TD error monitoring
        if (episode + 1) % 100 == 0 or episode == 0:
            recent_rewards = episode_rewards[-100:]
            recent_lengths = episode_lengths[-100:]
            recent_td = td_errors[-100:] if td_errors else [0]
            mean_reward = np.mean(recent_rewards)
            mean_length = np.mean(recent_lengths)
            mean_td = np.mean(recent_td)
            current_lr = lr_actor * (0.95 ** (episode // 100))
            print(f"Episode {episode+1}/{episodes}: "
                  f"Reward={total_reward:.1f}, "
                  f"Mean(100)={mean_reward:.1f}, "
                  f"Length={len(episode_data)}, "
                  f"AvgLen={mean_length:.1f}, "
                  f"TD|δ|={mean_td:.2f}, "
                  f"LR={current_lr:.4f}, "
                  f"Temp={policy.temperature:.2f}")
    
    # Final sanity check and reporting
    if len(episode_lengths) > 100:
        avg_length = np.mean(episode_lengths[-100:])
        avg_td = np.mean(td_errors[-100:]) if td_errors else 0
        print(f"Final metrics:")
        print(f"  Average episode length: {avg_length:.1f}")
        print(f"  Average TD error: {avg_td:.2f}")
        if avg_length < 5:
            print("⚠️  Average episode length < 5: turn progression bug likely present")
        else:
            print("✅ Episodes running long enough for learning")
    
    # Save best policy
    if best_weights is not None:
        policy_dict = {'linear_weights': best_weights}
        metadata = {
            'algorithm': 'Actor-Critic',
            'episodes': episodes,
            'learning_rate': learning_rate,
            'baseline_alpha': baseline_alpha,
            'temperature': temperature,
            'best_reward': float(best_reward),
            'gamma': gamma,
            'lr_actor': lr_actor,
            'lr_critic': lr_critic,
            'seed': seed
        }
        save_policy(policy_dict, out_path, metadata)
        print(f"Saved best policy to {out_path} (reward: {best_reward:.1f})")
    
    return {
        'episode_rewards': episode_rewards,
        'episode_lengths': episode_lengths,
        'td_errors': td_errors,
        'best_reward': best_reward,
        'best_weights': best_weights,
        'final_weights': weights,
        'final_critic_weights': v
    }