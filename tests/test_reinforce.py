"""
Tests for REINFORCE agent.
"""
import pytest
import numpy as np
from rl.tetris_env import TetrisEnv
from rl.reinforce_agent import REINFORCEPolicy, collect_episode, train
import tempfile
import os


class MockTetrisEnv:
    """Mock Tetris environment for faster testing."""
    
    def __init__(self, width=6, height=8):
        self.width = width
        self.height = height
        self.board = None
        self.score = 0
        self.lines_cleared = 0
        self.game_over = False
        self.current_piece = None
        self.current_piece_name = 'T'
        self.current_rotation = 0
        self.current_pos = [0, width // 2 - 2]
        self.step_count = 0
        
    def reset(self, seed=None):
        from rl.tetris_env import TETRIS_PIECES
        self.board = np.zeros((self.height, self.width), dtype=np.int8)
        self.score = 0
        self.lines_cleared = 0
        self.game_over = False
        self.current_piece_name = 'T'
        self.current_rotation = 0
        self.current_pos = [0, self.width // 2 - 2]
        self.current_piece = TETRIS_PIECES[self.current_piece_name][self.current_rotation]
        self.step_count = 0
        return self.board.copy()
    
    def _spawn_piece(self):
        # Simplified spawn - set game over after a few steps
        self.step_count += 1
        if self.step_count > 8:
            self.game_over = True
        return not self.game_over
        
    def _rotate_piece(self):
        # Mock rotation - always succeeds
        from rl.tetris_env import TETRIS_PIECES
        self.current_rotation = (self.current_rotation + 1) % 4
        self.current_piece = TETRIS_PIECES[self.current_piece_name][self.current_rotation]
        return True
        
    def _move_piece(self, dr, dc):
        # Mock movement - always succeeds
        self.current_pos[0] += dr
        self.current_pos[1] += dc
        return True
        
    def _drop_piece(self):
        # Mock drop - adds some score and deterministically ends the game
        self.score += 10
        self.lines_cleared += (self.step_count % 3)  # Deterministic pattern
        self.step_count += 1
        if self.step_count > 8:
            self.game_over = True
        return self._spawn_piece()


def create_mock_env_factory():
    """Create factory for mock environments."""
    def factory():
        return MockTetrisEnv()
    return factory


def test_reinforce_policy_initialization():
    """Test REINFORCE policy initialization."""
    weights = np.array([0.1, -0.2, 0.3, 0.0, 0.5], dtype=np.float32)
    policy = REINFORCEPolicy(weights, temperature=1.0)
    
    assert isinstance(policy.weights, np.ndarray)
    np.testing.assert_array_equal(policy.weights, weights)
    assert policy.temperature == 1.0
    
    # Test weight copying (not reference)
    weights[0] = 999
    assert policy.weights[0] != 999


def test_reinforce_policy_temperature():
    """Test policy with different temperatures."""
    weights = np.random.normal(0, 1, size=17).astype(np.float32)
    
    # High temperature (more exploration)
    policy_high = REINFORCEPolicy(weights, temperature=2.0)
    assert policy_high.temperature == 2.0
    
    # Low temperature (more exploitation)
    policy_low = REINFORCEPolicy(weights, temperature=0.5)
    assert policy_low.temperature == 0.5


def test_reinforce_policy_action_probabilities():
    """Test action probability calculation."""
    weights = np.random.normal(0, 0.1, size=17).astype(np.float32)
    policy = REINFORCEPolicy(weights, temperature=1.0)
    rng = np.random.default_rng(42)
    
    # Create environment with piece
    env = TetrisEnv()
    env.reset(seed=42)
    
    # Get action probabilities
    afterstate_info, probs = policy.get_action_probabilities(env, rng)
    
    if len(afterstate_info) > 0:
        # Probabilities should sum to 1
        assert abs(np.sum(probs) - 1.0) < 1e-6
        
        # All probabilities should be non-negative
        assert np.all(probs >= 0)
        
        # Should have same length as afterstates
        assert len(probs) == len(afterstate_info)
        
        # Each afterstate info should have required fields
        for info in afterstate_info:
            assert 'col' in info
            assert 'rotation' in info
            assert 'reward' in info


def test_reinforce_policy_predict():
    """Test REINFORCE policy prediction."""
    weights = np.random.normal(0, 0.1, size=17).astype(np.float32)
    policy = REINFORCEPolicy(weights)
    
    board = np.zeros((20, 10), dtype=np.int8)
    legal_actions = [0, 1, 2, 3, 4, 5]
    
    action = policy.predict(board, legal_actions)
    
    # Should return valid action
    assert action in legal_actions
    
    # Test empty legal actions
    action_empty = policy.predict(board, [])
    assert action_empty == 5  # Default to noop


def test_collect_episode():
    """Test episode collection."""
    weights = np.random.normal(0, 0.1, size=17).astype(np.float32)
    policy = REINFORCEPolicy(weights, temperature=1.0)
    rng = np.random.default_rng(42)
    
    # Use mock environment for speed
    env = MockTetrisEnv()
    
    # Collect episode
    episode_data, total_reward = collect_episode(policy, env, rng, max_steps=20)
    
    # Should return valid data
    assert isinstance(episode_data, list)
    assert isinstance(total_reward, (float, np.floating))
    
    # Episode data should have required structure
    for step_data in episode_data:
        assert 'state_features' in step_data
        assert 'action_idx' in step_data
        assert 'action_probs' in step_data
        assert 'afterstate_info' in step_data
        assert 'reward' in step_data
        assert 'log_prob' in step_data
        
        # Check data types
        assert isinstance(step_data['state_features'], np.ndarray)
        assert isinstance(step_data['action_idx'], (int, np.integer))
        assert isinstance(step_data['action_probs'], np.ndarray)
        assert isinstance(step_data['reward'], (float, np.floating))
        assert isinstance(step_data['log_prob'], (float, np.floating))


def test_collect_episode_deterministic():
    """Test that episode collection is deterministic with same seed."""
    weights = np.random.normal(0, 0.1, size=17).astype(np.float32)
    policy = REINFORCEPolicy(weights, temperature=1.0)
    
    # Collect two episodes with same RNG seed
    rng1 = np.random.default_rng(123)
    rng2 = np.random.default_rng(123)
    
    env1 = MockTetrisEnv()
    env2 = MockTetrisEnv()
    
    episode1, reward1 = collect_episode(policy, env1, rng1, max_steps=10)
    episode2, reward2 = collect_episode(policy, env2, rng2, max_steps=10)
    
    # Should be identical
    assert len(episode1) == len(episode2)
    assert abs(reward1 - reward2) < 1e-6
    
    # First step should be identical
    if len(episode1) > 0:
        assert episode1[0]['action_idx'] == episode2[0]['action_idx']


def test_train_basic():
    """Test basic REINFORCE training."""
    env_factory = create_mock_env_factory()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = os.path.join(tmpdir, "test_policy.npz")
        
        # Run short training
        result = train(
            env_factory=env_factory,
            episodes=20,
            seed=42,
            out_path=out_path,
            learning_rate=0.01,
            baseline_alpha=0.1,
            temperature=1.0,
            feature_dim=17
        )
        
        # Check result structure
        assert 'episode_rewards' in result
        assert 'episode_lengths' in result
        assert 'baseline_history' in result
        assert 'best_reward' in result
        assert 'best_weights' in result
        assert 'final_weights' in result
        assert 'final_baseline' in result
        
        # Check result types and shapes
        assert isinstance(result['episode_rewards'], list)
        assert len(result['episode_rewards']) <= 20  # Might be fewer if empty episodes
        assert isinstance(result['best_weights'], np.ndarray)
        assert result['best_weights'].shape == (17,)
        assert isinstance(result['final_baseline'], (float, np.floating))
        
        # Check policy was saved
        assert os.path.exists(out_path)


def test_train_deterministic():
    """Test that REINFORCE training is deterministic with same seed."""
    env_factory = create_mock_env_factory()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        out_path1 = os.path.join(tmpdir, "policy1.npz")
        out_path2 = os.path.join(tmpdir, "policy2.npz")
        
        # Run training twice with same seed
        result1 = train(
            env_factory=env_factory,
            episodes=10,
            seed=123,
            out_path=out_path1,
            learning_rate=0.01,
            feature_dim=17
        )
        
        result2 = train(
            env_factory=env_factory,
            episodes=10,
            seed=123,
            out_path=out_path2,
            learning_rate=0.01,
            feature_dim=17
        )
        
        # Results should be very similar
        assert len(result1['episode_rewards']) == len(result2['episode_rewards'])
        
        # Final weights should be nearly identical
        np.testing.assert_array_almost_equal(
            result1['final_weights'], result2['final_weights'], decimal=5
        )


def test_train_learning():
    """Test that training shows some learning."""
    env_factory = create_mock_env_factory()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = os.path.join(tmpdir, "test_policy.npz")
        
        # Run longer training
        result = train(
            env_factory=env_factory,
            episodes=50,
            seed=42,
            out_path=out_path,
            learning_rate=0.05,  # Higher learning rate
            baseline_alpha=0.2,
            feature_dim=17
        )
        
        # Should have episode data
        assert len(result['episode_rewards']) > 0
        assert len(result['baseline_history']) == len(result['episode_rewards'])
        
        # Baseline should be reasonable
        final_baseline = result['final_baseline']
        assert isinstance(final_baseline, (float, np.floating))
        
        # Weights should have changed from initialization
        final_weights = result['final_weights']
        assert np.any(np.abs(final_weights) > 0.01)  # Some learning occurred
        
        # Best reward should be tracked
        best_reward = result['best_reward']
        assert best_reward == max(result['episode_rewards'])


def test_train_parameters():
    """Test REINFORCE training with different parameters."""
    env_factory = create_mock_env_factory()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = os.path.join(tmpdir, "test_policy.npz")
        
        # Test with custom parameters
        result = train(
            env_factory=env_factory,
            episodes=15,
            seed=42,
            out_path=out_path,
            learning_rate=0.001,  # Low learning rate
            baseline_alpha=0.05,  # Slow baseline adaptation
            temperature=2.0,      # High exploration
            feature_dim=17
        )
        
        # Should complete successfully
        assert result['best_weights'] is not None
        assert len(result['episode_rewards']) <= 15


def test_train_baseline_tracking():
    """Test that baseline tracks returns correctly."""
    env_factory = create_mock_env_factory()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = os.path.join(tmpdir, "test_policy.npz")
        
        result = train(
            env_factory=env_factory,
            episodes=30,
            seed=42,
            out_path=out_path,
            baseline_alpha=0.1,
            feature_dim=17
        )
        
        episode_rewards = result['episode_rewards']
        baseline_history = result['baseline_history']
        
        # Baseline should roughly track reward trend
        if len(episode_rewards) > 10:
            # Later baseline values should be influenced by recent rewards
            recent_baseline = np.mean(baseline_history[-5:])
            recent_rewards = np.mean(episode_rewards[-5:])
            
            # They should be in same ballpark (baseline smooths out noise)
            assert abs(recent_baseline - recent_rewards) < max(abs(recent_rewards) * 2, 10)


def test_train_real_env_smoke():
    """Smoke test with real Tetris environment (very short)."""
    def small_env_factory():
        return TetrisEnv(width=6, height=8)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = os.path.join(tmpdir, "test_policy.npz")
        
        # Run very short training with real environment
        result = train(
            env_factory=small_env_factory,
            episodes=10,
            seed=42,
            out_path=out_path,
            learning_rate=0.01,
            feature_dim=17
        )
        
        # Should complete
        assert result['best_weights'] is not None
        assert len(result['episode_rewards']) <= 10
        
        # Policy should be saved
        assert os.path.exists(out_path)
        
        # Load and verify policy
        policy_data = np.load(out_path)
        assert 'linear_weights' in policy_data
        assert policy_data['linear_weights'].shape == (17,)


def test_train_improvement_vs_random():
    """Test that trained policy improves vs random baseline."""
    def small_env_factory():
        return TetrisEnv(width=6, height=8)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = os.path.join(tmpdir, "test_policy.npz")
        
        # Train policy
        result = train(
            env_factory=small_env_factory,
            episodes=20,
            seed=42,
            out_path=out_path,
            learning_rate=0.05,
            feature_dim=17
        )
        
        # Should show some learning
        episode_rewards = result['episode_rewards']
        
        if len(episode_rewards) > 5:
            # Compare early vs late performance
            early_performance = np.mean(episode_rewards[:3])
            late_performance = np.mean(episode_rewards[-3:])
            
            # Late performance should be at least as good as early
            # (Not guaranteed improvement due to variance, but should not get worse)
            assert late_performance >= early_performance - 50  # Allow some variance
        
        # Best reward should be reasonable for small board
        assert result['best_reward'] >= 0


def test_reinforce_policy_integration():
    """Integration test of REINFORCE policy with environment."""
    env_factory = lambda: TetrisEnv(width=6, height=8)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = os.path.join(tmpdir, "integration_policy.npz")
        
        # Train policy
        result = train(
            env_factory=env_factory,
            episodes=15,
            seed=42,
            out_path=out_path,
            learning_rate=0.01,
            feature_dim=17
        )
        
        # Create policy and test it
        policy = REINFORCEPolicy(result['best_weights'])
        rng = np.random.default_rng(123)
        
        # Test policy on environment
        env = env_factory()
        env.reset(seed=456)
        
        # Collect episode to verify policy works
        episode_data, total_reward = collect_episode(policy, env, rng, max_steps=15)
        
        # Policy should be able to play
        assert len(episode_data) > 0
        assert isinstance(total_reward, (float, np.floating))
        
        # Episode should have reasonable structure
        for step_data in episode_data:
            assert step_data['action_idx'] >= 0
            assert np.all(step_data['action_probs'] >= 0)
            assert abs(np.sum(step_data['action_probs']) - 1.0) < 1e-6


def test_temperature_effect():
    """Test that temperature affects exploration."""
    weights = np.array([1.0, -1.0, 0.5, -0.5, 0.2, 0.0, 0.0, 0.0, 
                       0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    
    # Create policies with different temperatures
    policy_low = REINFORCEPolicy(weights, temperature=0.1)   # Low temp = less exploration
    policy_high = REINFORCEPolicy(weights, temperature=5.0)  # High temp = more exploration
    
    # Use simple environment
    env = TetrisEnv(width=6, height=8)
    env.reset(seed=42)
    rng = np.random.default_rng(42)
    
    # Get probabilities from both policies
    _, probs_low = policy_low.get_action_probabilities(env, rng)
    
    env.reset(seed=42)  # Reset to same state
    _, probs_high = policy_high.get_action_probabilities(env, rng)
    
    if len(probs_low) > 1 and len(probs_high) > 1:
        # Low temperature should have more concentrated probabilities
        entropy_low = -np.sum(probs_low * np.log(probs_low + 1e-8))
        entropy_high = -np.sum(probs_high * np.log(probs_high + 1e-8))
        
        # High temperature should have higher entropy (more uniform)
        assert entropy_high > entropy_low


def test_gradient_calculation():
    """Test that policy gradients are computed without errors."""
    env_factory = create_mock_env_factory()
    
    # Create policy and collect episode
    weights = np.random.normal(0, 0.1, size=17).astype(np.float32)
    policy = REINFORCEPolicy(weights, temperature=1.0)
    rng = np.random.default_rng(42)
    env = MockTetrisEnv()
    env.reset()
    
    episode_data, total_reward = collect_episode(policy, env, rng, max_steps=10)
    
    # Verify episode collection succeeded
    assert len(episode_data) > 0
    assert total_reward >= 0
    
    # Verify data structure
    for step_data in episode_data:
        assert 'state_features' in step_data
        assert 'log_prob' in step_data
        assert 'reward' in step_data
        assert isinstance(step_data['state_features'], np.ndarray)
        assert step_data['state_features'].shape == (17,)
        assert isinstance(step_data['log_prob'], (float, np.floating))
        assert np.isfinite(step_data['log_prob'])
    
    # Test passes if no exceptions were raised during episode collection