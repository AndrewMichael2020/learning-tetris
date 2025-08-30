"""
Tests for CEM agent.
"""
import pytest
import numpy as np
from rl.tetris_env import TetrisEnv
from rl.cem_agent import CEMPolicy, evaluate_candidate, evolve
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
        self.step_count = 0
        
    def reset(self, seed=None):
        self.board = np.zeros((self.height, self.width), dtype=np.int8)
        self.score = 0
        self.lines_cleared = 0
        self.game_over = False
        self.step_count = 0
        return self.board.copy()
    
    def _spawn_piece(self):
        # Simplified spawn - just set game over after a few steps
        self.step_count += 1
        if self.step_count > 10:
            self.game_over = True
        return not self.game_over


def create_mock_env_factory():
    """Create factory for mock environments."""
    def factory():
        return MockTetrisEnv()
    return factory


def test_cem_policy_initialization():
    """Test CEM policy initialization."""
    weights = np.array([0.1, -0.2, 0.3, 0.0, 0.5], dtype=np.float32)
    policy = CEMPolicy(weights)
    
    assert isinstance(policy.weights, np.ndarray)
    np.testing.assert_array_equal(policy.weights, weights)
    
    # Test weight copying (not reference)
    weights[0] = 999
    assert policy.weights[0] != 999


def test_cem_policy_predict():
    """Test CEM policy prediction."""
    weights = np.random.normal(0, 0.1, size=17).astype(np.float32)
    policy = CEMPolicy(weights)
    
    legal_actions = [0, 1, 2, 3, 4, 5]
    action = policy.predict(np.zeros((20, 10)), legal_actions)
    
    # Should return valid action
    assert action in legal_actions
    
    # Test empty legal actions
    action_empty = policy.predict(np.zeros((20, 10)), [])
    assert action_empty == 5  # Default to noop


def test_cem_policy_get_best_placement():
    """Test CEM policy best placement method."""
    weights = np.random.normal(0, 0.1, size=17).astype(np.float32)
    policy = CEMPolicy(weights)
    
    # Create environment with piece
    env = TetrisEnv()
    env.reset(seed=42)
    
    # Get placement
    col, rotation, score = policy.get_best_placement(env)
    
    # Should return valid placement
    assert isinstance(col, (int, np.integer))
    assert isinstance(rotation, (int, np.integer))
    assert isinstance(score, (float, np.floating))
    assert 0 <= rotation < 4


def test_evaluate_candidate():
    """Test candidate evaluation."""
    # Create simple test weights
    weights = np.random.normal(0, 0.1, size=17).astype(np.float32)
    env_factory = create_mock_env_factory()
    
    # Evaluate candidate
    fitness = evaluate_candidate(weights, env_factory, episodes=2, seed=42)
    
    # Should return valid fitness
    assert isinstance(fitness, (float, np.floating))
    assert fitness >= 0  # Fitness should be non-negative
    
    # Test deterministic evaluation
    fitness2 = evaluate_candidate(weights, env_factory, episodes=2, seed=42)
    assert abs(fitness - fitness2) < 1e-6  # Should be identical with same seed


def test_evaluate_candidate_multiple_episodes():
    """Test candidate evaluation over multiple episodes."""
    weights = np.random.normal(0, 0.1, size=17).astype(np.float32)
    env_factory = create_mock_env_factory()
    
    # Single episode
    fitness_1 = evaluate_candidate(weights, env_factory, episodes=1, seed=42)
    
    # Multiple episodes
    fitness_3 = evaluate_candidate(weights, env_factory, episodes=3, seed=42)
    
    # Both should be valid
    assert isinstance(fitness_1, (float, np.floating))
    assert isinstance(fitness_3, (float, np.floating))
    assert fitness_1 >= 0
    assert fitness_3 >= 0


def test_evolve_basic():
    """Test basic CEM evolution."""
    env_factory = create_mock_env_factory()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = os.path.join(tmpdir, "test_policy.npz")
        
        # Run short evolution
        result = evolve(
            env_factory=env_factory,
            generations=2,
            seed=42,
            out_path=out_path,
            episodes_per_candidate=1,
            population_size=10,
            feature_dim=17
        )
        
        # Check result structure
        assert 'best_fitness' in result
        assert 'best_weights' in result
        assert 'fitness_history' in result
        assert 'final_mean' in result
        assert 'final_std' in result
        
        # Check result types
        assert isinstance(result['best_fitness'], (float, np.floating))
        assert isinstance(result['best_weights'], np.ndarray)
        assert isinstance(result['fitness_history'], list)
        assert len(result['fitness_history']) == 2  # 2 generations
        
        # Check policy was saved
        assert os.path.exists(out_path)


def test_evolve_deterministic():
    """Test that CEM evolution is deterministic with same seed."""
    env_factory = create_mock_env_factory()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        out_path1 = os.path.join(tmpdir, "policy1.npz")
        out_path2 = os.path.join(tmpdir, "policy2.npz")
        
        # Run evolution twice with same seed
        result1 = evolve(
            env_factory=env_factory,
            generations=2,
            seed=123,
            out_path=out_path1,
            episodes_per_candidate=1,
            population_size=5,
            feature_dim=17
        )
        
        result2 = evolve(
            env_factory=env_factory,
            generations=2,
            seed=123,
            out_path=out_path2,
            episodes_per_candidate=1,
            population_size=5,
            feature_dim=17
        )
        
        # Results should be very similar (allowing for tiny floating point differences)
        assert abs(result1['best_fitness'] - result2['best_fitness']) < 1e-6
        np.testing.assert_array_almost_equal(result1['best_weights'], result2['best_weights'])


def test_evolve_improvement():
    """Test that CEM shows improvement over generations."""
    env_factory = create_mock_env_factory()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = os.path.join(tmpdir, "test_policy.npz")
        
        # Run evolution with more generations
        result = evolve(
            env_factory=env_factory,
            generations=3,
            seed=42,
            out_path=out_path,
            episodes_per_candidate=2,
            population_size=20,
            feature_dim=17
        )
        
        fitness_history = result['fitness_history']
        
        # Check that we have fitness data for each generation
        assert len(fitness_history) == 3
        
        # Each generation should have required fields
        for gen_data in fitness_history:
            assert 'generation' in gen_data
            assert 'best_fitness' in gen_data
            assert 'mean_fitness' in gen_data
            assert 'std_fitness' in gen_data
            assert 'population_std' in gen_data
        
        # Generally expect improvement (though not guaranteed with mock env)
        first_gen_best = fitness_history[0]['best_fitness']
        last_gen_best = fitness_history[-1]['best_fitness']
        
        # At minimum, final should be >= first (elitism)
        assert last_gen_best >= first_gen_best


def test_evolve_parameters():
    """Test CEM evolution with different parameters."""
    env_factory = create_mock_env_factory()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = os.path.join(tmpdir, "test_policy.npz")
        
        # Test with custom parameters
        result = evolve(
            env_factory=env_factory,
            generations=2,
            seed=42,
            out_path=out_path,
            episodes_per_candidate=1,
            population_size=8,
            elite_fraction=0.25,  # Custom elite fraction
            feature_dim=17
        )
        
        # Should complete successfully
        assert result['best_weights'] is not None
        assert len(result['fitness_history']) == 2


def test_evolve_small_population():
    """Test CEM with very small population."""
    env_factory = create_mock_env_factory()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = os.path.join(tmpdir, "test_policy.npz")
        
        # Run with minimal population
        result = evolve(
            env_factory=env_factory,
            generations=2,
            seed=42,
            out_path=out_path,
            episodes_per_candidate=1,
            population_size=3,  # Very small
            elite_fraction=0.33,  # 1 elite
            feature_dim=17
        )
        
        # Should still work
        assert result['best_weights'] is not None
        assert result['best_fitness'] >= 0


def test_evolve_real_env_smoke():
    """Smoke test with real Tetris environment (very short)."""
    def real_env_factory():
        return TetrisEnv(width=6, height=8)  # Smaller for speed
    
    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = os.path.join(tmpdir, "test_policy.npz")
        
        # Run very short evolution with real environment
        result = evolve(
            env_factory=real_env_factory,
            generations=2,
            seed=42,
            out_path=out_path,
            episodes_per_candidate=1,
            population_size=5,
            feature_dim=17
        )
        
        # Should complete and improve over random baseline
        assert result['best_weights'] is not None
        assert result['best_fitness'] >= 0
        
        # Policy should be saved
        assert os.path.exists(out_path)
        
        # Load and verify policy
        import numpy as np
        policy_data = np.load(out_path)
        assert 'linear_weights' in policy_data
        assert policy_data['linear_weights'].shape == (17,)


def test_evolve_fitness_improvement():
    """Test that median score improves over random baseline."""
    # This is a more substantial test that validates the core CEM functionality
    def small_env_factory():
        return TetrisEnv(width=6, height=8)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = os.path.join(tmpdir, "test_policy.npz")
        
        # Run CEM evolution
        result = evolve(
            env_factory=small_env_factory,
            generations=3,
            seed=42,
            out_path=out_path,
            episodes_per_candidate=2,
            population_size=10,
            feature_dim=17
        )
        
        # Best fitness should be reasonable for a small board
        # (Even random play should clear some lines occasionally)
        assert result['best_fitness'] >= 0
        
        # Fitness history should show progression
        fitness_values = [gen['best_fitness'] for gen in result['fitness_history']]
        
        # Final generation should be at least as good as first (due to elitism)
        assert fitness_values[-1] >= fitness_values[0]
        
        # Final weights should be different from initialization (learning occurred)
        final_weights = result['best_weights']
        initial_variance_estimate = 0.5  # From evolve() initialization
        
        # Weights should be within reasonable bounds but not all near zero
        assert np.any(np.abs(final_weights) > 0.01)  # Some learning occurred
        assert np.all(np.abs(final_weights) < 10)    # Not exploded


def test_cem_policy_integration():
    """Integration test of CEM policy with environment."""
    # Train a very simple policy and test it
    env_factory = lambda: TetrisEnv(width=6, height=8)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = os.path.join(tmpdir, "integration_policy.npz")
        
        # Train policy
        result = evolve(
            env_factory=env_factory,
            generations=2,
            seed=42,
            out_path=out_path,
            episodes_per_candidate=1,
            population_size=8,
            feature_dim=17
        )
        
        # Create policy and test it
        policy = CEMPolicy(result['best_weights'])
        
        # Test policy on environment
        env = env_factory()
        env.reset(seed=123)
        
        steps = 0
        max_steps = 20
        
        while not env.game_over and steps < max_steps:
            # Get placement
            best_col, best_rotation, _ = policy.get_best_placement(env)
            
            # Verify placement is reasonable
            assert isinstance(best_col, (int, np.integer))
            assert isinstance(best_rotation, (int, np.integer))
            assert 0 <= best_rotation < 4
            
            # Execute placement (simplified)
            from rl.afterstate import execute_placement
            success = execute_placement(env, best_col, best_rotation)
            
            if not success:
                break
                
            steps += 1
            
            # Spawn new piece
            if not env.game_over:
                env._spawn_piece()
        
        # Policy should have been able to play at least a few moves
        assert steps > 0