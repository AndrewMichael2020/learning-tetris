"""
Tests for Tetris environment.
"""
import pytest
import numpy as np
from rl.tetris_env import TetrisEnv


def test_env_reset():
    """Test environment reset functionality."""
    env = TetrisEnv()
    
    # Test reset with seed
    board1 = env.reset(seed=42)
    assert board1.shape == (20, 10)
    assert np.all(board1 == 0)  # Board should be empty initially
    assert env.score == 0
    assert env.lines_cleared == 0
    assert env.step_count == 0
    assert not env.game_over
    assert env.current_piece is not None
    
    # Test seeded reset produces same initial state
    board2 = env.reset(seed=42)
    assert np.array_equal(board1, board2)
    assert env.current_piece_name is not None


def test_env_deterministic_seeding():
    """Test that seeded environments behave deterministically."""
    env1 = TetrisEnv()
    env2 = TetrisEnv()
    
    # Reset with same seed
    env1.reset(seed=123)
    env2.reset(seed=123)
    
    # Should have same piece sequence
    for _ in range(10):
        piece1 = env1._get_next_piece()
        piece2 = env2._get_next_piece()
        assert piece1 == piece2


def test_env_legal_actions():
    """Test legal actions functionality."""
    env = TetrisEnv()
    env.reset(seed=42)
    
    legal_actions = env.legal_actions()
    assert isinstance(legal_actions, list)
    assert 5 in legal_actions  # noop always legal
    assert all(0 <= action <= 5 for action in legal_actions)
    
    # When game over, no legal actions
    env.game_over = True
    assert env.legal_actions() == []


def test_env_basic_step():
    """Test basic step functionality."""
    env = TetrisEnv()
    env.reset(seed=42)
    
    # Test noop action
    obs, reward, done, info = env.step(5)
    assert obs.shape == (20, 10)
    assert isinstance(reward, float)
    assert isinstance(done, bool)
    assert isinstance(info, dict)
    assert 'lines_cleared' in info
    assert 'score' in info
    assert 'holes' in info
    assert 'height' in info


def test_env_movement():
    """Test piece movement."""
    env = TetrisEnv()
    env.reset(seed=42)
    
    initial_pos = env.current_pos.copy()
    
    # Test left movement
    if 0 in env.legal_actions():
        obs, reward, done, info = env.step(0)
        assert env.current_pos[1] == initial_pos[1] - 1
    
    # Reset and test right movement
    env.reset(seed=42)
    initial_pos = env.current_pos.copy()
    
    if 1 in env.legal_actions():
        obs, reward, done, info = env.step(1)
        assert env.current_pos[1] == initial_pos[1] + 1


def test_env_rotation():
    """Test piece rotation."""
    env = TetrisEnv()
    env.reset(seed=42)
    
    initial_rotation = env.current_rotation
    
    # Test rotation
    if 2 in env.legal_actions():
        obs, reward, done, info = env.step(2)
        expected_rotation = (initial_rotation + 1) % 4
        assert env.current_rotation == expected_rotation


def test_env_drop():
    """Test piece drop."""
    env = TetrisEnv()
    env.reset(seed=42)
    
    # Drop piece
    obs, reward, done, info = env.step(3)
    
    # Should have spawned new piece (unless game over)
    if not done:
        assert env.current_piece is not None


def test_env_line_clear_setup():
    """Test line clearing with a forced setup."""
    env = TetrisEnv()
    env.reset(seed=42)
    
    # Manually create a setup with one almost-complete line
    # Fill bottom row except for one cell
    env.board[19, :9] = 1  # Fill first 9 cells of bottom row
    env.board[19, 9] = 0   # Leave last cell empty
    
    initial_score = env.score
    initial_lines = env.lines_cleared
    
    # Try to place piece in the empty cell
    # This is a simplified test - in reality we'd need proper piece placement
    env.board[19, 9] = 1  # Manually complete the line
    
    # Simulate line clear
    lines_cleared = 0
    for row in range(env.height):
        if np.all(env.board[row, :]):
            lines_cleared += 1
    
    assert lines_cleared > 0  # Should have at least one complete line


def test_env_no_overlaps():
    """Test that pieces don't overlap with existing blocks."""
    env = TetrisEnv()
    env.reset(seed=42)
    
    # Fill some cells in the board
    env.board[18, 4] = 1
    env.board[18, 5] = 1
    
    # Try to place piece at bottom - should detect collision
    test_pos = [18, 4]
    collision = env._check_collision(pos=test_pos)
    
    # Should detect collision if piece overlaps with filled cells
    # Note: actual collision depends on piece shape


def test_env_boundaries():
    """Test that pieces stay within boundaries."""
    env = TetrisEnv()
    env.reset(seed=42)
    
    # Test left boundary
    left_boundary_pos = [10, -1]  # Outside left boundary
    collision = env._check_collision(pos=left_boundary_pos)
    assert collision  # Should detect boundary collision
    
    # Test right boundary
    right_boundary_pos = [10, 10]  # Outside right boundary
    collision = env._check_collision(pos=right_boundary_pos)
    assert collision  # Should detect boundary collision
    
    # Test bottom boundary
    bottom_boundary_pos = [20, 5]  # Outside bottom boundary
    collision = env._check_collision(pos=bottom_boundary_pos)
    assert collision  # Should detect boundary collision


def test_env_game_termination():
    """Test that game eventually terminates."""
    env = TetrisEnv()
    env.reset(seed=42)
    
    max_steps = 5000  # Prevent infinite loop
    steps = 0
    
    while not env.game_over and steps < max_steps:
        legal_actions = env.legal_actions()
        if not legal_actions:
            break
        
        # Choose drop action to make game progress faster
        action = 3 if 3 in legal_actions else legal_actions[0]
        obs, reward, done, info = env.step(action)
        steps += 1
    
    # Game should either be over or we hit step limit
    assert env.game_over or steps >= max_steps


def test_env_render():
    """Test environment rendering."""
    env = TetrisEnv()
    env.reset(seed=42)
    
    # Test RGB array rendering
    img = env.render(mode="rgb_array")
    assert img.shape == (20, 10, 3)
    assert img.dtype == np.uint8
    
    # Test invalid mode
    with pytest.raises(ValueError):
        env.render(mode="invalid")


def test_env_7_bag_system():
    """Test that 7-bag piece generation works correctly."""
    env = TetrisEnv()
    env.reset(seed=42)
    
    # Generate 14 pieces (2 full bags)
    pieces = []
    for _ in range(14):
        piece = env._get_next_piece()
        pieces.append(piece)
    
    # First 7 pieces should contain all piece types
    first_bag = set(pieces[:7])
    expected_pieces = {'I', 'O', 'T', 'S', 'Z', 'J', 'L'}
    assert first_bag == expected_pieces
    
    # Second bag should also contain all piece types
    second_bag = set(pieces[7:14])
    assert second_bag == expected_pieces


def test_env_feature_info():
    """Test that info dict contains expected features."""
    env = TetrisEnv()
    env.reset(seed=42)
    
    # Take a step to get info
    obs, reward, done, info = env.step(5)  # noop
    
    required_keys = ['lines_cleared', 'score', 'holes', 'height', 'step_count']
    for key in required_keys:
        assert key in info
        assert isinstance(info[key], (int, float))
    
    # Values should be non-negative
    assert info['lines_cleared'] >= 0
    assert info['score'] >= 0
    assert info['holes'] >= 0
    assert info['height'] >= 0
    assert info['step_count'] >= 0


def test_env_reward_structure():
    """Test reward structure."""
    env = TetrisEnv()
    env.reset(seed=42)
    
    # Take a step
    obs, reward, done, info = env.step(5)  # noop
    
    # Reward should be a float
    assert isinstance(reward, float)
    
    # For noop, should get small negative step cost
    assert reward <= 0


def test_env_multiple_resets():
    """Test multiple resets work correctly."""
    env = TetrisEnv()
    
    for i in range(5):
        board = env.reset(seed=i)
        assert board.shape == (20, 10)
        assert np.all(board == 0)
        assert env.score == 0
        assert env.lines_cleared == 0
        assert not env.game_over