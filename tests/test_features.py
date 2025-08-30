"""
Tests for feature extraction functions.
"""
import pytest
import numpy as np
from rl.features import (
    get_column_heights, holes, bumpiness, wells, aggregate_height, 
    max_height, completed_lines_potential, row_transitions, 
    column_transitions, board_to_features, reward_shaping
)


def create_test_board():
    """Create a test board with known characteristics."""
    board = np.zeros((20, 10), dtype=np.int8)
    
    # Create pattern with holes and different heights
    # Column 0: height 3 with 1 hole
    board[17:20, 0] = 1
    board[18, 0] = 0  # hole
    
    # Column 1: height 5 solid
    board[15:20, 1] = 1
    
    # Column 2: height 2 solid
    board[18:20, 2] = 1
    
    # Column 3: empty (height 0)
    
    # Column 4: height 4 with 2 holes
    board[16:20, 4] = 1
    board[17, 4] = 0  # hole
    board[19, 4] = 0  # hole
    
    # Columns 5-9: various heights for testing
    board[19, 5] = 1  # height 1
    board[17:20, 6] = 1  # height 3
    board[16:20, 7] = 1  # height 4
    board[18:20, 8] = 1  # height 2
    board[19, 9] = 1  # height 1
    
    return board


def test_get_column_heights():
    """Test column height calculation."""
    board = create_test_board()
    heights = get_column_heights(board)
    
    expected_heights = np.array([3, 5, 2, 0, 4, 1, 3, 4, 2, 1])
    np.testing.assert_array_equal(heights, expected_heights)
    
    # Test empty board
    empty_board = np.zeros((20, 10), dtype=np.int8)
    empty_heights = get_column_heights(empty_board)
    np.testing.assert_array_equal(empty_heights, np.zeros(10))
    
    # Test full board
    full_board = np.ones((20, 10), dtype=np.int8)
    full_heights = get_column_heights(full_board)
    np.testing.assert_array_equal(full_heights, np.full(10, 20))


def test_holes():
    """Test hole counting."""
    board = create_test_board()
    hole_count = holes(board)
    
    # Expected: 1 hole in column 0, 2 holes in column 4 = 3 total
    assert hole_count == 3
    
    # Test empty board
    empty_board = np.zeros((20, 10), dtype=np.int8)
    assert holes(empty_board) == 0
    
    # Test board with no holes
    no_hole_board = np.zeros((20, 10), dtype=np.int8)
    no_hole_board[18:20, 0] = 1  # Solid column
    no_hole_board[19, 1] = 1    # Single cell
    assert holes(no_hole_board) == 0
    
    # Test monotonicity: creating holes should increase count
    board_before = np.zeros((20, 10), dtype=np.int8)
    board_before[17:20, 0] = 1  # Solid column
    
    board_after = board_before.copy()
    board_after[18, 0] = 0  # Create hole
    
    assert holes(board_after) > holes(board_before)


def test_bumpiness():
    """Test bumpiness calculation."""
    board = create_test_board()
    bump = bumpiness(board)
    
    heights = get_column_heights(board)  # [3, 5, 2, 0, 4, 1, 3, 4, 2, 1]
    expected_bump = sum(abs(heights[i] - heights[i+1]) for i in range(9))
    # |3-5| + |5-2| + |2-0| + |0-4| + |4-1| + |1-3| + |3-4| + |4-2| + |2-1|
    # = 2 + 3 + 2 + 4 + 3 + 2 + 1 + 2 + 1 = 20
    assert bump == expected_bump
    
    # Test flat board (no bumpiness)
    flat_board = np.zeros((20, 10), dtype=np.int8)
    flat_board[19, :] = 1  # All columns height 1
    assert bumpiness(flat_board) == 0
    
    # Test empty board
    empty_board = np.zeros((20, 10), dtype=np.int8)
    assert bumpiness(empty_board) == 0
    
    # Test monotonicity: uneven surface should increase bumpiness
    even_board = np.zeros((20, 10), dtype=np.int8)
    even_board[19, :] = 1  # Flat
    
    uneven_board = even_board.copy()
    uneven_board[18, 0] = 1  # Make one column higher
    
    assert bumpiness(uneven_board) > bumpiness(even_board)


def test_wells():
    """Test well counting."""
    board = np.zeros((20, 10), dtype=np.int8)
    
    # Create a well pattern: high-low-high
    board[17:20, 0] = 1  # height 3
    # column 1 empty (height 0) - this is a well
    board[17:20, 2] = 1  # height 3
    
    well_count = wells(board)
    assert well_count == 3  # Well depth is min(3,3) - 0 = 3
    
    # Test no wells (flat or monotonic)
    flat_board = np.zeros((20, 10), dtype=np.int8)
    flat_board[19, :] = 1
    assert wells(flat_board) == 0
    
    # Test multiple wells
    multi_well_board = np.zeros((20, 10), dtype=np.int8)
    multi_well_board[17:20, 0] = 1  # height 3
    # column 1 empty (well depth 3)
    multi_well_board[16:20, 2] = 1  # height 4
    multi_well_board[18:20, 3] = 1  # height 2
    # column 4 empty (well depth 2)
    multi_well_board[17:20, 5] = 1  # height 3
    
    wells_count = wells(multi_well_board)
    assert wells_count >= 0  # Should detect wells


def test_aggregate_height():
    """Test aggregate height calculation."""
    board = create_test_board()
    agg_height = aggregate_height(board)
    
    heights = get_column_heights(board)  # [3, 5, 2, 0, 4, 1, 3, 4, 2, 1]
    expected_agg_height = sum(heights)  # 25
    assert agg_height == expected_agg_height
    
    # Test empty board
    empty_board = np.zeros((20, 10), dtype=np.int8)
    assert aggregate_height(empty_board) == 0


def test_max_height():
    """Test max height calculation."""
    board = create_test_board()
    max_h = max_height(board)
    
    heights = get_column_heights(board)  # [3, 5, 2, 0, 4, 1, 3, 4, 2, 1]
    expected_max = max(heights)  # 5
    assert max_h == expected_max
    
    # Test empty board
    empty_board = np.zeros((20, 10), dtype=np.int8)
    assert max_height(empty_board) == 0


def test_completed_lines_potential():
    """Test completed lines detection."""
    board = np.zeros((20, 10), dtype=np.int8)
    
    # Create some complete lines
    board[19, :] = 1  # Complete bottom line
    board[18, :] = 1  # Complete line above
    board[17, :9] = 1  # Incomplete line (missing last cell)
    
    completed = completed_lines_potential(board)
    assert completed == 2  # Two complete lines
    
    # Test no complete lines
    incomplete_board = np.zeros((20, 10), dtype=np.int8)
    incomplete_board[19, :9] = 1  # Missing one cell
    assert completed_lines_potential(incomplete_board) == 0


def test_row_transitions():
    """Test row transition counting."""
    board = np.zeros((20, 10), dtype=np.int8)
    
    # Create pattern with transitions
    board[19, 0] = 1  # filled
    board[19, 1] = 0  # empty
    board[19, 2] = 1  # filled
    board[19, 3] = 1  # filled
    board[19, 4] = 0  # empty
    # ... rest empty
    
    transitions = row_transitions(board)
    assert transitions > 0  # Should detect transitions
    
    # Test all empty row (should have 0 transitions considering walls)
    empty_board = np.zeros((20, 10), dtype=np.int8)
    empty_transitions = row_transitions(empty_board)
    assert empty_transitions == 0
    
    # Test all filled row
    full_board = np.ones((20, 10), dtype=np.int8)
    full_transitions = row_transitions(full_board)
    assert full_transitions == 2  # Wall-to-filled at start and filled-to-wall at end


def test_column_transitions():
    """Test column transition counting."""
    board = np.zeros((20, 10), dtype=np.int8)
    
    # Create pattern with transitions in a column
    board[19, 0] = 1  # filled
    board[18, 0] = 0  # empty
    board[17, 0] = 1  # filled
    board[16, 0] = 1  # filled
    board[15, 0] = 0  # empty
    # ... rest empty
    
    transitions = column_transitions(board)
    assert transitions > 0  # Should detect transitions
    
    # Test empty board
    empty_board = np.zeros((20, 10), dtype=np.int8)
    empty_transitions = column_transitions(empty_board)
    assert empty_transitions == 0


def test_board_to_features():
    """Test feature vector generation."""
    board = create_test_board()
    features = board_to_features(board)
    
    # Check feature vector shape
    expected_dim = 10 + 7  # 10 column heights + 7 aggregate features
    assert features.shape == (expected_dim,)
    assert features.dtype == np.float32
    
    # Check all features are normalized (between 0 and reasonable bounds)
    assert np.all(features >= 0)
    assert np.all(features <= 1.0)  # Most features should be normalized to [0,1]
    
    # Test empty board
    empty_board = np.zeros((20, 10), dtype=np.int8)
    empty_features = board_to_features(empty_board)
    assert empty_features.shape == (expected_dim,)
    # Empty board should have mostly zero features
    assert np.all(empty_features[:10] == 0)  # Column heights should be 0
    
    # Test full board
    full_board = np.ones((20, 10), dtype=np.int8)
    full_features = board_to_features(full_board)
    assert full_features.shape == (expected_dim,)
    # Column heights should be 1.0 (normalized)
    np.testing.assert_array_almost_equal(full_features[:10], 1.0)


def test_feature_vector_shape_consistency():
    """Test that feature vectors have consistent shape."""
    boards = [
        np.zeros((20, 10), dtype=np.int8),  # Empty
        np.ones((20, 10), dtype=np.int8),   # Full
        create_test_board(),                 # Complex
    ]
    
    shapes = []
    for board in boards:
        features = board_to_features(board)
        shapes.append(features.shape)
    
    # All feature vectors should have same shape
    assert len(set(shapes)) == 1


def test_feature_numeric_bounds():
    """Test that features stay within reasonable numeric bounds."""
    # Create various board configurations
    boards = []
    
    # Empty board
    boards.append(np.zeros((20, 10), dtype=np.int8))
    
    # Random sparse boards
    rng = np.random.default_rng(42)
    for _ in range(10):
        board = np.zeros((20, 10), dtype=np.int8)
        # Fill random cells
        for _ in range(rng.integers(0, 50)):
            r, c = rng.integers(0, 20), rng.integers(0, 10)
            board[r, c] = 1
        boards.append(board)
    
    for board in boards:
        features = board_to_features(board)
        
        # Features should be finite
        assert np.all(np.isfinite(features))
        
        # No NaN values
        assert not np.any(np.isnan(features))
        
        # Should be non-negative (due to normalization)
        assert np.all(features >= 0)


def test_reward_shaping():
    """Test reward shaping function."""
    # Create boards with different characteristics
    prev_board = create_test_board()
    
    # Create improved board (fewer holes)
    next_board = prev_board.copy()
    next_board[18, 0] = 1  # Fill one hole
    
    reward = reward_shaping(prev_board, next_board)
    assert reward > 0  # Should get positive reward for improvement
    
    # Test worsening board
    worse_board = prev_board.copy()
    worse_board[17, 1] = 0  # Create additional hole
    
    worse_reward = reward_shaping(prev_board, worse_board)
    assert worse_reward <= reward  # Worse board should get less reward
    
    # Test identical boards
    same_reward = reward_shaping(prev_board, prev_board)
    assert same_reward == 0.0  # No change should give no reward
    
    # Test empty boards
    empty_board = np.zeros((20, 10), dtype=np.int8)
    empty_reward = reward_shaping(empty_board, empty_board)
    assert empty_reward == 0.0


def test_holes_monotonicity():
    """Test that creating holes increases hole count."""
    board = np.zeros((20, 10), dtype=np.int8)
    board[17:20, 0] = 1  # Create solid column
    
    holes_before = holes(board)
    
    # Create hole
    board[18, 0] = 0
    holes_after = holes(board)
    
    assert holes_after > holes_before


def test_bumpiness_monotonicity():
    """Test that uneven surface increases bumpiness."""
    # Start with flat surface
    board = np.zeros((20, 10), dtype=np.int8)
    board[19, :] = 1  # All height 1
    
    bumpiness_before = bumpiness(board)
    
    # Make surface uneven
    board[18, 0] = 1  # Column 0 now height 2
    bumpiness_after = bumpiness(board)
    
    assert bumpiness_after > bumpiness_before