"""
Feature extraction for Tetris board states.
Implements standard Tetris features: heights, holes, bumpiness, wells, etc.
"""
import numpy as np
from typing import Tuple


def get_column_heights(board: np.ndarray) -> np.ndarray:
    """Get height of each column (distance from bottom to highest filled cell)."""
    height, width = board.shape
    heights = np.zeros(width, dtype=np.int32)
    
    for col in range(width):
        for row in range(height):
            if board[row, col]:
                heights[col] = height - row
                break
    
    return heights


def holes(board: np.ndarray) -> int:
    """Count number of holes (empty cells with filled cells above them)."""
    height, width = board.shape
    hole_count = 0
    
    for col in range(width):
        filled_found = False
        for row in range(height):
            if board[row, col]:
                filled_found = True
            elif filled_found and not board[row, col]:
                hole_count += 1
    
    return hole_count


def bumpiness(board: np.ndarray) -> int:
    """Calculate bumpiness (sum of height differences between adjacent columns)."""
    heights = get_column_heights(board)
    bump = 0
    
    for i in range(len(heights) - 1):
        bump += abs(heights[i] - heights[i + 1])
    
    return bump


def wells(board: np.ndarray) -> int:
    """Count wells (empty columns surrounded by filled columns or walls)."""
    heights = get_column_heights(board)
    well_count = 0
    width = len(heights)
    
    for col in range(width):
        left_height = heights[col - 1] if col > 0 else float('inf')
        right_height = heights[col + 1] if col < width - 1 else float('inf')
        current_height = heights[col]
        
        # A well is formed when current column is lower than both neighbors
        if current_height < left_height and current_height < right_height:
            well_depth = min(left_height, right_height) - current_height
            well_count += well_depth
    
    return well_count


def aggregate_height(board: np.ndarray) -> int:
    """Sum of all column heights."""
    return int(np.sum(get_column_heights(board)))


def max_height(board: np.ndarray) -> int:
    """Maximum column height."""
    heights = get_column_heights(board)
    return int(np.max(heights)) if len(heights) > 0 else 0


def completed_lines_potential(board: np.ndarray) -> int:
    """Count number of rows that are completely filled."""
    height, width = board.shape
    complete_lines = 0
    
    for row in range(height):
        if np.all(board[row, :]):
            complete_lines += 1
    
    return complete_lines


def row_transitions(board: np.ndarray) -> int:
    """
    Count row transitions (horizontal).
    Counts transitions between filled/empty cells within rows and with walls.
    Walls are considered empty/boundary.
    """
    height, width = board.shape
    transitions = 0
    
    # Check if board is completely empty
    if not np.any(board):
        return 0
    
    for row in range(height):
        # Count transitions for this row including walls
        prev_filled = False  # Left wall is considered empty
        for col in range(width):
            current_filled = bool(board[row, col])
            if prev_filled != current_filled:
                transitions += 1
            prev_filled = current_filled
        # Transition to right wall (right wall is considered empty)
        if prev_filled:  # If last cell is filled, transition to empty wall
            transitions += 1
    
    return transitions


def column_transitions(board: np.ndarray) -> int:
    """
    Count column transitions (vertical).
    For empty boards, return 0. For non-empty boards, count transitions including boundaries.
    """
    height, width = board.shape
    transitions = 0
    
    # Check if board is completely empty
    if not np.any(board):
        return 0
    
    for col in range(width):
        # Count transitions including boundaries for non-empty columns
        if np.any(board[:, col]):  # Only count for columns with some filled cells
            prev_filled = False  # Top boundary is considered empty
            for row in range(height):
                current_filled = bool(board[row, col])
                if prev_filled != current_filled:
                    transitions += 1
                prev_filled = current_filled
            # Transition to bottom boundary (bottom is considered filled)
            if not prev_filled:  # If last cell is empty, transition to filled bottom
                transitions += 1
    
    return transitions


def board_to_features_ml(board: np.ndarray) -> np.ndarray:
    """
    Convert board state to feature vector for ML algorithms (CEM, REINFORCE).
    
    Features (17 total):
    - Per-column heights (10) - always 10, padded with zeros for narrower boards
    - Total holes (1) - NEGATIVE (penalty)
    - Aggregate height (1) - NEGATIVE (penalty) 
    - Bumpiness (1) - NEGATIVE (penalty)
    - Wells (1) - NEGATIVE (penalty)
    - Max height (1) - NEGATIVE (penalty)
    - Completed lines (1) - POSITIVE (reward)
    - Row transitions (1) - NEGATIVE (penalty)
    
    All features are normalized to reasonable ranges.
    """
    height, width = board.shape
    features = []
    
    # Per-column heights (always 10, padded with zeros for narrower boards)
    heights = get_column_heights(board) / height
    # Pad or truncate to exactly 10 columns
    if width < 10:
        heights = np.pad(heights, (0, 10 - width), mode='constant', constant_values=0)
    elif width > 10:
        heights = heights[:10]
    features.extend(heights)
    
    # Count-based features (make bad features negative for ML discrimination)
    board_area = height * width
    
    # Holes: direct penalty (negative feature)
    holes_count = -holes(board)  # Direct penalty, no normalization
    features.append(holes_count)
    
    # Aggregate height: penalty for being too high
    agg_height = -aggregate_height(board) / (height * width)
    features.append(agg_height)
    
    # Bumpiness: direct penalty
    bump = -bumpiness(board)  # Direct penalty
    features.append(bump)
    
    # Wells: penalty for deep wells
    wells_count = -wells(board)
    features.append(wells_count)
    
    # Max height: penalty for being too high
    max_h = -max_height(board) / height
    features.append(max_h)
    
    # Completed lines: positive reward
    completed = completed_lines_potential(board)  # Keep positive
    features.append(completed)
    
    # Row transitions: penalty for irregular patterns
    transitions = -column_transitions(board) / width
    features.append(transitions)
    
    return np.array(features, dtype=np.float32)


def board_to_features_deterministic(board: np.ndarray) -> np.ndarray:
    """
    Convert board state to feature vector for deterministic algorithms 
    (Greedy, Tabu, SA, ACO).
    
    Features (17 total):
    - Per-column heights (10) - always 10, padded with zeros for narrower boards
    - Total holes (1) - POSITIVE (penalty)
    - Aggregate height (1) - POSITIVE (penalty) 
    - Bumpiness (1) - POSITIVE (penalty)
    - Wells (1) - POSITIVE (penalty)
    - Max height (1) - POSITIVE (penalty)
    - Completed lines (1) - NEGATIVE (reward, so cost is negative)
    - Row transitions (1) - POSITIVE (penalty)
    
    All features are normalized to reasonable ranges for cost calculation.
    """
    height, width = board.shape
    features = []
    
    # Per-column heights (always 10, padded with zeros for narrower boards)
    heights = get_column_heights(board) / height
    # Pad or truncate to exactly 10 columns
    if width < 10:
        heights = np.pad(heights, (0, 10 - width), mode='constant', constant_values=0)
    elif width > 10:
        heights = heights[:10]
    features.extend(heights)
    
    # Count-based features (positive penalties for deterministic cost functions)
    board_area = height * width
    
    # Holes: positive penalty for cost calculation
    holes_count = holes(board) / 100.0  # Normalized positive penalty
    features.append(holes_count)
    
    # Aggregate height: positive penalty
    agg_height = aggregate_height(board) / (height * width * 100.0)
    features.append(agg_height)
    
    # Bumpiness: positive penalty
    bump = bumpiness(board) / 100.0  # Normalized positive penalty
    features.append(bump)
    
    # Wells: positive penalty
    wells_count = wells(board) / 100.0
    features.append(wells_count)
    
    # Max height: positive penalty
    max_h = max_height(board) / height
    features.append(max_h)
    
    # Completed lines: negative cost (reward)
    completed = -completed_lines_potential(board) / 10.0  # Negative for reward
    features.append(completed)
    
    # Row transitions: positive penalty
    transitions = column_transitions(board) / (width * 100.0)
    features.append(transitions)
    
    return np.array(features, dtype=np.float32)


def board_to_features(board: np.ndarray) -> np.ndarray:
    """
    Legacy function for backward compatibility.
    Defaults to ML version for existing code.
    """
    return board_to_features_ml(board)


def reward_shaping(prev_board: np.ndarray, next_board: np.ndarray) -> float:
    """
    Calculate reward shaping bonus based on board improvements.
    
    Heavily penalize bad features and reward good features.
    """
    prev_holes = holes(prev_board)
    next_holes = holes(next_board)
    
    prev_bump = bumpiness(prev_board)
    next_bump = bumpiness(next_board)
    
    prev_height = aggregate_height(prev_board)
    next_height = aggregate_height(next_board)
    
    # Strong penalties and rewards for changes
    reward = 0.0
    
    # Heavy penalty for creating holes, big reward for filling them
    hole_change = next_holes - prev_holes
    reward -= hole_change * 5.0  # 5 points penalty per new hole
    
    # Penalty for increased bumpiness, reward for smoothing
    bump_change = next_bump - prev_bump
    reward -= bump_change * 1.0  # 1 point penalty per bump increase
    
    # Small penalty for height increase (encourage keeping low)
    height_change = next_height - prev_height
    reward -= height_change * 0.1
    
    return reward