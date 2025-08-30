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
    """Count transitions from filled to empty (or vice versa) within rows."""
    height, width = board.shape
    transitions = 0
    
    for row in range(height):
        prev_filled = True  # Consider walls as filled
        for col in range(width):
            current_filled = bool(board[row, col])
            if prev_filled != current_filled:
                transitions += 1
            prev_filled = current_filled
        # Transition to wall at end
        if not prev_filled:
            transitions += 1
    
    return transitions


def column_transitions(board: np.ndarray) -> int:
    """Count transitions from filled to empty (or vice versa) within columns."""
    height, width = board.shape
    transitions = 0
    
    for col in range(width):
        prev_filled = False  # Top is empty
        for row in range(height):
            current_filled = bool(board[row, col])
            if prev_filled != current_filled:
                transitions += 1
            prev_filled = current_filled
        # Transition to bottom wall
        if not prev_filled:
            transitions += 1
    
    return transitions


def board_to_features(board: np.ndarray) -> np.ndarray:
    """
    Convert board state to feature vector.
    
    Features (17 total):
    - Per-column heights (10)
    - Total holes (1)
    - Aggregate height (1) 
    - Bumpiness (1)
    - Wells (1)
    - Max height (1)
    - Completed lines (1)
    - Row transitions (1)
    
    All features are normalized to reasonable ranges.
    """
    height, width = board.shape
    features = []
    
    # Per-column heights (normalized by board height)
    heights = get_column_heights(board) / height
    features.extend(heights)
    
    # Count-based features (normalized by board area or reasonable maximums)
    board_area = height * width
    
    holes_count = holes(board) / (board_area * 0.5)  # Normalize by half board area
    features.append(holes_count)
    
    agg_height = aggregate_height(board) / (height * width)  # Normalize by max possible
    features.append(agg_height)
    
    bump = bumpiness(board) / (height * (width - 1))  # Normalize by max possible
    features.append(bump)
    
    wells_count = wells(board) / (height * width * 0.25)  # Normalize conservatively
    features.append(wells_count)
    
    max_h = max_height(board) / height  # Normalize by board height
    features.append(max_h)
    
    completed = completed_lines_potential(board) / height  # Normalize by board height
    features.append(completed)
    
    row_trans = row_transitions(board) / (height * (width + 1))  # Max transitions per row
    features.append(row_trans)
    
    return np.array(features, dtype=np.float32)


def reward_shaping(prev_board: np.ndarray, next_board: np.ndarray) -> float:
    """
    Calculate reward shaping bonus based on board improvements.
    
    Gives small positive rewards for:
    - Reducing holes
    - Reducing bumpiness
    - Reducing aggregate height
    """
    prev_holes = holes(prev_board)
    next_holes = holes(next_board)
    
    prev_bump = bumpiness(prev_board)
    next_bump = bumpiness(next_board)
    
    prev_height = aggregate_height(prev_board)
    next_height = aggregate_height(next_board)
    
    # Small bonuses for improvements
    reward = 0.0
    
    if next_holes < prev_holes:
        reward += 0.5 * (prev_holes - next_holes)
    
    if next_bump < prev_bump:
        reward += 0.1 * (prev_bump - next_bump)
    
    if next_height < prev_height:
        reward += 0.01 * (prev_height - next_height)
    
    return reward