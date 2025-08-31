"""
Afterstate enumeration for Tetris.
Enumerates all legal piece placements and their resulting board states.
"""
import numpy as np
from typing import List, Tuple, Dict, Any
from .tetris_env import TetrisEnv, TETRIS_PIECES


def enumerate_afterstates(env: TetrisEnv) -> List[Tuple[np.ndarray, Dict[str, Any]]]:
    """
    Enumerate all possible afterstates from current state.
    
    Returns list of (afterstate_board, action_info) tuples where:
    - afterstate_board: board state after placing and locking current piece
    - action_info: dict with placement details {'col': int, 'rotation': int, 'reward': float}
    """
    if env.game_over or env.current_piece is None:
        return []
    
    afterstates = []
    current_piece_name = env.current_piece_name
    
    # Try all rotations
    for rotation in range(4):
        piece = TETRIS_PIECES[current_piece_name][rotation]
        
        # Try all horizontal positions - only valid board positions to avoid left bias
        # Calculate actual piece width more accurately
        piece_cols = set()
        for i in range(4):
            for j in range(4):
                if piece[i, j]:
                    piece_cols.add(j)
        
        if not piece_cols:
            continue  # Empty piece, skip
            
        piece_left_offset = min(piece_cols)
        piece_right_offset = max(piece_cols) 
        piece_width = piece_right_offset - piece_left_offset + 1
        
        # Calculate valid column range - piece can be placed with its leftmost part at these positions
        min_col = -piece_left_offset  # Allow piece to align its filled parts with board
        max_col = env.width - piece_right_offset - 1
        
        for col in range(min_col, max_col + 1):
            
            # Find the lowest valid position for this placement
            valid_placement = False
            final_row = 0
            
            # Start from top and drop piece down
            for row in range(env.height + 4):  # Allow starting above board
                
                # Check if piece can be placed at this position
                collision = False
                piece_bounds = {'min_row': env.height, 'max_row': -1, 
                               'min_col': env.width, 'max_col': -1}
                
                for i in range(4):
                    for j in range(4):
                        if piece[i, j]:
                            board_row = row + i
                            board_col = col + j
                            
                            # Update piece bounds
                            piece_bounds['min_row'] = min(piece_bounds['min_row'], board_row)
                            piece_bounds['max_row'] = max(piece_bounds['max_row'], board_row)
                            piece_bounds['min_col'] = min(piece_bounds['min_col'], board_col)
                            piece_bounds['max_col'] = max(piece_bounds['max_col'], board_col)
                            
                            # Check boundaries - must be completely within board
                            if board_col < 0 or board_col >= env.width:
                                collision = True
                                break
                            
                            if board_row >= env.height:
                                collision = True
                                break
                            
                            # Check board collision (only if within bounds)
                            if board_row >= 0 and env.board[board_row, board_col]:
                                collision = True
                                break
                    
                    if collision:
                        break
                
                # If no collision, this is a valid position
                if not collision and piece_bounds['max_row'] >= 0:
                    valid_placement = True
                    final_row = row
                else:
                    # If we found a valid position before, this is where piece locks
                    if valid_placement:
                        break
            
            # Skip if piece never found valid position
            if not valid_placement:
                continue
            
            # Create afterstate by simulating piece lock
            afterstate_board = env.board.copy()
            
            # Place piece on board
            for i in range(4):
                for j in range(4):
                    if piece[i, j]:
                        board_row = final_row + i
                        board_col = col + j
                        
                        # Only place if within board bounds
                        if (0 <= board_row < env.height and 
                            0 <= board_col < env.width):
                            afterstate_board[board_row, board_col] = 1
            
            # Simulate line clears
            lines_to_clear = []
            for row in range(env.height):
                if np.all(afterstate_board[row, :]):
                    lines_to_clear.append(row)
            
            # Calculate reward for this placement
            reward = len(lines_to_clear) * 10  # Base line clear reward
            
            # Apply line clears
            if lines_to_clear:
                remaining_board = np.delete(afterstate_board, lines_to_clear, axis=0)
                empty_lines = np.zeros((len(lines_to_clear), env.width), dtype=np.int8)
                afterstate_board = np.vstack([empty_lines, remaining_board])
            
            # Add reward shaping
            from .features import reward_shaping
            reward += reward_shaping(env.board, afterstate_board)
            
            action_info = {
                'col': col,
                'rotation': rotation, 
                'reward': reward,
                'lines_cleared': len(lines_to_clear),
                'final_row': final_row,
                'after_board': afterstate_board.copy()  # Include the computed afterstate
            }
            
            afterstates.append((afterstate_board, action_info))
    
    return afterstates


def apply_afterstate(env: TetrisEnv, after_board: np.ndarray, action_info: Dict[str, Any]) -> bool:
    """
    Deterministic executor that sets env.board to after_board and updates game state.
    This is the single source of truth for applying placements.
    
    Args:
        env: Tetris environment to update
        after_board: Pre-computed board state after placement and line clears
        action_info: Action information with lines_cleared count
        
    Returns:
        True (should never fail if afterstate was properly enumerated)
    """
    # Apply the pre-computed afterstate board
    env.board = after_board.copy()
    
    # Update score based on lines cleared (if any)
    lines_cleared = action_info.get('lines_cleared', 0)
    if lines_cleared > 0:
        # Standard Tetris scoring: 40, 100, 300, 1200 for 1,2,3,4 lines
        line_scores = [0, 40, 100, 300, 1200]
        env.score += line_scores[min(lines_cleared, 4)]
        env.lines_cleared += lines_cleared
    else:
        # Small score for piece placement (soft drop points)
        env.score += 1
    
    # Spawn next piece (single place that spawns)
    if not env.game_over:
        env._spawn_piece()
        
        # Check if spawned piece immediately causes game over
        if env.current_piece is not None:
            # Check if new piece collides with board
            piece = env.current_piece
            for i in range(4):
                for j in range(4):
                    if piece[i, j]:
                        board_row = env.current_pos[0] + i
                        board_col = env.current_pos[1] + j
                        if (0 <= board_row < env.height and 
                            0 <= board_col < env.width and 
                            env.board[board_row, board_col]):
                            env.game_over = True
                            break
                if env.game_over:
                    break
    
    return True


def get_best_placement(env: TetrisEnv, weights: np.ndarray) -> Tuple[int, int, float]:
    """
    Get best piece placement using linear evaluation of features.
    
    Args:
        env: Tetris environment
        weights: Linear weights for feature evaluation
        
    Returns:
        (best_col, best_rotation, best_score) tuple
    """
    afterstates = enumerate_afterstates(env)
    
    if not afterstates:
        return 4, 0, float('-inf')  # Default to center column
    
    from .features import board_to_features
    
    best_score = float('-inf')
    best_col = 0
    best_rotation = 0
    
    for afterstate_board, action_info in afterstates:
        # Get features for afterstate
        features = board_to_features(afterstate_board)
        
        # Calculate linear score
        score = np.dot(weights, features) + action_info['reward']
        
        if score > best_score:
            best_score = score
            best_col = action_info['col']
            best_rotation = action_info['rotation']
    
    return best_col, best_rotation, best_score


def execute_placement(env: TetrisEnv, target_col: int, target_rotation: int) -> bool:
    """
    Execute a specific piece placement in the environment.
    
    Args:
        env: Tetris environment
        target_col: Target column position
        target_rotation: Target rotation
        
    Returns:
        True if placement was successful, False otherwise
    """
    if env.game_over:
        return False
    
    # Rotate to target rotation
    current_rotation = env.current_rotation
    rotation_attempts = 0
    max_rotation_attempts = 4  # Prevent infinite loops
    
    while current_rotation != target_rotation and rotation_attempts < max_rotation_attempts:
        if not env._rotate_piece():
            return False  # Rotation failed
        current_rotation = env.current_rotation
        rotation_attempts += 1
    
    # Move to target column - improved movement logic
    current_col = env.current_pos[1]
    movement_attempts = 0
    max_movement_attempts = 20  # Prevent infinite loops
    
    while current_col != target_col and movement_attempts < max_movement_attempts:
        if current_col < target_col:
            if not env._move_piece(0, 1):  # Move right
                break  # Can't move right anymore
            current_col += 1
        else:
            if not env._move_piece(0, -1):  # Move left  
                break  # Can't move left anymore
            current_col -= 1
        movement_attempts += 1
    
    # If we couldn't reach the exact target column, that's okay - 
    # the piece will be placed at the closest valid position
    
    # Drop piece
    env._drop_piece()
    
    return True