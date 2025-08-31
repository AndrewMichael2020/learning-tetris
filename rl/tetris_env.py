"""
Tetris environment implementation with standard 10x20 board and 7 piece types.
Supports seeded random generation and provides standard gym-like interface.
"""
import numpy as np
from typing import Optional, Tuple, Dict, Any, List


# Tetris piece definitions (4x4 rotation matrices for each piece)
TETRIS_PIECES = {
    'I': [
        np.array([[0, 0, 0, 0],
                  [1, 1, 1, 1],
                  [0, 0, 0, 0],
                  [0, 0, 0, 0]], dtype=np.int8),
        np.array([[0, 0, 1, 0],
                  [0, 0, 1, 0],
                  [0, 0, 1, 0],
                  [0, 0, 1, 0]], dtype=np.int8),
        np.array([[0, 0, 0, 0],
                  [0, 0, 0, 0],
                  [1, 1, 1, 1],
                  [0, 0, 0, 0]], dtype=np.int8),
        np.array([[0, 1, 0, 0],
                  [0, 1, 0, 0],
                  [0, 1, 0, 0],
                  [0, 1, 0, 0]], dtype=np.int8)
    ],
    'O': [
        np.array([[0, 0, 0, 0],
                  [0, 1, 1, 0],
                  [0, 1, 1, 0],
                  [0, 0, 0, 0]], dtype=np.int8)
    ] * 4,  # O piece same in all rotations
    'T': [
        np.array([[0, 0, 0, 0],
                  [0, 1, 0, 0],
                  [1, 1, 1, 0],
                  [0, 0, 0, 0]], dtype=np.int8),
        np.array([[0, 0, 0, 0],
                  [0, 1, 0, 0],
                  [0, 1, 1, 0],
                  [0, 1, 0, 0]], dtype=np.int8),
        np.array([[0, 0, 0, 0],
                  [0, 0, 0, 0],
                  [1, 1, 1, 0],
                  [0, 1, 0, 0]], dtype=np.int8),
        np.array([[0, 0, 0, 0],
                  [0, 1, 0, 0],
                  [1, 1, 0, 0],
                  [0, 1, 0, 0]], dtype=np.int8)
    ],
    'S': [
        np.array([[0, 0, 0, 0],
                  [0, 1, 1, 0],
                  [1, 1, 0, 0],
                  [0, 0, 0, 0]], dtype=np.int8),
        np.array([[0, 0, 0, 0],
                  [0, 1, 0, 0],
                  [0, 1, 1, 0],
                  [0, 0, 1, 0]], dtype=np.int8),
        np.array([[0, 0, 0, 0],
                  [0, 0, 0, 0],
                  [0, 1, 1, 0],
                  [1, 1, 0, 0]], dtype=np.int8),
        np.array([[0, 0, 0, 0],
                  [1, 0, 0, 0],
                  [1, 1, 0, 0],
                  [0, 1, 0, 0]], dtype=np.int8)
    ],
    'Z': [
        np.array([[0, 0, 0, 0],
                  [1, 1, 0, 0],
                  [0, 1, 1, 0],
                  [0, 0, 0, 0]], dtype=np.int8),
        np.array([[0, 0, 0, 0],
                  [0, 0, 1, 0],
                  [0, 1, 1, 0],
                  [0, 1, 0, 0]], dtype=np.int8),
        np.array([[0, 0, 0, 0],
                  [0, 0, 0, 0],
                  [1, 1, 0, 0],
                  [0, 1, 1, 0]], dtype=np.int8),
        np.array([[0, 0, 0, 0],
                  [0, 1, 0, 0],
                  [1, 1, 0, 0],
                  [1, 0, 0, 0]], dtype=np.int8)
    ],
    'J': [
        np.array([[0, 0, 0, 0],
                  [1, 0, 0, 0],
                  [1, 1, 1, 0],
                  [0, 0, 0, 0]], dtype=np.int8),
        np.array([[0, 0, 0, 0],
                  [0, 1, 1, 0],
                  [0, 1, 0, 0],
                  [0, 1, 0, 0]], dtype=np.int8),
        np.array([[0, 0, 0, 0],
                  [0, 0, 0, 0],
                  [1, 1, 1, 0],
                  [0, 0, 1, 0]], dtype=np.int8),
        np.array([[0, 0, 0, 0],
                  [0, 1, 0, 0],
                  [0, 1, 0, 0],
                  [1, 1, 0, 0]], dtype=np.int8)
    ],
    'L': [
        np.array([[0, 0, 0, 0],
                  [0, 0, 1, 0],
                  [1, 1, 1, 0],
                  [0, 0, 0, 0]], dtype=np.int8),
        np.array([[0, 0, 0, 0],
                  [0, 1, 0, 0],
                  [0, 1, 0, 0],
                  [0, 1, 1, 0]], dtype=np.int8),
        np.array([[0, 0, 0, 0],
                  [0, 0, 0, 0],
                  [1, 1, 1, 0],
                  [1, 0, 0, 0]], dtype=np.int8),
        np.array([[0, 0, 0, 0],
                  [1, 1, 0, 0],
                  [0, 1, 0, 0],
                  [0, 1, 0, 0]], dtype=np.int8)
    ]
}

PIECE_NAMES = ['I', 'O', 'T', 'S', 'Z', 'J', 'L']


class TetrisEnv:
    """
    10x20 Tetris environment with standard rules and pieces.
    
    Actions: 0=left, 1=right, 2=rotate, 3=drop, 4=soft_drop, 5=noop
    """
    
    def __init__(self, width: int = 10, height: int = 20):
        self.width = width
        self.height = height
        self.board = None
        self.current_piece = None
        self.current_piece_name = None
        self.current_rotation = 0
        self.current_pos = [0, 0]  # [row, col]
        self.score = 0
        self.lines_cleared = 0
        self.step_count = 0
        self.game_over = False
        self.rng = None
        self.piece_bag = []
        self.bag_index = 0
        
    def copy(self):
        """Create a deep copy of the environment state."""
        import copy as copy_module
        
        env_copy = TetrisEnv(self.width, self.height)
        env_copy.board = self.board.copy() if self.board is not None else None
        env_copy.current_piece = self.current_piece.copy() if self.current_piece is not None else None
        env_copy.current_piece_name = self.current_piece_name
        env_copy.current_rotation = self.current_rotation
        env_copy.current_pos = self.current_pos.copy()
        env_copy.score = self.score
        env_copy.lines_cleared = self.lines_cleared
        env_copy.step_count = self.step_count
        env_copy.game_over = self.game_over
        
        # Copy RNG state
        if self.rng is not None:
            env_copy.rng = copy_module.deepcopy(self.rng)
        
        # Copy piece bag
        env_copy.piece_bag = self.piece_bag.copy()
        env_copy.bag_index = self.bag_index
        
        return env_copy
        
    def reset(self, seed: Optional[int] = None) -> np.ndarray:
        """Reset environment and return initial board state."""
        self.rng = np.random.default_rng(seed)
        self.board = np.zeros((self.height, self.width), dtype=np.int8)
        self.score = 0
        self.lines_cleared = 0
        self.step_count = 0
        self.game_over = False
        self.piece_bag = []
        self.bag_index = 0
        
        # Spawn first piece
        self._spawn_piece()
        
        return self.board.copy()
    
    def _generate_7_bag(self) -> List[str]:
        """Generate a 7-bag of pieces (one of each type in random order)."""
        bag = PIECE_NAMES.copy()
        self.rng.shuffle(bag)
        return bag
    
    def _get_next_piece(self) -> str:
        """Get next piece from 7-bag system."""
        if not self.piece_bag or self.bag_index >= len(self.piece_bag):
            self.piece_bag = self._generate_7_bag()
            self.bag_index = 0
        
        piece_name = self.piece_bag[self.bag_index]
        self.bag_index += 1
        return piece_name
    
    def _spawn_piece(self) -> bool:
        """Spawn new piece at top center. Returns False if game over."""
        self.current_piece_name = self._get_next_piece()
        self.current_rotation = 0
        self.current_piece = TETRIS_PIECES[self.current_piece_name][self.current_rotation]
        self.current_pos = [0, self.width // 2 - 2]  # Start at top center
        
        # Check if spawn position is valid
        if self._check_collision():
            self.game_over = True
            return False
        
        return True
    
    def _check_collision(self, piece: Optional[np.ndarray] = None, 
                        pos: Optional[List[int]] = None) -> bool:
        """Check if piece collides with board or boundaries."""
        if piece is None:
            piece = self.current_piece
        if pos is None:
            pos = self.current_pos
            
        for i in range(4):
            for j in range(4):
                if piece[i, j]:
                    board_row = pos[0] + i
                    board_col = pos[1] + j
                    
                    # Check boundaries
                    if (board_row < 0 or board_row >= self.height or
                        board_col < 0 or board_col >= self.width):
                        return True
                    
                    # Check board collision
                    if self.board[board_row, board_col]:
                        return True
        
        return False
    
    def _lock_piece(self) -> int:
        """Lock current piece to board and return lines cleared."""
        # Place piece on board
        for i in range(4):
            for j in range(4):
                if self.current_piece[i, j]:
                    board_row = self.current_pos[0] + i
                    board_col = self.current_pos[1] + j
                    if 0 <= board_row < self.height and 0 <= board_col < self.width:
                        self.board[board_row, board_col] = 1
        
        # Check for line clears
        lines_to_clear = []
        for row in range(self.height):
            if np.all(self.board[row, :]):
                lines_to_clear.append(row)
        
        # Clear lines
        if lines_to_clear:
            # Remove cleared lines and add empty lines at top
            remaining_board = np.delete(self.board, lines_to_clear, axis=0)
            empty_lines = np.zeros((len(lines_to_clear), self.width), dtype=np.int8)
            self.board = np.vstack([empty_lines, remaining_board])
        
        return len(lines_to_clear)
    
    def _move_piece(self, delta_row: int, delta_col: int) -> bool:
        """Try to move piece by delta. Returns True if successful."""
        new_pos = [self.current_pos[0] + delta_row, self.current_pos[1] + delta_col]
        
        if not self._check_collision(pos=new_pos):
            self.current_pos = new_pos
            return True
        return False
    
    def _rotate_piece(self) -> bool:
        """Try to rotate piece clockwise. Returns True if successful."""
        new_rotation = (self.current_rotation + 1) % 4
        new_piece = TETRIS_PIECES[self.current_piece_name][new_rotation]
        
        if not self._check_collision(piece=new_piece):
            self.current_rotation = new_rotation
            self.current_piece = new_piece
            return True
        return False
    
    def _calculate_score(self, lines_cleared: int) -> int:
        """Calculate score based on lines cleared using standard Tetris scoring."""
        if lines_cleared == 0:
            return 0
        elif lines_cleared == 1:
            return 100  # Single
        elif lines_cleared == 2:
            return 300  # Double  
        elif lines_cleared == 3:
            return 500  # Triple
        elif lines_cleared == 4:
            return 800  # Tetris (4 lines)
        else:
            # More than 4 lines (rare edge case)
            return 800 + (lines_cleared - 4) * 200
    
    def _drop_piece(self) -> bool:
        """Drop piece to bottom instantly. Returns True if piece locked."""
        while not self._check_collision(pos=[self.current_pos[0] + 1, self.current_pos[1]]):
            self.current_pos[0] += 1
        
        # Lock piece
        lines_cleared = self._lock_piece()
        self.lines_cleared += lines_cleared
        
        # Use improved scoring system
        score_gained = self._calculate_score(lines_cleared)
        self.score += score_gained
        
        # Add small bonus for piece placement to reward longer games
        self.score += 1  # 1 point per piece placed
        
        return True
    
    def legal_actions(self) -> List[int]:
        """Return list of legal actions in current state."""
        if self.game_over:
            return []
        
        actions = [5]  # noop always legal
        
        # Check left move
        if not self._check_collision(pos=[self.current_pos[0], self.current_pos[1] - 1]):
            actions.append(0)
        
        # Check right move  
        if not self._check_collision(pos=[self.current_pos[0], self.current_pos[1] + 1]):
            actions.append(1)
        
        # Check rotation
        new_rotation = (self.current_rotation + 1) % 4
        new_piece = TETRIS_PIECES[self.current_piece_name][new_rotation]
        if not self._check_collision(piece=new_piece):
            actions.append(2)
        
        # Drop and soft drop always legal (will lock piece if at bottom)
        actions.extend([3, 4])
        
        return actions
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """Execute action and return (obs, reward, done, info)."""
        if self.game_over:
            return self.board.copy(), 0.0, True, self._get_info()
        
        reward = -0.1  # Small negative step cost
        self.step_count += 1
        
        # Execute action
        if action == 0:  # left
            self._move_piece(0, -1)
        elif action == 1:  # right
            self._move_piece(0, 1)
        elif action == 2:  # rotate
            self._rotate_piece()
        elif action == 3:  # drop
            self._drop_piece()
            if not self._spawn_piece():
                self.game_over = True
        elif action == 4:  # soft drop
            if self._move_piece(1, 0):
                reward += 0.1  # Small bonus for soft drop
            else:
                # Can't move down, lock piece
                lines_cleared = self._lock_piece()
                self.lines_cleared += lines_cleared
                
                # Use improved scoring system
                score_gained = self._calculate_score(lines_cleared)
                self.score += score_gained
                reward += score_gained  # Line clear reward
                
                # Add small bonus for piece placement
                self.score += 1  # 1 point per piece placed
                reward += 1
                
                if not self._spawn_piece():
                    self.game_over = True
        # action == 5 is noop, do nothing
        
        return self.board.copy(), reward, self.game_over, self._get_info()
    
    def _get_info(self) -> Dict[str, Any]:
        """Get info dict with game statistics."""
        # Calculate additional stats
        holes = self._count_holes()
        height = self._get_aggregate_height()
        
        return {
            'lines_cleared': self.lines_cleared,
            'score': self.score,
            'holes': holes,
            'height': height,
            'step_count': self.step_count
        }
    
    def _count_holes(self) -> int:
        """Count number of holes in the board."""
        holes = 0
        for col in range(self.width):
            filled_found = False
            for row in range(self.height):
                if self.board[row, col]:
                    filled_found = True
                elif filled_found and not self.board[row, col]:
                    holes += 1
        return holes
    
    def _get_aggregate_height(self) -> int:
        """Get sum of column heights."""
        height = 0
        for col in range(self.width):
            for row in range(self.height):
                if self.board[row, col]:
                    height += self.height - row
                    break
        return height
    
    def render(self, mode: str = "rgb_array") -> np.ndarray:
        """Render board as RGB array."""
        if mode != "rgb_array":
            raise ValueError("Only rgb_array mode supported")
        
        # Create RGB image (height, width, 3)
        img = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        
        # Fill board with colors
        for row in range(self.height):
            for col in range(self.width):
                if self.board[row, col]:
                    img[row, col] = [255, 255, 255]  # White for filled
                else:
                    img[row, col] = [0, 0, 0]  # Black for empty
        
        # Draw current piece if game not over
        if not self.game_over and self.current_piece is not None:
            for i in range(4):
                for j in range(4):
                    if self.current_piece[i, j]:
                        board_row = self.current_pos[0] + i
                        board_col = self.current_pos[1] + j
                        if (0 <= board_row < self.height and 
                            0 <= board_col < self.width):
                            img[board_row, board_col] = [128, 128, 128]  # Gray for current piece
        
        return img