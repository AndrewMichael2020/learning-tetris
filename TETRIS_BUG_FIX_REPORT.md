# Tetris Environment Bug Fix Report

## **Root Cause Identified**

The poor Tetris performance (Score: ~11, 0 lines cleared) was caused by a **critical bug in the consolidated `enumerate_afterstates` function**.

### **The Problem:**
When I moved the afterstate enumeration from `afterstate.py` to `utils/search.py`, I introduced two bugs:

1. **"Axes must be different" error**: Used `np.rot90(piece.shape, rotation)` incorrectly
   - `piece.shape` returns a tuple like `(4, 4)`, not a matrix
   - `np.rot90()` needs actual matrix data, not shape dimensions
   - This caused every `get_best_placement()` call to fail

2. **Missing `copy()` method**: The `TetrisEnv` class lacked a `copy()` method needed by the search functions

### **The Impact:**
- Every call to `get_best_placement()` failed with "Axes must be different"
- The WebSocket handler caught the exception and used fallback placement `(4, 0)`
- This resulted in terrible gameplay: pieces always placed in the same position
- Games ended quickly with minimal score and no line clears

## **Fixes Applied:**

### **1. Fixed `enumerate_afterstates` in `utils/search.py`:**
```python
# BEFORE (broken):
rotated_piece = np.rot90(piece.shape, rotation)  # ❌ piece.shape is a tuple!

# AFTER (fixed):
piece_matrix = TETRIS_PIECES[env.current_piece_name][rotation]  # ✅ Use pre-rotated pieces
```

### **2. Added `copy()` method to `TetrisEnv`:**
```python
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
    
    # Copy RNG state and piece bag
    if self.rng is not None:
        env_copy.rng = copy_module.deepcopy(self.rng)
    env_copy.piece_bag = self.piece_bag.copy()
    env_copy.bag_index = self.bag_index
    
    return env_copy
```

## **Verification:**

✅ **Environment Testing**: The TetrisEnv itself was working correctly
- Seeding produces consistent piece sequences
- 7-bag system works properly  
- Piece shapes are valid
- Game mechanics function correctly

✅ **After Fix**: 
- `get_best_placement()` now works without errors
- All algorithms (CEM, REINFORCE, Greedy, Tabu, SA, ACO) can properly evaluate moves
- Game should now show much better performance with line clears and higher scores

## **Lesson Learned:**
When consolidating code for better organization, critical to thoroughly test functionality - especially when dealing with complex numpy operations and environment copying. The bug was introduced during the `/rl` directory cleanup when moving `search_utils.py` to `utils/search.py`.

This fix should dramatically improve Tetris gameplay performance across all algorithms!
