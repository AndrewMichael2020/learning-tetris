"""
Cross-Entropy Method (CEM) agent for Tetris.
Uses population-based search over linear feature weights.
"""
import numpy as np
from typing import Callable, Dict, List, Any, Optional
from .tetris_env import TetrisEnv
from .features import board_to_features_ml
from .afterstate import get_best_placement, execute_placement
from .policy_store import save_policy


class CEMPolicy:
    """Linear policy for Tetris using feature weights."""
    
    def __init__(self, weights: np.ndarray):
        """
        Initialize policy with feature weights.
        
        Args:
            weights: Linear weights for features (shape: [feature_dim])
        """
        self.weights = weights.copy()
        
    def predict(self, board: np.ndarray, legal_actions: List[int]) -> int:
        """
        Predict best action for current board state.
        This is unused for afterstate-based policies, but kept for interface compatibility.
        """
        # For afterstate policies, we don't use action-based prediction
        return legal_actions[0] if legal_actions else 5  # Default to first action or noop
        
    def get_best_placement(self, env: TetrisEnv) -> tuple[int, int, float]:
        """Get best piece placement using feature evaluation."""
        return get_best_placement(env, self.weights)


def evaluate_candidate(weights: np.ndarray, env_factory: Callable[[], TetrisEnv], 
                      episodes: int, seed: int) -> float:
    """
    Evaluate a candidate policy over multiple episodes.
    
    Args:
        weights: Policy weights to evaluate
        env_factory: Function that creates a new environment
        episodes: Number of episodes to run
        seed: Random seed
        
    Returns:
        Average fitness score across episodes
    """
    rng = np.random.default_rng(seed)
    policy = CEMPolicy(weights)
    
    total_score = 0.0
    total_lines = 0
    
    for episode in range(episodes):
        env = env_factory()
        env.reset(seed=rng.integers(0, 1000000))
        
        episode_reward = 0.0
        steps = 0
        max_steps = 1000  # Prevent infinite episodes
        
        while not env.game_over and steps < max_steps:
            # Get best placement using afterstate evaluation
            best_col, best_rotation, best_score = policy.get_best_placement(env)
            
            # Execute placement
            prev_score = env.score
            prev_lines = env.lines_cleared
            
            success = execute_placement(env, best_col, best_rotation)
            
            if not success:
                break  # Something went wrong
            
            # Calculate reward for this placement
            reward = (env.score - prev_score) + (env.lines_cleared - prev_lines) * 10
            episode_reward += reward
            steps += 1
            
            # Spawn new piece
            if not env.game_over:
                env._spawn_piece()
        
        # Fitness includes score and lines cleared with bonuses
        fitness = env.score + env.lines_cleared * 100
        total_score += fitness
        total_lines += env.lines_cleared
    
    return total_score / episodes if episodes > 0 else 0.0


def evolve(env_factory: Callable[[], TetrisEnv], generations: int = 10, 
           seed: int = 42, out_path: str = "policies/best.npz",
           episodes_per_candidate: int = 3, population_size: int = 50,
           elite_fraction: float = 0.2, feature_dim: int = 17) -> Dict[str, Any]:
    """
    Evolve CEM policy over multiple generations.
    
    Args:
        env_factory: Function that creates TetrisEnv instances
        generations: Number of generations to evolve
        seed: Random seed
        out_path: Path to save best policy
        episodes_per_candidate: Episodes to evaluate each candidate
        population_size: Size of population per generation
        elite_fraction: Fraction of population to keep as elites
        feature_dim: Dimension of feature vector
        
    Returns:
        Dictionary with training metrics and final policy
    """
    rng = np.random.default_rng(seed)
    n_elite = max(1, int(population_size * elite_fraction))
    
    # Initialize population with random weights
    mean = np.zeros(feature_dim, dtype=np.float32)
    # Even wider exploration for feature discrimination
    std = np.ones(feature_dim, dtype=np.float32) * 2.0
    
    best_fitness = float('-inf')
    best_weights = None
    fitness_history = []
    
    print(f"Starting CEM evolution: {generations} generations, {population_size} population")
    
    for generation in range(generations):
        # Generate population
        population = []
        for i in range(population_size):
            candidate = rng.normal(mean, std).astype(np.float32)
            # Allow larger weights for stronger feature discrimination
            candidate = np.clip(candidate, -5.0, 5.0).astype(np.float32)
            population.append(candidate)
        
        # Evaluate population
        fitness_scores = []
        for i, candidate in enumerate(population):
            candidate_seed = rng.integers(0, 1000000)
            fitness = evaluate_candidate(candidate, env_factory, 
                                       episodes_per_candidate, candidate_seed)
            fitness_scores.append(fitness)
            
            # Track best candidate
            if fitness > best_fitness:
                best_fitness = fitness
                best_weights = candidate.copy()
        
        # Sort by fitness (descending)
        sorted_indices = np.argsort(fitness_scores)[::-1]
        elite_indices = sorted_indices[:n_elite]
        
        # Update mean and std from elites
        elite_weights = np.array([population[i] for i in elite_indices])
        mean = np.mean(elite_weights, axis=0)
        # Maintain healthy exploration but avoid collapse/explosion
        std = np.clip(np.std(elite_weights, axis=0), 0.15, 2.5).astype(np.float32)
        
        # Record metrics
        gen_best = fitness_scores[sorted_indices[0]]
        gen_mean = np.mean(fitness_scores)
        gen_std = np.std(fitness_scores)
        
        fitness_history.append({
            'generation': generation,
            'best_fitness': gen_best,
            'mean_fitness': gen_mean,
            'std_fitness': gen_std,
            'population_std': np.mean(std)
        })
        
        print(f"Gen {generation+1}/{generations}: Best={gen_best:.1f}, "
              f"Mean={gen_mean:.1f}, Std={gen_std:.1f}")
        
        # Decay standard deviation over time (slightly faster for convergence)
        std *= 0.90
    
    # Save best policy
    if best_weights is not None:
        policy_dict = {'linear_weights': best_weights}
        metadata = {
            'algorithm': 'CEM',
            'generations': generations,
            'population_size': population_size,
            'episodes_per_candidate': episodes_per_candidate,
            'best_fitness': float(best_fitness),
            'seed': seed
        }
        save_policy(policy_dict, out_path, metadata)
        print(f"Saved best policy to {out_path} (fitness: {best_fitness:.1f})")
    
    return {
        'best_fitness': best_fitness,
        'best_weights': best_weights,
        'fitness_history': fitness_history,
        'final_mean': mean,
        'final_std': std
    }


async def evolve_with_progress(env_factory: Callable[[], TetrisEnv], generations: int = 10, 
                              seed: int = 42, out_path: str = "policies/best.npz",
                              episodes_per_candidate: int = 3, population_size: int = 50,
                              elite_fraction: float = 0.2, feature_dim: int = 17,
                              progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
    """
    Async version of evolve with progress callback support.
    
    Args:
        env_factory: Function that creates TetrisEnv instances
        generations: Number of generations to evolve
        seed: Random seed
        out_path: Path to save best policy
        episodes_per_candidate: Episodes to evaluate each candidate
        population_size: Size of population per generation
        elite_fraction: Fraction of population to keep as elites
        feature_dim: Dimension of feature vector
        progress_callback: Optional async callback for progress updates
        
    Returns:
        Dictionary with training metrics and final policy
    """
    import asyncio
    
    rng = np.random.default_rng(seed)
    n_elite = max(1, int(population_size * elite_fraction))
    
    # Initialize population with random weights
    mean = np.zeros(feature_dim, dtype=np.float32)
    std = np.ones(feature_dim, dtype=np.float32) * 2.0
    
    best_fitness = float('-inf')
    best_weights = None
    fitness_history = []
    
    print(f"Starting async CEM evolution: {generations} generations, {population_size} population")
    
    for generation in range(generations):
        # Generate population
        population = []
        for i in range(population_size):
            candidate = rng.normal(mean, std).astype(np.float32)
            candidate = np.clip(candidate, -5.0, 5.0).astype(np.float32)
            population.append(candidate)
        
        # Evaluate population
        fitness_scores = []
        for i, candidate in enumerate(population):
            candidate_seed = rng.integers(0, 1000000)
            fitness = evaluate_candidate(candidate, env_factory, 
                                       episodes_per_candidate, candidate_seed)
            fitness_scores.append(fitness)
            
            # Track best candidate
            if fitness > best_fitness:
                best_fitness = fitness
                best_weights = candidate.copy()
        
        # Sort by fitness (descending)
        sorted_indices = np.argsort(fitness_scores)[::-1]
        elite_indices = sorted_indices[:n_elite]
        
        # Update mean and std from elites
        elite_weights = np.array([population[i] for i in elite_indices])
        mean = np.mean(elite_weights, axis=0)
        std = np.clip(np.std(elite_weights, axis=0), 0.15, 2.5).astype(np.float32)
        
        # Record metrics
        gen_best = fitness_scores[sorted_indices[0]]
        gen_mean = np.mean(fitness_scores)
        gen_std = np.std(fitness_scores)
        
        fitness_history.append({
            'generation': generation,
            'best_fitness': gen_best,
            'mean_fitness': gen_mean,
            'std_fitness': gen_std,
            'population_std': np.mean(std)
        })
        
        print(f"Gen {generation+1}/{generations}: Best={gen_best:.1f}, "
              f"Mean={gen_mean:.1f}, Std={gen_std:.1f}")
        
        # Send progress update
        if progress_callback:
            should_continue = await progress_callback(generation, gen_best, fitness_scores)
            if not should_continue:
                print("Training cancelled by progress callback")
                break
        
        # Allow event loop to process other tasks
        await asyncio.sleep(0.01)
        
        # Decay standard deviation over time
        std *= 0.90
    
    # Save best policy
    if best_weights is not None:
        policy_dict = {'linear_weights': best_weights}
        metadata = {
            'algorithm': 'CEM',
            'generations': generations,
            'population_size': population_size,
            'episodes_per_candidate': episodes_per_candidate,
            'best_fitness': float(best_fitness),
            'seed': seed
        }
        save_policy(policy_dict, out_path, metadata)
        print(f"Saved best policy to {out_path} (fitness: {best_fitness:.1f})")
    
    return {
        'best_fitness': best_fitness,
        'best_weights': best_weights,
        'fitness_history': fitness_history,
        'final_mean': mean,
        'final_std': std
    }


async def evolve_with_progress(env_factory, generations=10, seed=42, out_path="policies/best.npz",
                              episodes_per_candidate=3, population_size=50, elite_fraction=0.2, 
                              feature_dim=17, progress_callback=None):
    """Async version of evolve with progress callbacks and cancellation support."""
    import asyncio
    
    rng = np.random.default_rng(seed)
    n_elite = max(1, int(population_size * elite_fraction))
    
    mean = np.zeros(feature_dim, dtype=np.float32)
    std = np.ones(feature_dim, dtype=np.float32) * 2.0
    
    best_fitness = float('-inf')
    best_weights = None
    fitness_history = []
    
    print(f"Starting async CEM evolution: {generations} generations, {population_size} population")
    
    for generation in range(generations):
        # Generate population
        population = []
        for i in range(population_size):
            candidate = rng.normal(mean, std).astype(np.float32)
            candidate = np.clip(candidate, -5.0, 5.0).astype(np.float32)
            population.append(candidate)
        
        # Evaluate population
        fitness_scores = []
        for i, candidate in enumerate(population):
            # Check for cancellation before evaluating each candidate
            if progress_callback:
                should_continue = await progress_callback(generation, best_fitness, fitness_scores)
                if not should_continue:
                    print(f"Training cancelled during candidate evaluation {i+1}/{population_size}")
                    return {
                        'best_fitness': best_fitness,
                        'best_weights': best_weights,
                        'fitness_history': fitness_history,
                        'final_mean': mean,
                        'final_std': std
                    }
            
            # Add more frequent cancellation checks during candidate evaluation
            candidate_seed = rng.integers(0, 1000000)
            
            # Yield control more frequently during evaluation
            if i % 2 == 0:  # Check every 2 candidates
                await asyncio.sleep(0.01)
                
            fitness = evaluate_candidate(candidate, env_factory, 
                                       episodes_per_candidate, candidate_seed)
            fitness_scores.append(fitness)
            
            if fitness > best_fitness:
                best_fitness = fitness
                best_weights = candidate.copy()
                
            # Allow other async operations more frequently
            if i % 5 == 0:  # Every 5 candidates
                await asyncio.sleep(0.01)
                
                # Additional cancellation check after fitness evaluation
                if progress_callback:
                    should_continue = await progress_callback(generation, best_fitness, fitness_scores)
                    if not should_continue:
                        print(f"Training cancelled during candidate evaluation {i+1}/{population_size}")
                        return {
                            'best_fitness': best_fitness,
                            'best_weights': best_weights,
                            'fitness_history': fitness_history,
                            'final_mean': mean,
                            'final_std': std
                        }
        
        # Update distribution from elites
        sorted_indices = np.argsort(fitness_scores)[::-1]
        elite_indices = sorted_indices[:n_elite]
        elite_weights = np.array([population[i] for i in elite_indices])
        mean = np.mean(elite_weights, axis=0)
        std = np.clip(np.std(elite_weights, axis=0), 0.15, 2.5).astype(np.float32)
        
        # Record metrics
        gen_best = fitness_scores[sorted_indices[0]]
        fitness_history.append({
            'generation': generation,
            'best_fitness': gen_best,
            'mean_fitness': np.mean(fitness_scores),
            'std_fitness': np.std(fitness_scores),
            'population_std': np.mean(std)
        })
        
        print(f"Gen {generation+1}/{generations}: Best={gen_best:.1f}")
        
        # Call progress callback if provided
        if progress_callback:
            should_continue = await progress_callback(generation, gen_best, fitness_scores)
            if not should_continue:
                print("Training cancelled by progress callback")
                break
        
        std *= 0.90
        await asyncio.sleep(0.01)  # Allow other tasks
    
    # Save best policy
    if best_weights is not None:
        policy_dict = {'linear_weights': best_weights}
        metadata = {
            'algorithm': 'CEM',
            'generations': generations,
            'population_size': population_size,
            'episodes_per_candidate': episodes_per_candidate,
            'best_fitness': float(best_fitness),
            'seed': seed
        }
        save_policy(policy_dict, out_path, metadata)
        print(f"Saved best policy to {out_path} (fitness: {best_fitness:.1f})")
    
    return {
        'best_fitness': best_fitness,
        'best_weights': best_weights,
        'fitness_history': fitness_history,
        'final_mean': mean,
        'final_std': std
    }
