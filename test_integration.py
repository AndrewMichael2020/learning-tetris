#!/usr/bin/env python3
"""
Simple integration test for the new algorithms.
Tests the agents in isolation without external dependencies.
"""
import sys
import os

# Add project root to path
sys.path.insert(0, '.')

def test_agents():
    """Test all four new agents."""
    print("Testing new Tetris AI algorithms...")
    
    try:
        # Import agents
        from rl.greedy_agent import GreedyAgent
        from rl.tabu_agent import TabuAgent  
        from rl.sa_agent import SimulatedAnnealingAgent
        from rl.aco_agent import ACOAgent
        print("✓ All agents imported successfully")
        
        # Test Greedy Agent
        print("\n--- Testing Greedy Agent (Nurse Dictator) ---")
        greedy = GreedyAgent(w_holes=8.0, w_max_height=1.0, w_bumpiness=1.0)
        stats = greedy.get_stats()
        print(f"Algorithm: {stats['algorithm']}")
        print(f"Weights: {stats['weights']}")
        
        # Test Tabu Agent
        print("\n--- Testing Tabu Agent (Nurse Gossip) ---")
        tabu = TabuAgent(tenure=25, neighborhood_top_k=10, aspiration=True)
        tabu.reset()
        stats = tabu.get_stats()
        print(f"Algorithm: {stats['algorithm']}")
        print(f"Tabu tenure: {stats['tabu_tenure']}")
        print(f"Tabu list size: {stats['tabu_list_size']}")
        
        # Test Simulated Annealing Agent
        print("\n--- Testing Simulated Annealing Agent (Coffee Break) ---")
        anneal = SimulatedAnnealingAgent(T0=10.0, alpha=0.99, proposal_top_k=10)
        anneal.reset()
        stats = anneal.get_stats()
        print(f"Algorithm: {stats['algorithm']}")
        print(f"Temperature: {stats['temperature']}")
        print(f"Cooling rate: {stats['cooling_rate']}")
        
        # Test ACO Agent
        print("\n--- Testing ACO Agent (Night Shift Ant March) ---")
        aco = ACOAgent(alpha=1.0, beta=2.0, rho=0.10, ants=20, elite=1)
        aco.reset()
        stats = aco.get_stats()
        print(f"Algorithm: {stats['algorithm']}")
        print(f"Number of ants: {stats['num_ants']}")
        print(f"Elite ants: {stats['elite_ants']}")
        
        print("\n✓ All agents tested successfully!")
        return True
        
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_search_utils():
    """Test search utilities."""
    print("\n--- Testing Search Utilities ---")
    
    try:
        from rl.search_utils import softmax, argmin_with_index, metropolis_accept
        
        # Test softmax
        probs = softmax([1.0, 2.0, 3.0])
        print(f"Softmax([1,2,3]) = {[f'{p:.3f}' for p in probs]}")
        assert abs(sum(probs) - 1.0) < 1e-6, "Softmax should sum to 1"
        
        # Test argmin
        idx, val = argmin_with_index([3.0, 1.0, 2.0])
        print(f"Argmin([3,1,2]) = index {idx}, value {val}")
        assert idx == 1 and val == 1.0, "Should find minimum at index 1"
        
        # Test metropolis acceptance
        accept1 = metropolis_accept(-1.0, 1.0)  # Improvement, should accept
        accept2 = metropolis_accept(1.0, 0.01)   # Worse at low temp, likely reject
        print(f"Metropolis(-1, T=1.0): {accept1}, Metropolis(1, T=0.01): {accept2}")
        
        print("✓ Search utilities working correctly")
        return True
        
    except Exception as e:
        print(f"✗ Search utils test failed: {e}")
        return False

if __name__ == "__main__":
    success = True
    success &= test_search_utils()
    success &= test_agents()
    
    if success:
        print("\n🎉 All tests passed! The new algorithms are ready to use.")
        exit(0)
    else:
        print("\n❌ Some tests failed.")
        exit(1)