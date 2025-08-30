"""
Tests for FastAPI endpoints.
"""
import pytest
import asyncio
import json
import numpy as np
from fastapi.testclient import TestClient
import tempfile
import os
from unittest.mock import patch

# Import the FastAPI app
from app.main import app
from app.config import config
from rl.policy_store import save_policy


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


@pytest.fixture
def mock_policy():
    """Create a mock policy for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        policy_path = os.path.join(tmpdir, "test_policy.npz")
        
        # Create and save test policy
        weights = np.random.normal(0, 0.1, size=17).astype(np.float32)
        policy_dict = {'linear_weights': weights}
        metadata = {
            'algorithm': 'test',
            'created': 'test_fixture'
        }
        save_policy(policy_dict, policy_path, metadata)
        
        # Patch the config to use our test policy
        original_path = config.policy_path
        config.policy_path = policy_path
        
        # Force reload policy
        from app.main import load_policy
        load_policy()
        
        yield policy_path
        
        # Restore original config
        config.policy_path = original_path


def test_health_endpoint(client):
    """Test /api/health endpoint."""
    response = client.get("/api/health")
    
    assert response.status_code == 200
    data = response.json()
    
    assert "status" in data
    assert data["status"] == "ok"
    assert "train_enabled" in data
    assert "policy_loaded" in data
    assert isinstance(data["train_enabled"], bool)
    assert isinstance(data["policy_loaded"], bool)


def test_health_endpoint_structure(client):
    """Test health endpoint returns correct structure."""
    response = client.get("/api/health")
    data = response.json()
    
    # Check all required fields
    required_fields = ["status", "train_enabled", "policy_loaded"]
    for field in required_fields:
        assert field in data
    
    # Check types
    assert data["status"] == "ok"
    assert isinstance(data["train_enabled"], bool)
    assert isinstance(data["policy_loaded"], bool)


def test_play_endpoint_basic(client, mock_policy):
    """Test basic /api/play functionality."""
    request_data = {
        "episodes": 1,
        "seed": 42,
        "algo": "cem"
    }
    
    response = client.post("/api/play", json=request_data)
    
    assert response.status_code == 200
    data = response.json()
    
    # Check response structure
    required_fields = ["total_lines", "avg_score", "episodes", "scores", "lines_cleared", "episode_lengths"]
    for field in required_fields:
        assert field in data
    
    # Check data types and values
    assert isinstance(data["total_lines"], int)
    assert isinstance(data["avg_score"], (int, float))
    assert data["episodes"] == 1
    assert len(data["scores"]) == 1
    assert len(data["lines_cleared"]) == 1
    assert len(data["episode_lengths"]) == 1
    
    # Values should be non-negative
    assert data["total_lines"] >= 0
    assert data["avg_score"] >= 0
    assert data["scores"][0] >= 0
    assert data["lines_cleared"][0] >= 0
    assert data["episode_lengths"][0] >= 0


def test_play_endpoint_multiple_episodes(client, mock_policy):
    """Test /api/play with multiple episodes."""
    request_data = {
        "episodes": 3,
        "seed": 123,
        "algo": "cem"
    }
    
    response = client.post("/api/play", json=request_data)
    
    assert response.status_code == 200
    data = response.json()
    
    assert data["episodes"] == 3
    assert len(data["scores"]) == 3
    assert len(data["lines_cleared"]) == 3
    assert len(data["episode_lengths"]) == 3
    
    # Average score should match
    expected_avg = sum(data["scores"]) / len(data["scores"])
    assert abs(data["avg_score"] - expected_avg) < 1e-6
    
    # Total lines should match
    expected_total = sum(data["lines_cleared"])
    assert data["total_lines"] == expected_total


def test_play_endpoint_deterministic(client, mock_policy):
    """Test that /api/play is deterministic with same seed."""
    request_data = {
        "episodes": 2,
        "seed": 456,
        "algo": "cem"
    }
    
    # Make two identical requests
    response1 = client.post("/api/play", json=request_data)
    response2 = client.post("/api/play", json=request_data)
    
    assert response1.status_code == 200
    assert response2.status_code == 200
    
    data1 = response1.json()
    data2 = response2.json()
    
    # Results should be identical
    assert data1["scores"] == data2["scores"]
    assert data1["lines_cleared"] == data2["lines_cleared"]
    assert data1["episode_lengths"] == data2["episode_lengths"]


def test_play_endpoint_different_algorithms(client, mock_policy):
    """Test /api/play with different algorithms."""
    for algo in ["cem", "reinforce"]:
        request_data = {
            "episodes": 1,
            "seed": 789,
            "algo": algo
        }
        
        response = client.post("/api/play", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        assert isinstance(data["scores"], list)
        assert len(data["scores"]) == 1


def test_play_endpoint_validation(client, mock_policy):
    """Test /api/play request validation."""
    # Test invalid episodes count
    response = client.post("/api/play", json={"episodes": 0})
    assert response.status_code == 422  # Validation error
    
    response = client.post("/api/play", json={"episodes": 1000})
    assert response.status_code == 422  # Exceeds maximum
    
    # Test invalid algorithm
    response = client.post("/api/play", json={"episodes": 1, "algo": "invalid"})
    assert response.status_code == 422
    
    # Test missing required fields (should use defaults)
    response = client.post("/api/play", json={})
    assert response.status_code == 200  # Should use defaults


def test_play_endpoint_no_policy(client):
    """Test /api/play when no policy is loaded."""
    # Force policy to None
    with patch('app.main.policy_weights', None):
        response = client.post("/api/play", json={"episodes": 1})
        assert response.status_code == 500


def test_train_endpoint_disabled(client):
    """Test /api/train when training is disabled."""
    # Ensure training is disabled
    original_train_enabled = config.train_enabled
    config.train_enabled = False
    
    try:
        request_data = {
            "algo": "cem",
            "seed": 42,
            "generations": 2
        }
        
        response = client.post("/api/train", json=request_data)
        assert response.status_code == 403  # Forbidden
        
        data = response.json()
        assert "detail" in data
        assert "disabled" in data["detail"].lower()
        
    finally:
        config.train_enabled = original_train_enabled


@pytest.mark.skipif(not config.train_enabled, reason="Training disabled")
def test_train_endpoint_cem(client, mock_policy):
    """Test /api/train with CEM algorithm."""
    request_data = {
        "algo": "cem",
        "seed": 42,
        "generations": 2,
        "population_size": 10,
        "episodes_per_candidate": 1
    }
    
    # Mock the training to be fast
    with patch('app.main.cem_evolve') as mock_evolve:
        mock_evolve.return_value = {
            'best_fitness': 150.0,
            'best_weights': np.random.normal(0, 0.1, 17),
            'fitness_history': []
        }
        
        response = client.post("/api/train", json=request_data)
        
        if response.status_code == 403:
            pytest.skip("Training disabled in environment")
        
        assert response.status_code == 200
        data = response.json()
        
        # Check response structure
        required_fields = ["success", "message", "algo", "best_performance", "training_time"]
        for field in required_fields:
            assert field in data
        
        assert data["success"] is True
        assert data["algo"] == "cem"
        assert isinstance(data["best_performance"], (int, float))
        assert isinstance(data["training_time"], (int, float))


@pytest.mark.skipif(not config.train_enabled, reason="Training disabled")
def test_train_endpoint_reinforce(client, mock_policy):
    """Test /api/train with REINFORCE algorithm."""
    request_data = {
        "algo": "reinforce",
        "seed": 42,
        "episodes": 50,
        "learning_rate": 0.01
    }
    
    # Mock the training to be fast
    with patch('app.main.reinforce_train') as mock_train:
        mock_train.return_value = {
            'best_reward': 75.0,
            'episode_rewards': [10, 20, 30],
            'best_weights': np.random.normal(0, 0.1, 17)
        }
        
        response = client.post("/api/train", json=request_data)
        
        if response.status_code == 403:
            pytest.skip("Training disabled in environment")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["success"] is True
        assert data["algo"] == "reinforce"
        assert isinstance(data["best_performance"], (int, float))


def test_train_endpoint_validation(client):
    """Test /api/train request validation."""
    # Test invalid algorithm
    response = client.post("/api/train", json={"algo": "invalid"})
    assert response.status_code == 422
    
    # Test invalid parameters
    response = client.post("/api/train", json={"algo": "cem", "generations": 0})
    assert response.status_code == 422
    
    response = client.post("/api/train", json={"algo": "reinforce", "episodes": 0})
    assert response.status_code == 422


def test_websocket_stream_basic(client, mock_policy):
    """Test WebSocket stream endpoint basic functionality."""
    with client.websocket_connect("/ws/stream") as websocket:
        # Should receive at least one frame
        data = websocket.receive_json()
        
        # Check frame structure
        required_fields = ["frame", "lines", "score", "step"]
        for field in required_fields:
            assert field in data
        
        assert isinstance(data["frame"], str)  # Base64 encoded
        assert isinstance(data["lines"], int)
        assert isinstance(data["score"], (int, float))
        assert isinstance(data["step"], int)
        
        # Initial frame should have step 0
        assert data["step"] == 0
        assert data["lines"] >= 0
        assert data["score"] >= 0


def test_websocket_stream_sequence(client, mock_policy):
    """Test WebSocket stream produces sequence of frames."""
    with client.websocket_connect("/ws/stream") as websocket:
        frames = []
        max_frames = 5
        
        try:
            for _ in range(max_frames):
                data = websocket.receive_json()
                frames.append(data)
                
                # Stop if done
                if data.get("done", False):
                    break
                    
        except Exception:
            pass  # Connection might close
        
        # Should have received at least one frame
        assert len(frames) >= 1
        
        # Frames should have increasing step numbers
        for i in range(1, len(frames)):
            assert frames[i]["step"] >= frames[i-1]["step"]


def test_websocket_stream_no_policy(client):
    """Test WebSocket stream when no policy loaded."""
    # Force policy to None
    with patch('app.main.policy_weights', None):
        with client.websocket_connect("/ws/stream") as websocket:
            data = websocket.receive_json()
            assert "error" in data


def test_serve_index(client):
    """Test serving index.html."""
    response = client.get("/")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")


def test_endpoints_integration(client, mock_policy):
    """Integration test of multiple endpoints."""
    # 1. Check health
    health_response = client.get("/api/health")
    assert health_response.status_code == 200
    health_data = health_response.json()
    assert health_data["policy_loaded"] is True
    
    # 2. Play episodes
    play_response = client.post("/api/play", json={"episodes": 2, "seed": 42})
    assert play_response.status_code == 200
    play_data = play_response.json()
    assert play_data["episodes"] == 2
    
    # 3. Test WebSocket briefly
    with client.websocket_connect("/ws/stream") as websocket:
        frame = websocket.receive_json()
        assert "frame" in frame
        assert "score" in frame


def test_api_error_handling(client, mock_policy):
    """Test API error handling."""
    # Invalid JSON should return 422
    response = client.post("/api/play", data="invalid json")
    assert response.status_code == 422
    
    # Valid JSON with wrong content-type should still work (modern behavior)
    response = client.post("/api/play", data=json.dumps({"episodes": 1}))
    assert response.status_code == 200  # Should succeed with valid data
    
    # Invalid data should return 422
    response = client.post("/api/play", json={"episodes": 0})  # Below minimum
    assert response.status_code == 422


def test_play_endpoint_edge_cases(client, mock_policy):
    """Test edge cases for play endpoint."""
    # Maximum episodes
    response = client.post("/api/play", json={"episodes": 100})
    assert response.status_code == 200
    
    # Minimum episodes
    response = client.post("/api/play", json={"episodes": 1})
    assert response.status_code == 200
    
    # Null seed (should use random)
    response = client.post("/api/play", json={"episodes": 1, "seed": None})
    assert response.status_code == 200


def test_websocket_stream_fps_control(client, mock_policy):
    """Test that WebSocket stream respects FPS control."""
    import time
    
    with client.websocket_connect("/ws/stream") as websocket:
        start_time = time.time()
        frames = []
        
        try:
            # Collect a few frames
            for _ in range(3):
                data = websocket.receive_json()
                frames.append((time.time(), data))
                
                if data.get("done", False):
                    break
                    
        except Exception:
            pass
        
        if len(frames) >= 2:
            # Check timing between frames (allowing for test variance)
            expected_interval = 1.0 / config.stream_fps
            actual_interval = frames[1][0] - frames[0][0]
            
            # Should be roughly the expected interval (with tolerance for test environment)
            assert actual_interval >= expected_interval * 0.5  # Allow 50% variance


@pytest.mark.asyncio
async def test_websocket_disconnect_handling(client, mock_policy):
    """Test WebSocket disconnect handling."""
    # This test ensures the WebSocket endpoint handles disconnections gracefully
    with client.websocket_connect("/ws/stream") as websocket:
        # Receive first frame
        data = websocket.receive_json()
        assert "frame" in data
        
        # Close connection
        websocket.close()
    
    # Should not raise exception