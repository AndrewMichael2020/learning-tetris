# RL Tetris Web App

A reinforcement learning Tetris web application built with Python, FastAPI, and NumPy.

## Features

- 10x20 Tetris environment with standard pieces
- Reinforcement learning agents:
  - Cross-Entropy Method (CEM) 
  - REINFORCE with baseline
- Feature-based afterstate evaluation
- FastAPI web interface with WebSocket streaming
- Comprehensive test suite with 95%+ coverage

## Quick Start

```bash
# Install dependencies
python -m venv .venv && source .venv/bin/activate
pip install -e .

# Run tests
pytest -q

# Start web server
uvicorn app.main:app --host 0.0.0.0 --port 8080
```

## API Endpoints

- `GET /api/health` - Health check
- `POST /api/play` - Run agent episodes
- `POST /api/train` - Train agents (if enabled)
- `GET /ws/stream` - WebSocket streaming

## Training

```bash
# Train CEM agent
python -m rl.train --algo cem --generations 10 --seed 42

# Train REINFORCE agent  
python -m rl.train --algo reinforce --episodes 1000 --seed 42

# Evaluate trained policy
python -m rl.eval --policy policies/best.npz --episodes 10
```