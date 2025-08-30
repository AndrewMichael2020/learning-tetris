"""
FastAPI application for Tetris RL web interface.
"""
import asyncio
import base64
import io
import json
import time
from contextlib import asynccontextmanager
from typing import Optional, Dict, Any
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse

from .config import config
from .schemas import (
    PlayRequest, PlayResponse, TrainRequest, TrainResponse,
    HealthResponse, StreamFrame
)
from rl.tetris_env import TetrisEnv
from rl.policy_store import load_or_create_policy
from rl.afterstate import get_best_placement, execute_placement
from rl.cem_agent import evolve as cem_evolve
from rl.reinforce_agent import train as reinforce_train


# Global state
policy_weights: Optional[np.ndarray] = None
policy_metadata: Optional[Dict[str, Any]] = None
rng = np.random.default_rng(config.demo_seed)


def create_env_factory():
    """Create environment factory."""
    def factory():
        return TetrisEnv(width=10, height=20)
    return factory


def load_policy():
    """Load policy weights on startup."""
    global policy_weights, policy_metadata
    
    try:
        policy_dict = load_or_create_policy(
            path=config.policy_path,
            feature_dim=17,
            seed=config.demo_seed
        )
        policy_weights = policy_dict['linear_weights']
        policy_metadata = policy_dict.get('_metadata', {})
        print(f"Loaded policy from {config.policy_path}")
        
        if policy_metadata:
            print(f"Policy metadata: {policy_metadata}")
            
    except Exception as e:
        print(f"Error loading policy: {e}")
        # Create random fallback policy
        policy_weights = np.random.normal(0, 0.1, size=17).astype(np.float32)
        policy_metadata = {'created': 'random_fallback'}


def frame_to_base64(frame: np.ndarray) -> str:
    """Convert RGB frame to base64 PNG string."""
    try:
        from PIL import Image
        # Convert numpy array to PIL Image and then to PNG
        if frame.dtype != np.uint8:
            frame = frame.astype(np.uint8)
            
        # Convert to PIL Image
        img = Image.fromarray(frame)
        
        # Save to base64 PNG
        buffer = io.BytesIO()
        img.save(buffer, format='PNG')
        png_data = buffer.getvalue()
        return base64.b64encode(png_data).decode('utf-8')
        
    except ImportError:
        # Fallback if PIL not available - create a simple data structure
        height, width = frame.shape[:2]
        
        # Create JSON representation of the board
        board_data = {
            'width': int(width),
            'height': int(height),
            'data': frame.tolist()  # Convert numpy array to nested list
        }
        
        # Encode as base64 JSON
        json_str = json.dumps(board_data)
        return base64.b64encode(json_str.encode('utf-8')).decode('utf-8')


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan."""
    # Startup
    load_policy()
    yield
    # Shutdown (nothing to cleanup)


# Initialize FastAPI app
app = FastAPI(
    title="RL Tetris", 
    description="Reinforcement Learning Tetris Web App",
    lifespan=lifespan
)

# Serve static files
app.mount("/static", StaticFiles(directory="app/static"), name="static")


@app.get("/", response_class=HTMLResponse)
async def serve_index():
    """Serve main HTML page."""
    try:
        with open("app/static/index.html", "r") as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        return HTMLResponse(content="<h1>Tetris RL App</h1><p>Static files not found</p>")


@app.get("/api/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="ok",
        train_enabled=config.train_enabled,
        policy_loaded=policy_weights is not None
    )


@app.post("/api/play", response_model=PlayResponse)
async def play_episodes(request: PlayRequest):
    """Run agent for specified number of episodes."""
    if policy_weights is None:
        raise HTTPException(status_code=500, detail="No policy loaded")
    
    episode_scores = []
    episode_lines = []
    episode_lengths = []
    
    # Create RNG with seed
    episode_rng = np.random.default_rng(request.seed)
    
    for episode in range(request.episodes):
        env = TetrisEnv()
        env.reset(seed=episode_rng.integers(0, 1000000))
        
        steps = 0
        max_steps = 1000
        
        while not env.game_over and steps < max_steps:
            # Get best placement using loaded policy
            best_col, best_rotation, _ = get_best_placement(env, policy_weights)
            
            # Execute placement
            success = execute_placement(env, best_col, best_rotation)
            
            if not success:
                break
            
            steps += 1
            
            # Spawn new piece
            if not env.game_over:
                env._spawn_piece()
        
        episode_scores.append(env.score)
        episode_lines.append(env.lines_cleared)
        episode_lengths.append(steps)
    
    return PlayResponse(
        total_lines=sum(episode_lines),
        avg_score=float(np.mean(episode_scores)),
        episodes=request.episodes,
        scores=episode_scores,
        lines_cleared=episode_lines,
        episode_lengths=episode_lengths
    )


@app.post("/api/train", response_model=TrainResponse)
async def train_agent(request: TrainRequest):
    """Train agent (only if training enabled)."""
    if not config.train_enabled:
        raise HTTPException(status_code=403, detail="Training is disabled")
    
    start_time = time.time()
    env_factory = create_env_factory()
    
    try:
        if request.algo == "cem":
            # Run CEM training
            result = cem_evolve(
                env_factory=env_factory,
                generations=min(request.generations, config.max_generations),
                seed=request.seed,
                out_path=config.policy_path,
                episodes_per_candidate=request.episodes_per_candidate,
                population_size=request.population_size
            )
            best_performance = result['best_fitness']
            
        elif request.algo == "reinforce":
            # Run REINFORCE training
            result = reinforce_train(
                env_factory=env_factory,
                episodes=min(request.episodes, config.max_episodes),
                seed=request.seed,
                out_path=config.policy_path,
                learning_rate=request.learning_rate
            )
            best_performance = result['best_reward']
        
        else:
            raise ValueError(f"Unknown algorithm: {request.algo}")
        
        # Reload policy after training
        load_policy()
        
        training_time = time.time() - start_time
        
        return TrainResponse(
            success=True,
            message=f"Training completed successfully with {request.algo.upper()}",
            algo=request.algo,
            best_performance=best_performance,
            training_time=training_time
        )
        
    except Exception as e:
        return TrainResponse(
            success=False,
            message=f"Training failed: {str(e)}",
            algo=request.algo,
            best_performance=0.0,
            training_time=time.time() - start_time
        )


@app.websocket("/ws/stream")
async def websocket_stream(websocket: WebSocket, episodes: int = 1, seed: Optional[int] = None, algo: str = "cem"):
    """WebSocket endpoint for streaming agent gameplay."""
    await websocket.accept()
    
    if policy_weights is None:
        await websocket.send_json({"error": "No policy loaded"})
        await websocket.close()
        return
    
    try:
        # Stream multiple episodes if requested
        for episode in range(episodes):
            # Create environment
            env = TetrisEnv()
            
            # Use provided seed or generate random one
            episode_seed = seed if seed is not None else rng.integers(0, 1000000)
            env.reset(seed=episode_seed)
            
            step_count = 0
            max_steps = 500  # Limit for streaming
            
            # Send initial frame with episode info
            frame = env.render(mode="rgb_array")
            frame_data = StreamFrame(
                frame=frame_to_base64(frame),
                lines=env.lines_cleared,
                score=float(env.score),
                step=step_count
            )
            await websocket.send_json({
                **frame_data.model_dump(),
                "episode": episode + 1,
                "total_episodes": episodes,
                "algorithm": algo
            })
            
            while not env.game_over and step_count < max_steps:
                # Control FPS (sleep before processing to ensure consistent timing)
                await asyncio.sleep(1.0 / config.stream_fps)
                
                # Get best placement using the loaded policy
                best_col, best_rotation, _ = get_best_placement(env, policy_weights)
                
                # Execute placement
                success = execute_placement(env, best_col, best_rotation)
                
                if not success:
                    break
                
                step_count += 1
                
                # Send frame
                frame = env.render(mode="rgb_array")
                frame_data = StreamFrame(
                    frame=frame_to_base64(frame),
                    lines=env.lines_cleared,
                    score=float(env.score),
                    step=step_count,
                    done=env.game_over
                )
                await websocket.send_json({
                    **frame_data.model_dump(),
                    "episode": episode + 1,
                    "total_episodes": episodes
                })
                
                # Spawn new piece if game continues
                if not env.game_over:
                    env._spawn_piece()
            
            # Send episode completion message
            await websocket.send_json({
                "episode_complete": True,
                "episode": episode + 1,
                "total_episodes": episodes,
                "score": float(env.score),
                "lines": env.lines_cleared,
                "final": episode == episodes - 1
            })
            
            # Brief pause between episodes if there are more to play
            if episode < episodes - 1:
                await asyncio.sleep(1.0)  # 1 second pause between episodes
                
    except WebSocketDisconnect:
        # Client disconnected, this is normal
        pass
    except Exception as e:
        # Only try to send error if websocket is still connected
        try:
            await websocket.send_json({"error": str(e)})
        except:
            # Websocket already closed, ignore
            pass
    finally:
        # Only try to close if websocket is still connected
        try:
            await websocket.close()
        except:
            # Already closed, ignore
            pass


@app.websocket("/ws/play-once")
async def websocket_play_once(websocket: WebSocket, seed: Optional[int] = None, algo: str = "cem"):
    """WebSocket endpoint for single episode gameplay with step-by-step visualization."""
    await websocket.accept()
    
    if policy_weights is None:
        await websocket.send_json({"error": "No policy loaded"})
        await websocket.close()
        return
    
    try:
        # Create environment
        env = TetrisEnv()
        
        # Use provided seed or generate random one
        episode_seed = seed if seed is not None else rng.integers(0, 1000000)
        env.reset(seed=episode_seed)
        
        step_count = 0
        max_steps = 1000  # Higher limit for single episode
        
        # Send initial frame
        frame = env.render(mode="rgb_array")
        frame_data = StreamFrame(
            frame=frame_to_base64(frame),
            lines=env.lines_cleared,
            score=float(env.score),
            step=step_count
        )
        await websocket.send_json({
            **frame_data.model_dump(),
            "algorithm": algo,
            "seed": episode_seed,
            "mode": "play-once"
        })
        
        while not env.game_over and step_count < max_steps:
            # Slightly slower than stream for better visualization 
            await asyncio.sleep(1.0 / max(config.stream_fps - 2, 1))
            
            # Get best placement using the loaded policy
            best_col, best_rotation, _ = get_best_placement(env, policy_weights)
            
            # Execute placement
            success = execute_placement(env, best_col, best_rotation)
            
            if not success:
                break
            
            step_count += 1
            
            # Send frame update
            frame = env.render(mode="rgb_array")
            frame_data = StreamFrame(
                frame=frame_to_base64(frame),
                lines=env.lines_cleared,
                score=float(env.score),
                step=step_count,
                done=env.game_over
            )
            await websocket.send_json({
                **frame_data.model_dump(),
                "placement": {"col": best_col, "rotation": best_rotation}
            })
            
            # Spawn new piece if game continues
            if not env.game_over:
                env._spawn_piece()
        
        # Send final completion message
        await websocket.send_json({
            "final": True,
            "score": float(env.score),
            "lines": env.lines_cleared,
            "steps": step_count,
            "algorithm": algo,
            "seed": episode_seed
        })
            
    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await websocket.send_json({"error": str(e)})
        except:
            pass
    finally:
        try:
            await websocket.close()
        except:
            pass


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=config.port)