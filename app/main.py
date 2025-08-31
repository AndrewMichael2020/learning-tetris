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
from fastapi.middleware.cors import CORSMiddleware

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

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins in development
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
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
        policy_loaded=policy_weights is not None,
        current_algorithm=policy_metadata.get("algorithm", "cem") if policy_metadata else "cem"
    )


@app.post("/api/play", response_model=PlayResponse)
async def play_episodes(request: PlayRequest):
    """Run agent for specified number of episodes."""
    
    # Import agent factory
    from .agent_factory import make_agent, get_default_params
    
    episode_scores = []
    episode_lines = []
    episode_lengths = []
    
    # Create RNG with seed
    episode_rng = np.random.default_rng(request.seed)
    
    # Check if using new agents or legacy CEM/REINFORCE
    if request.algo in ["greedy", "tabu", "anneal", "aco"]:
        # Use new agent system
        try:
            params = get_default_params(request.algo)
            agent = make_agent(request.algo, params)
            
            for episode in range(request.episodes):
                env = TetrisEnv()
                env.reset(seed=episode_rng.integers(0, 1000000))
                agent.reset()  # Reset agent state for new episode
                
                steps = 0
                max_steps = 1000
                
                while not env.game_over and steps < max_steps:
                    # Get action from agent
                    try:
                        col, rotation = agent.select_action(env)
                        success = execute_placement(env, col, rotation)
                        
                        if not success:
                            break
                            
                        steps += 1
                        
                        # Spawn new piece
                        if not env.game_over:
                            env._spawn_piece()
                            
                    except Exception as e:
                        print(f"Agent error: {e}")
                        break
                
                episode_scores.append(env.score)
                episode_lines.append(env.lines_cleared)
                episode_lengths.append(steps)
                
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Agent error: {str(e)}")
            
    else:
        # Use legacy policy-based system for CEM/REINFORCE
        if policy_weights is None:
            raise HTTPException(status_code=500, detail="No policy loaded")
        
        for episode in range(request.episodes):
            env = TetrisEnv()
            env.reset(seed=episode_rng.integers(0, 1000000))
            
            steps = 0
            max_steps = 1000
            
            while not env.game_over and steps < max_steps:
                # Get best placement using correct algorithm
                try:
                    if request.algo in ["cem", "reinforce"]:
                        # Use trained policy weights
                        best_col, best_rotation, _ = get_best_placement(env, policy_weights)
                    else:
                        # Use specific algorithm agents
                        from .agent_factory import make_agent
                        agent = make_agent(request.algo)
                        if agent is None:
                            # Fallback to policy weights
                            best_col, best_rotation, _ = get_best_placement(env, policy_weights)
                        else:
                            best_col, best_rotation = agent.get_action(env)
                except Exception as e:
                    print(f"Error getting best placement in play: {e}")
                    # Use center fallback placement instead of left-biased
                    best_col, best_rotation = 4, 0
                
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
        avg_score=float(np.mean(episode_scores)) if episode_scores else 0.0,
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
        
        elif request.algo in ["greedy", "tabu", "anneal", "aco"]:
            # New algorithms don't need training - they work directly with their parameters
            # For these, we'll just validate the parameters and run a quick evaluation
            from .agent_factory import make_agent, get_default_params
            
            # Use provided parameters or defaults
            if hasattr(request, 'params') and request.params is not None:
                params = request.params
            else:
                params = get_default_params(request.algo)
            
            print(f"Training {request.algo} with params: {params}")
            
            # Validate agent creation
            agent = make_agent(request.algo, params)
            if agent is None:
                raise ValueError(f"Failed to create {request.algo} agent")
            
            # Run a quick evaluation to get performance
            env = env_factory()
            env.reset(seed=request.seed)
            agent.reset()
            
            steps = 0
            max_steps = 200  # More steps for better evaluation
            
            while not env.game_over and steps < max_steps:
                try:
                    col, rotation = agent.select_action(env)
                    success = execute_placement(env, col, rotation) 
                    
                    if not success:
                        break
                    
                    steps += 1
                    
                    if not env.game_over:
                        env._spawn_piece()
                        
                except Exception as e:
                    print(f"Error during agent evaluation: {e}")
                    break
            
            # Calculate performance score
            best_performance = env.score + env.lines_cleared * 100
            print(f"Agent performance: Score={env.score}, Lines={env.lines_cleared}, Total={best_performance}")
        
        else:
            raise ValueError(f"Unknown algorithm: {request.algo}")
        
        # Reload policy after training (for CEM/REINFORCE only)
        if request.algo in ["cem", "reinforce"]:
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
        print(f"Training error: {e}")  # Server-side logging
        return TrainResponse(
            success=False,
            message=f"Training failed: {str(e)}",
            algo=request.algo,
            best_performance=0.0,
            training_time=time.time() - start_time
        )


@app.post("/api/reset-training")
async def reset_training():
    """Reset training state and statistics for all models."""
    global policy_weights, policy_metadata
    
    try:
        # Reload the original baseline policy to reset CEM/REINFORCE weights
        policy_dict = load_or_create_policy(
            path=config.policy_path,
            feature_dim=17,
            seed=config.demo_seed
        )
        policy_weights = policy_dict['linear_weights']
        policy_metadata = policy_dict.get('_metadata', {})
        
        # Reset any cached agent states or parameters for new algorithms
        # This ensures all algorithms start fresh from their defaults
        from .agent_factory import reset_all_agents
        try:
            reset_all_agents()  # Clear any cached agent state
        except:
            pass  # If function doesn't exist, that's OK
        
        # Log the reset action
        print("Training state reset - all models reset to defaults")
        
        return {
            "success": True,
            "message": "All training models reset successfully",
            "policy_metadata": policy_metadata
        }
        
    except Exception as e:
        print(f"Reset training error: {e}")
        return {
            "success": False, 
            "message": f"Reset failed: {str(e)}"
        }


@app.websocket("/ws/stream")
async def websocket_stream(websocket: WebSocket, episodes: int = 1, seed: Optional[int] = None, algo: str = "cem"):
    """WebSocket endpoint for streaming agent gameplay."""
    await websocket.accept()
    
    if policy_weights is None:
        await websocket.send_json({"error": "No policy loaded"})
        await websocket.close()
        return
    
    try:
        # For CEM and REINFORCE, indicate this is a learning process
        if algo in ["cem", "reinforce"]:
            await websocket.send_json({
                "type": "learning_mode",
                "algorithm": algo.upper(),
                "message": f"{algo.upper()} Streaming: Showing policy evolution over {episodes} episodes"
            })
        
        # Create agent once if using new algorithms
        agent = None
        if algo not in ["cem", "reinforce"]:
            from .agent_factory import make_agent, get_default_params
            params = get_default_params(algo)
            agent = make_agent(algo, params)
            if agent is None:
                print(f"Failed to create agent for {algo}, falling back to policy")
        
        # Stream multiple episodes if requested
        for episode in range(episodes):
            # Create environment
            env = TetrisEnv()
            
            # Use provided seed or generate random one
            episode_seed = seed if seed is not None else rng.integers(0, 1000000)
            env.reset(seed=episode_seed)
            
            # Reset agent state for new episode
            if agent is not None:
                agent.reset()
            
            step_count = 0
            max_steps = 500  # Limit for streaming
            
            # Send initial frame with episode info
            frame = env.render(mode="rgb_array")
            frame_data = StreamFrame(
                frame=frame_to_base64(frame),
                lines=int(env.lines_cleared),
                score=float(env.score),
                step=int(step_count)
            )
            await websocket.send_json({
                **frame_data.model_dump(),
                "episode": episode + 1,
                "total_episodes": episodes,
                "algorithm": algo,
                "learning_info": f"{algo.upper()} Learning: Episode {episode + 1}/{episodes}" if algo in ["cem", "reinforce"] else None
            })
            
            while not env.game_over and step_count < max_steps:
                # Control FPS (sleep before processing to ensure consistent timing)
                await asyncio.sleep(1.0 / config.stream_fps)
                
                # Get best placement using the correct algorithm
                try:
                    if algo in ["cem", "reinforce"]:
                        # Use trained policy weights
                        best_col, best_rotation, _ = get_best_placement(env, policy_weights)
                    else:
                        # Use pre-created agent
                        if agent is None:
                            # Fallback to policy weights
                            best_col, best_rotation, _ = get_best_placement(env, policy_weights)
                        else:
                            best_col, best_rotation = agent.select_action(env)
                except Exception as e:
                    print(f"Error getting best placement in stream: {e}")
                    # Use center fallback placement instead of left-biased
                    best_col, best_rotation = 4, 0
                
                # Execute placement
                success = execute_placement(env, best_col, best_rotation)
                
                if not success:
                    break
                
                step_count += 1
                
                # Send frame
                frame = env.render(mode="rgb_array")
                frame_data = StreamFrame(
                    frame=frame_to_base64(frame),
                    lines=int(env.lines_cleared),
                    score=float(env.score),
                    step=int(step_count),
                    done=bool(env.game_over)
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
    try:
        await websocket.accept()
        print(f"Play Once WebSocket accepted: seed={seed}, algo={algo}")
        
        if policy_weights is None:
            await websocket.send_json({"error": "No policy loaded"})
            await websocket.close()
            return

        # Create agent once if using new algorithms
        agent = None
        if algo not in ["cem", "reinforce"]:
            from .agent_factory import make_agent, get_default_params
            params = get_default_params(algo)
            agent = make_agent(algo, params)
            if agent is None:
                print(f"Failed to create agent for {algo}, falling back to policy")
        
        # Create environment
        env = TetrisEnv()
        
        # Use provided seed or generate random one
        episode_seed = seed if seed is not None else rng.integers(0, 1000000)
        env.reset(seed=episode_seed)
        print(f"Environment created and reset with seed: {episode_seed}")
        
        # Reset agent state for new episode
        if agent is not None:
            agent.reset()
        
        step_count = 0
        max_steps = 1000  # Higher limit for single episode
        
        # Send initial frame
        frame = env.render(mode="rgb_array")
        frame_data = StreamFrame(
            frame=frame_to_base64(frame),
            lines=int(env.lines_cleared),
            score=float(env.score),
            step=int(step_count)
        )
        await websocket.send_json({
            **frame_data.model_dump(),
            "algorithm": algo,
            "seed": int(episode_seed),
            "mode": "play-once"
        })
        print(f"Initial frame sent")
        
        while not env.game_over and step_count < max_steps:
            # Slightly slower than stream for better visualization 
            await asyncio.sleep(1.0 / max(config.stream_fps - 2, 1))
            
            # Get best placement using the correct algorithm
            try:
                if algo in ["cem", "reinforce"]:
                    # Use trained policy weights
                    best_col, best_rotation, _ = get_best_placement(env, policy_weights)
                else:
                    # Use pre-created agent
                    if agent is None:
                        # Fallback to policy weights
                        best_col, best_rotation, _ = get_best_placement(env, policy_weights)
                    else:
                        best_col, best_rotation = agent.select_action(env)
                        
                print(f"Step {step_count}: best placement col={best_col}, rotation={best_rotation}")
            except Exception as e:
                print(f"Error getting best placement at step {step_count}: {e}")
                # Use center fallback placement instead of left-biased
                best_col, best_rotation = 4, 0
            
            # Execute placement
            success = execute_placement(env, best_col, best_rotation)
            
            if not success:
                print(f"Placement failed at step {step_count}")
                break
                
            step_count += 1
            
            print(f"After placement: score={env.score}, lines={env.lines_cleared}, game_over={env.game_over}")
            
            # Send frame update
            frame = env.render(mode="rgb_array")
            frame_data = StreamFrame(
                frame=frame_to_base64(frame),
                lines=int(env.lines_cleared),
                score=float(env.score),
                step=int(step_count),
                done=bool(env.game_over)
            )
            message_data = {
                **frame_data.model_dump(),
                "placement": {"col": int(best_col), "rotation": int(best_rotation)}
            }
            print(f"Sending WebSocket message: score={message_data.get('score')}, lines={message_data.get('lines')}, step={message_data.get('step')}")
            await websocket.send_json(message_data)
            
            # Spawn new piece if game continues
            if not env.game_over:
                env._spawn_piece()
        
        print(f"Game completed: score={env.score}, lines={env.lines_cleared}, steps={step_count}")
        
        # Send final completion message
        await websocket.send_json({
            "final": True,
            "score": float(env.score),
            "lines": int(env.lines_cleared),
            "steps": int(step_count),
            "algorithm": algo,
            "seed": int(episode_seed) if episode_seed is not None else None
        })
        print(f"Final message sent")
            
    except WebSocketDisconnect:
        print("WebSocket disconnected")
        pass
    except Exception as e:
        print(f"Exception in play-once: {e}")
        try:
            await websocket.send_json({"error": str(e)})
        except:
            pass
    finally:
        try:
            await websocket.close()
            print("WebSocket closed")
        except:
            pass


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=config.port)