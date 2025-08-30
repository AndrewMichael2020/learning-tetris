# Reinforcement Learning Tetris Web Application

A modern web application that demonstrates how artificial intelligence can learn to play Tetris using reinforcement learning algorithms. Watch as AI agents learn to stack blocks, clear lines, and achieve high scores through trial and error - just like humans do, but much faster!

## Application Screenshot

![Reinforcement Learning Tetris Web App](app-screenshot.png)

*The complete web interface showing the game board, real-time statistics, control panel, and helpful instructions for using the application.*

## What This Application Does

This is an interactive web application where you can:

1. **Watch AI agents play Tetris in real-time** - See the AI make decisions about where to place each Tetris piece
2. **Compare different learning algorithms** - Two different AI approaches compete to achieve the best scores
3. **Train your own AI agents** - Run training sessions to create better-performing agents
4. **Understand how AI learns** - Observe the learning process through comprehensive statistics and visualizations

The application features a standard 10x20 Tetris game board with all seven classic pieces (I, O, T, S, Z, J, L) and uses sophisticated algorithms to evaluate the best placement for each piece.

## Features

### Core Functionality
- **Full Tetris Environment**: Standard 10x20 game board with proper piece rotation, line clearing, and game over conditions
- **Real-time Visualization**: Watch the AI play through an interactive web interface with live game board updates
- **Multiple AI Algorithms**: 
  - **Cross-Entropy Method (CEM)**: Evolves a population of strategies to find optimal play patterns
  - **REINFORCE with Baseline**: Learns through policy gradients with variance reduction
- **Feature-based Evaluation**: AI analyzes board states using advanced features like holes, bumpiness, and line completion potential
- **WebSocket Streaming**: Real-time game state updates for smooth gameplay visualization

### Technical Features
- **FastAPI Web Interface**: Modern REST API with automatic documentation
- **Comprehensive Test Suite**: 95%+ code coverage ensuring reliability
- **Docker Support**: Easy deployment with containerization
- **Cloud Ready**: Optimized for Google Cloud Run deployment
- **Policy Management**: Save, load, and evaluate trained AI models

## Getting Started

### Prerequisites
- Python 3.11 or higher
- Basic familiarity with command line/terminal

### Installation and Setup

1. **Clone and Navigate to the Project**:
   ```bash
   git clone https://github.com/AndrewMichael2020/learning-tetris.git
   cd learning-tetris
   ```

2. **Create a Virtual Environment** (recommended):
   ```bash
   python -m venv .venv
   
   # On Windows:
   .venv\Scripts\activate
   
   # On macOS/Linux:
   source .venv/bin/activate
   ```

3. **Install the Application**:
   ```bash
   pip install -e .
   ```

4. **Verify Installation** (run tests):
   ```bash
   pytest -q
   ```
   **What to expect**: You should see output like `## passed, ## skipped in #.##s` indicating all tests pass.

5. **Start the Web Server**:
   ```bash
   uvicorn app.main:app --host 0.0.0.0 --port 8080
   ```

6. **Open Your Browser**:
   Navigate to `http://localhost:8080`

### What You'll See

When you open the web interface, you'll find:

1. **Game Board**: A visual Tetris board where you can watch the AI play
2. **Control Buttons**:
   - **"Stream Agent"**: Watch the AI play continuously in real-time with live updates
   - **"Play Once"**: Run a single complete game episode and view final statistics
   - **"Quick Train"**: Train a new AI agent (available when training is enabled)
3. **Statistics Panel**: Real-time updates showing:
   - Current score
   - Lines cleared
   - Steps taken
   - Game status
4. **Episode Results**: Historical data from completed games
5. **Button Instructions**: Clear explanations of what each control button does, displayed right below the game board

## How to Use the Application

**Note**: Each button includes helpful instructions right in the interface - look for the "How to Use" section below the game board for detailed explanations of what each button does.

### Watching the AI Play

1. Click **"Stream Agent"** to start real-time gameplay
2. **What you'll see**: 
   - Tetris pieces falling and being placed automatically
   - The game board updating in real-time
   - Statistics changing as lines are cleared
   - The AI attempting to maximize its score by clearing lines efficiently

3. Click the button again to stop streaming

### Running Single Episodes

1. Click **"Play Once"** to run a complete game
2. **What you'll see**:
   - The game runs quickly to completion
   - Final statistics appear in the results panel
   - Multiple episodes can be run to compare performance

### Understanding the AI's Strategy

The AI evaluates each possible placement using several factors:
- **Holes**: Empty spaces with blocks above them (bad)
- **Height**: How tall the columns are (generally worse when higher)
- **Bumpiness**: Uneven column heights (bad for future placements)
- **Line Clearing**: Potential for completing full rows (good)
- **Wells**: Deep vertical spaces (can be good or bad depending on context)

## API Usage

The application provides a REST API for programmatic access:

### Health Check
```bash
curl http://localhost:8080/api/health
```
**Expected response**:
```json
{
  "status": "ok",
  "train_enabled": false,
  "policy_loaded": true
}
```

### Run Agent Episodes
```bash
curl -X POST http://localhost:8080/api/play \
  -H "Content-Type: application/json" \
  -d '{"episodes": 5, "algo": "cem", "seed": 42}'
```
**Expected response**:
```json
{
  "total_lines": 23,
  "avg_score": 145.6,
  "episodes": 5,
  "scores": [132, 158, 143, 156, 139],
  "lines_cleared": [4, 6, 5, 5, 3],
  "episode_lengths": [89, 112, 98, 107, 94]
}
```

### WebSocket Streaming
Connect to `ws://localhost:8080/ws/stream` for real-time game updates. Each message contains:
- Base64-encoded game board image
- Current score and lines cleared
- Step number and completion status

## Training Your Own Agents

### Understanding the Training Process

The application supports training new AI agents using two different algorithms:

#### Cross-Entropy Method (CEM)
- **How it works**: Creates a population of strategies, evaluates them, keeps the best ones, and creates variations
- **Best for**: Finding robust strategies quickly
- **Training time**: Usually converges within 10-20 generations

#### REINFORCE with Baseline  
- **How it works**: Uses gradient-based learning with variance reduction
- **Best for**: Fine-tuning policies and handling complex decision spaces
- **Training time**: Requires more episodes but can achieve higher performance

### Training Commands

**Note**: Training is disabled by default in production. To enable training, set the environment variable `TRAIN_ENABLED=true`.

#### Train a CEM Agent:
```bash
python -m rl.train --algo cem --generations 10 --seed 42
```
**What happens**:
- Creates a population of 20 candidate strategies
- Evaluates each strategy over multiple games
- Selects the top 10 performers
- Creates new variations for the next generation
- Saves the best strategy to `policies/best.npz`

**Expected output**:
```
Generation 1/10: Best fitness = 89.3 (avg = 45.2)
Generation 2/10: Best fitness = 134.7 (avg = 78.9)
...
Training completed! Best fitness: 187.4
```

#### Train a REINFORCE Agent:
```bash
python -m rl.train --algo reinforce --episodes 1000 --seed 42
```
**What happens**:
- Runs episodes and collects experience
- Updates the policy based on rewards received
- Applies a baseline to reduce variance
- Gradually improves performance over time

**Expected output**:
```
Episode 100: Avg reward = 67.3 (±12.4)
Episode 200: Avg reward = 89.1 (±15.2)
...
Training completed! Final avg reward: 156.8
```

### Evaluating Trained Policies

```bash
python -m rl.eval --policy policies/best.npz --episodes 10
```
**Expected output**:
```
Evaluating policy: policies/best.npz
Episode 1: Score = 145, Lines = 5, Steps = 98
Episode 2: Score = 167, Lines = 7, Steps = 112
...
Average Score: 152.3 (±18.7)
Average Lines: 5.8 (±1.4)
```

## Docker Deployment

### Building and Running Locally

```bash
# Build the Docker image
docker build -t rl-tetris .

# Run the container
docker run -p 8080:8080 rl-tetris
```

### What the Container Includes
- Python 3.11 runtime
- All application dependencies
- Pre-trained AI models (if available)
- Web interface and API endpoints
- Training disabled by default for security

## Cloud Deployment

### Google Cloud Run (Recommended)

1. **Setup Google Cloud Project**:
   ```bash
   gcloud config set project YOUR-PROJECT-ID
   gcloud auth login
   ```

2. **Build and Deploy**:
   ```bash
   gcloud builds submit --tag gcr.io/YOUR-PROJECT-ID/rl-tetris
   gcloud run deploy rl-tetris \
     --image gcr.io/YOUR-PROJECT-ID/rl-tetris \
     --region us-central1 \
     --platform managed \
     --allow-unauthenticated
   ```

3. **What you get**: A publicly accessible URL where anyone can watch your AI play Tetris!

## Troubleshooting

### Common Issues

**"No module named 'rl'" or Import Errors**:
- Ensure you installed with `pip install -e .`
- Check that you're in the correct virtual environment

**"Connection refused" when accessing localhost:8080**:
- Ensure the server is running: `uvicorn app.main:app --host 0.0.0.0 --port 8080`
- Check that port 8080 isn't being used by another application

**Tests failing**:
- Ensure you have Python 3.11+
- Try running `pip install -e .` again
- Check that all dependencies are properly installed

**Poor AI performance**:
- The default policy might be random or poorly trained
- Try training a new agent with more generations/episodes
- Check that the `policies/` directory contains valid policy files

**WebSocket connection issues**:
- Ensure your browser supports WebSockets (most modern browsers do)
- Check browser developer console for error messages
- Try refreshing the page

### Getting Help

If you encounter issues:
1. Check the server logs for error messages
2. Verify all prerequisites are met
3. Try the troubleshooting steps above
4. Run the test suite to ensure everything is working: `pytest -v`

## Understanding the Results

### What Makes a Good Score?
- **Beginner AI**: 50-100 points per game, clears 1-3 lines
- **Intermediate AI**: 100-200 points per game, clears 4-8 lines  
- **Advanced AI**: 200+ points per game, clears 8+ lines consistently

### Performance Metrics
- **Score**: Points earned (more lines cleared at once = exponentially higher score)
- **Lines Cleared**: Total rows completed
- **Episode Length**: Number of pieces placed before game over
- **Efficiency**: Lines cleared per piece placed

### Interpreting Training Progress
- **Early stages**: High variance, occasional good games mixed with poor ones
- **Learning**: Variance decreases, average performance steadily improves
- **Convergence**: Performance plateaus at the agent's skill ceiling

The AI learns to balance immediate rewards (clearing lines now) with long-term strategy (maintaining a clean board for future opportunities).