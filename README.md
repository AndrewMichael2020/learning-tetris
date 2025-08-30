# 🎮 Reinforcement Learning Tetris: AI Training Playground

**A comprehensive web-based platform for training, comparing, and visualizing AI agents learning to play Tetris.** Experience cutting-edge machine learning algorithms in action with real-time gameplay visualization, advanced scoring systems, and creative algorithm implementations.

## 🌟 Key Features

### 🧠 **6 Advanced AI Algorithms**
- **🔬 Cross-Entropy Method (CEM)**: Evolution-based policy optimization
- **🎯 REINFORCE**: Classic policy gradient reinforcement learning  
- **⚡ Greedy Heuristic ("Nurse Dictator")**: Fast, deterministic decision-making
- **🔄 Tabu Search ("Nurse Gossip")**: Memory-based local search with aspiration criteria
- **🌡️ Simulated Annealing ("Coffee Break")**: Temperature-based probabilistic optimization
- **🐜 Ant Colony Optimization ("Night Shift Ant March")**: Swarm intelligence approach

### 🎯 **Interactive Gameplay Modes**
- **🎬 Stream Agent**: Watch AI play continuously with real-time visualization
- **🎲 Play Once**: Single episode with step-by-step gameplay analysis
- **📊 Play Multiple**: Batch episodes with comprehensive statistics tracking
- **⚡ Quick Train**: Rapid algorithm training with progress monitoring

### 📈 **Advanced Features**
- **Enhanced Tetris Scoring**: Standard scoring system (100/300/500/800 points for 1/2/3/4 lines)
- **Flicker-Free Rendering**: Double-buffered canvas for smooth visual experience
- **Algorithm-Specific Models**: Each algorithm uses its optimal trained implementation
- **Real-Time Statistics**: Live score tracking, episode averaging, and performance metrics
- **Comprehensive Training Controls**: Customizable parameters for each algorithm

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Modern web browser with WebSocket support

### Installation & Setup
```bash
# Clone the repository
git clone <repository-url>
cd learning-tetris

# Install dependencies
pip install -r requirements.txt

# Start the server
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000

# Open your browser
# Navigate to: http://localhost:8000
```

## 🎮 How to Use

### 1. **Training AI Agents**
- Select an algorithm (CEM, REINFORCE, Greedy, etc.)
- Adjust training parameters (generations, population size, learning rate)
- Click **Quick Train** to start evolution/learning
- Monitor progress in real-time

### 2. **Testing Performance**  
- **Play Once**: Watch a single episode unfold step-by-step
- **Stream Agent**: Continuous play with live performance metrics
- **Play Multiple**: Run batch episodes for statistical analysis

### 3. **Comparing Algorithms**
- Train different algorithms with same parameters
- Use **Play Multiple** to gather performance statistics
- Compare average scores, line clearing efficiency, and consistency

### 4. **Understanding Results**
- **Score**: Enhanced Tetris scoring system (not just 10 points per line!)
- **Best Score**: Highest score achieved in current session  
- **Max Score**: Theoretical maximum (999999) for reference
- **Episodes/Avg Score**: Statistical performance over multiple games

## 🔬 Algorithm Details

### **Cross-Entropy Method (CEM)**
- **Type**: Evolutionary optimization
- **Trains**: Linear policy weights for feature-based decision making
- **Best for**: Stable, consistent performance
- **Parameters**: Generations, population size, elite selection

### **REINFORCE** 
- **Type**: Policy gradient reinforcement learning
- **Trains**: Neural network policy with gradient ascent
- **Best for**: Learning complex strategies through trial and error
- **Parameters**: Learning rate, episodes per update

### **Greedy Heuristic ("Nurse Dictator")**
- **Type**: Deterministic rule-based
- **Strategy**: Immediate optimization of hand-crafted features
- **Best for**: Fast decisions, baseline comparison
- **Parameters**: Feature weights (holes, height, bumpiness)

### **Tabu Search ("Nurse Gossip")**
- **Type**: Memory-based local search  
- **Strategy**: Avoid recently explored solutions, use aspiration criteria
- **Best for**: Escaping local optima in feature space
- **Parameters**: Tabu tenure, neighborhood size, aspiration threshold

### **Simulated Annealing ("Coffee Break")**
- **Type**: Temperature-based probabilistic search
- **Strategy**: Accept worse solutions early, converge to best later
- **Best for**: Global optimization with controlled exploration
- **Parameters**: Initial temperature, cooling rate, proposal distribution

### **Ant Colony Optimization ("Night Shift Ant March")**
- **Type**: Swarm intelligence
- **Strategy**: Pheromone trails guide exploration of solution space
- **Best for**: Distributed search with emergent intelligence  
- **Parameters**: Alpha, beta, evaporation rate, colony size

## 🎯 Understanding Performance

### **Scoring System**
- **Single Line**: 100 points + piece placement bonus
- **Double Line**: 300 points + piece placement bonus  
- **Triple Line**: 500 points + piece placement bonus
- **Tetris (4 lines)**: 800 points + piece placement bonus
- **Piece Placement**: +1 point per piece placed

### **Performance Benchmarks**
- **Beginner AI**: 50-150 points, 1-2 lines per game
- **Intermediate AI**: 200-400 points, 3-5 lines per game
- **Advanced AI**: 500+ points, 6+ lines per game
- **Expert AI**: 700+ points with consistent line clearing

### **Key Metrics**
- **Average Score**: Consistent performance indicator
- **Best Score**: Peak performance capability
- **Lines per Episode**: Efficiency metric
- **Episode Length**: Survival ability (steps before game over)

## 🧪 Testing & Verification

### **Automated Testing**
```bash
# Run complete test suite
pytest -v

# Individual test modules
pytest tests/test_env.py -v          # Game environment
pytest tests/test_api.py -v          # API endpoints
pytest tests/test_features.py -v     # Feature extraction
pytest tests/test_new_algorithms.py -v # Algorithm implementations
```

### **Manual Verification**
1. **Health Check**: Ensure API responds at `/api/health`
2. **Training Test**: Train each algorithm and verify policy updates
3. **Gameplay Test**: Run Play Once/Multiple and verify statistics update
4. **Algorithm Test**: Compare different algorithms on same seed

### **Demo Script**
```bash
# Automated verification
python create_demo_screenshot.py
```

## 🔧 Technical Architecture

### **Backend (FastAPI)**
- **WebSocket Endpoints**: Real-time gameplay streaming
- **REST API**: Training, health checks, batch gameplay
- **Policy Management**: Automatic loading/saving of trained models
- **Algorithm Factory**: Dynamic agent creation and configuration

### **Frontend (Vanilla JS)**  
- **Double-Buffered Canvas**: Flicker-free rendering system
- **Real-Time Statistics**: Live DOM updates during gameplay
- **WebSocket Client**: Bi-directional communication with server
- **Responsive UI**: Parameter controls for all algorithms

### **AI Algorithms**
- **Modular Design**: Each algorithm as independent agent class
- **Consistent Interface**: Standardized `get_action()` method
- **Parameter Validation**: Schema-based configuration validation
- **Performance Tracking**: Built-in fitness evaluation

## 🎨 Creative Features

### **Algorithm Personalities**
- **"Nurse Dictator"** (Greedy): Bossy, efficient, no-nonsense decision making
- **"Nurse Gossip"** (Tabu): Remembers everything, spreads information  
- **"Coffee Break"** (Simulated Annealing): Starts hot, cools down gradually
- **"Night Shift Ant March"** (ACO): Collective intelligence, follows trails

### **Visual Enhancements**
- **Colorful Tetris Pieces**: Vibrant, distinct piece coloring
- **Smooth Animations**: Frame-by-frame gameplay visualization
- **Real-Time Metrics**: Live performance tracking during play
- **Interactive Controls**: Customizable parameters for each algorithm

## 🏆 Advanced Usage

### **Hyperparameter Tuning**
```python
# Example: Optimize CEM parameters
generations = [10, 20, 50]  
population_sizes = [20, 50, 100]
# Test combinations and compare performance
```

### **Algorithm Comparison**
```python
# Fixed seed testing for fair comparison
seed = 42
algorithms = ['cem', 'reinforce', 'greedy', 'tabu']
# Run each algorithm with identical conditions
```

### **Performance Analysis**
- Track convergence rates during training
- Analyze score distributions across episodes
- Compare consistency (variance) between algorithms
- Evaluate sample efficiency (performance vs. training time)

## 🔮 Future Enhancements

### **Planned Features**
- **Multi-Agent Tournament Mode**: Algorithms compete head-to-head
- **Advanced Visualization**: Training progress charts and heatmaps
- **Custom Algorithm Builder**: User-defined heuristics and parameters  
- **Replay System**: Save and replay interesting games
- **Leaderboard**: Track best performances across all algorithms

### **Research Directions**  
- **Deep Reinforcement Learning**: DQN, A3C, PPO implementations
- **Meta-Learning**: Algorithms that learn to learn faster
- **Multi-Objective Optimization**: Balance speed vs. score vs. style
- **Human vs. AI**: Interactive play mode for comparison

## 📊 Results & Insights

### **Algorithm Performance Summary**
Based on extensive testing across multiple seeds and configurations:

1. **CEM**: Most consistent, good average performance
2. **REINFORCE**: Highest potential, more variance
3. **Greedy**: Fast, reliable baseline performance  
4. **Tabu Search**: Good at escaping local optima
5. **Simulated Annealing**: Balanced exploration/exploitation
6. **Ant Colony**: Unique emergent behaviors, moderate performance

### **Key Learning Insights**
- **Feature Engineering**: Hole count and height matter most
- **Exploration vs. Exploitation**: Balance crucial for learning
- **Sample Efficiency**: Some algorithms learn faster than others
- **Generalization**: Performance consistency across different game states

## 🤝 Contributing

### **Development Setup**
```bash
# Install development dependencies  
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Run tests before committing
pytest -v
black . --check
flake8 .
```

### **Adding New Algorithms**
1. Create agent class in `rl/` directory
2. Implement `get_action(env)` method
3. Add to `agent_factory.py` 
4. Update schema in `algo_schemas.py`
5. Add tests in `tests/test_new_algorithms.py`

### **Bug Reports & Feature Requests**
- Use GitHub Issues with detailed descriptions
- Include reproduction steps and environment info
- Attach screenshots/videos for UI issues

## 📝 License & Citations

This project combines educational content with research-grade implementations. Please cite appropriate papers when using specific algorithms in academic work.

---

## 🎉 Get Started Now!

Ready to see AI learn Tetris? Start the server and open your browser:

```bash
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
# Navigate to http://localhost:8000 and start training!
```

**Experience the future of AI learning - where algorithms develop strategies, compete for high scores, and evolve before your eyes!** 🚀🧠🎮
🎨 **Colorful Gameplay**: Vibrant, gradient-enhanced Tetris pieces make every game visually stunning  
⚡ **Real-Time Learning**: Watch AI improve its strategies as it plays
🎮 **Interactive Controls**: Train your own agents, compare algorithms, and explore AI behavior
📊 **Live Statistics**: See scores, lines cleared, and training progress in real-time
🚀 **Easy to Use**: One-click setup, web-based interface, no complex installation

## 🎯 Quick Start Guide

### 1️⃣ Setup (One Command!)
```bash
git clone https://github.com/AndrewMichael2020/learning-tetris.git
cd learning-tetris
pip install -r requirements.txt
```

### 2️⃣ Launch the App  
```bash
# Enable training (optional but recommended!)
TRAIN_ENABLED=true python -m uvicorn app.main:app --host 0.0.0.0 --port 8000

# Or just watch pre-trained agents play
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### 3️⃣ Open in Browser
Visit `http://localhost:8000` and start exploring!
![Screen](./Screenshot%202025-08-30%20011325.png)

### ✅ Verified Working Status
The application has been thoroughly tested and is fully functional:
- ✅ **Server Health**: All endpoints responding correctly
- ✅ **AI Agents**: CEM and REINFORCE algorithms operational  
- ✅ **Game Engine**: Tetris environment running smoothly
- ✅ **Web Interface**: Interactive controls and real-time visualization
- ✅ **API Integration**: REST endpoints and WebSocket streaming active

🎯 **Live Demo Verification**: The app successfully processes game requests, loads AI policies, and provides interactive gameplay. All core features including Stream Agent, Play Once, and Play Multiple modes are operational.

*View the [demo visualization](./app_demo_visualization.html) to see the app interface in action!*

## 🎮 How to Use the App

### 🔴 **Stream Agent** - Watch AI Play Live
- Click "Stream Agent" to watch the current AI play continuously
- Choose algorithm: CEM (evolution-based) or REINFORCE (neural network-like)
- Set number of episodes and random seed
- Perfect for: Understanding how AI makes decisions

### 🟡 **Play Once** - Single Game Analysis  
- Click "Play Once" for a detailed single-game analysis
- Great for studying specific AI behaviors and strategies
- Shows move-by-move decision making
- Perfect for: Learning how the AI evaluates each position

### 🟢 **Play Multiple** - Batch Testing
- Run multiple games to see average performance
- Compare different algorithms and settings
- Get statistical insights across many games
- Perfect for: Evaluating AI consistency and reliability

### 🔵 **Quick Train** - Create Your Own AI! ⭐
**This is where the magic happens!**

1. **Choose Your Algorithm**:
   - **CEM**: Evolves a population of strategies (like natural selection for AI!)
   - **REINFORCE**: Learns through trial and error with neural network principles

2. **Set Training Parameters**:
   - **CEM**: Generations (3-10), Population Size (10-50)  
   - **REINFORCE**: Episodes (50-500), Learning Rate (0.001-1.0)

3. **Click "Quick Train"** and watch your AI learn in the Activity Log!

4. **Test Your Trained Agent**: After training completes, click "Play Once" to see your newly trained AI in action! 🎉

## 🧠 The AI Algorithms Explained

### 🧬 **Cross-Entropy Method (CEM)**
Think of this as "evolution for AI strategies":
- Creates a population of different playing strategies  
- Tests them all on Tetris games
- Keeps the best performers and creates variations
- Gradually evolves better and better strategies
- **Best for**: Finding robust, consistent strategies

### 🎯 **REINFORCE with Baseline**  
Like a student learning from feedback:
- Makes moves and sees the results
- Gets rewarded for good moves, penalized for bad ones
- Gradually learns which positions are good or bad
- Uses a "baseline" to reduce learning noise
- **Best for**: Adaptive, flexible play styles

## 📊 Understanding the Statistics

- **Score**: Points earned (clearing lines gives more points)
- **Lines**: Total lines cleared during gameplay  
- **Steps**: Number of pieces placed
- **Episodes**: Number of complete games played
- **Best/Average**: Tracking performance across multiple games

## 🎨 Visual Features

- **Colorful Pieces**: Each Tetris block uses vibrant colors with gradient effects
- **Real-Time Updates**: Smooth gameplay with live statistics
- **Training Progress**: Watch AI learning with detailed logs
- **Activity Log**: See exactly what the AI is thinking and learning

## 🚀 Advanced Features

### Training Your Own Agents
When you click "Quick Train", the app:
1. Trains a new AI using your selected algorithm and parameters
2. Saves the trained model automatically  
3. **Immediately loads it for gameplay** - your next "Play Once" uses YOUR trained agent!
4. Shows detailed training progress in the Activity Log

### Comparing Algorithms
- Train with CEM, test with "Play Multiple"
- Train with REINFORCE, compare the results  
- See which approach works better for your preferences
- Each algorithm has different strengths and learning styles

## 🛠️ Technical Details
## 🛠️ Technical Details

### Architecture
- **Backend**: FastAPI with WebSocket support for real-time gameplay
- **Frontend**: Vanilla JavaScript with HTML5 Canvas for smooth graphics
- **AI Engine**: NumPy-based reinforcement learning with custom Tetris environment
- **Visualization**: Real-time pixel manipulation and colorful rendering

### AI Features  
- **State Representation**: Board analysis using holes, heights, bumpiness, and line potential
- **Action Space**: 40 possible moves per piece (10 columns × 4 rotations)
- **Policy Optimization**: Both evolutionary and gradient-based learning
- **Performance Tracking**: Comprehensive metrics and learning curves

## 🐛 Troubleshooting

### Training Button Not Working?
Make sure you started the server with training enabled:
```bash
TRAIN_ENABLED=true python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### No Colorful Pieces?
- Refresh your browser to load the latest JavaScript
- Check that the server is running properly
- Try different play modes (Stream, Play Once, Play Multiple)

### Training Takes Too Long?
- Start with smaller parameters: 3 generations, 20 population for CEM
- Use fewer episodes (50-100) for REINFORCE initially  
- Training progress shows in the Activity Log

## 🎓 Learning More

### For Beginners
1. Start with "Play Once" to see how the AI makes decisions
2. Try "Stream Agent" to watch continuous gameplay
3. Experiment with "Quick Train" using small parameters
4. Read the Activity Log to understand what's happening

### For Advanced Users
- Modify training parameters to see how they affect learning
- Compare CEM vs REINFORCE performance on different metrics
- Analyze the feature extraction in `rl/features.py`
- Explore the policy evaluation in `rl/afterstate.py`

## 📁 Project Structure

```
app/                    # Web application
├── main.py            # FastAPI server and WebSocket endpoints
├── static/            
│   ├── index.html     # Web interface
│   ├── app.js         # Frontend JavaScript with colorful rendering
│   └── styles.css     # Modern styling
└── config.py          # Configuration management

rl/                     # Reinforcement Learning core
├── tetris_env.py      # Tetris game environment
├── features.py        # Board state analysis
├── afterstate.py      # Move evaluation and selection  
├── cem_agent.py       # Cross-Entropy Method training
├── reinforce_agent.py # REINFORCE with baseline training
└── policy_store.py    # Save/load trained models

tests/                  # Comprehensive test suite
policies/              # Saved trained AI models
```

## 🚀 Deployment Options

### Local Development
```bash
# Basic mode
python -m uvicorn app.main:app --reload

# Full-featured mode  
TRAIN_ENABLED=true python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### Docker
```bash
docker build -t tetris-ai .
docker run -p 8000:8080 -e TRAIN_ENABLED=true tetris-ai
```

### Cloud Deployment
Ready for deployment on:
- Google Cloud Run
- AWS Lambda/Fargate  
- Azure Container Instances
- Any containerized hosting platform

## 🧪 Testing

Run the comprehensive test suite:
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=app --cov=rl --cov-report=html

# Run specific test categories
pytest tests/test_env.py      # Tetris environment tests
pytest tests/test_features.py # Feature extraction tests  
pytest tests/test_api.py      # Web API tests
```

## 🤝 Contributing

Want to improve the AI or add features?

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b amazing-feature`
3. **Make your changes**: Add new algorithms, improve the UI, enhance training
4. **Add tests**: Ensure your changes work properly
5. **Submit a pull request**: Share your improvements!

### Ideas for Contributions
- 🧠 New AI algorithms (PPO, A3C, Rainbow DQN)
- 🎨 Enhanced visualizations and animations
- 📊 Advanced statistics and performance metrics
- 🎮 Different game variants or piece sets
- 🔧 Performance optimizations
- 📱 Mobile-responsive interface

## 📄 License

MIT License - feel free to use this project for learning, teaching, or commercial applications!

## 🙏 Acknowledgments

- Tetris game mechanics inspired by the classic puzzle game
- Reinforcement learning algorithms based on academic research
- Web interface designed for educational accessibility
- Colorful rendering for enhanced user experience

---

## 🎮 Ready to Start?

1. **Clone the repository**
2. **Install dependencies**: `pip install -r requirements.txt`  
3. **Start with training enabled**: `TRAIN_ENABLED=true python -m uvicorn app.main:app --host 0.0.0.0 --port 8000`
4. **Open**: `http://localhost:8000`
5. **Click "Quick Train"** to create your first AI agent!
6. **Watch it play** with "Play Once" - that's YOUR trained AI! 🎉

**Have fun exploring artificial intelligence through the classic game of Tetris!** 🚀
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
