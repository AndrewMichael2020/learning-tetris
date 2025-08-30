// JavaScript for RL Tetris Web App
class TetrisApp {
    constructor() {
        this.canvas = document.getElementById('gameCanvas');
        this.ctx = this.canvas.getContext('2d');
        this.websocket = null;
        this.isStreaming = false;
        this.currentGame = null;
        
        this.initializeElements();
        this.setupEventListeners();
        this.checkHealth();
        this.updateTrainingControls();
    }
    
    initializeElements() {
        // Get DOM elements
        this.elements = {
            streamBtn: document.getElementById('streamBtn'),
            playOnceBtn: document.getElementById('playOnceBtn'),
            quickTrainBtn: document.getElementById('quickTrainBtn'),
            
            // Stats
            currentScore: document.getElementById('currentScore'),
            currentLines: document.getElementById('currentLines'),
            currentSteps: document.getElementById('currentSteps'),
            gameStatus: document.getElementById('gameStatus'),
            
            // Results
            totalEpisodes: document.getElementById('totalEpisodes'),
            avgScore: document.getElementById('avgScore'),
            totalLines: document.getElementById('totalLines'),
            bestScore: document.getElementById('bestScore'),
            
            // Controls
            episodes: document.getElementById('episodes'),
            seed: document.getElementById('seed'),
            algorithm: document.getElementById('algorithm'),
            
            // Training
            trainAlgo: document.getElementById('trainAlgo'),
            trainSeed: document.getElementById('trainSeed'),
            generations: document.getElementById('generations'),
            populationSize: document.getElementById('populationSize'),
            trainEpisodes: document.getElementById('trainEpisodes'),
            learningRate: document.getElementById('learningRate'),
            
            // Status
            policyStatus: document.getElementById('policyStatus'),
            trainingStatus: document.getElementById('trainingStatus'),
            activityLog: document.getElementById('activityLog')
        };
    }
    
    setupEventListeners() {
        // Button event listeners
        this.elements.streamBtn.addEventListener('click', () => this.toggleStream());
        this.elements.playOnceBtn.addEventListener('click', () => this.playEpisodes());
        this.elements.quickTrainBtn.addEventListener('click', () => this.quickTrain());
        
        // Training algorithm change
        this.elements.trainAlgo.addEventListener('change', () => this.updateTrainingControls());
    }
    
    updateTrainingControls() {
        const algo = this.elements.trainAlgo.value;
        const cemControls = document.getElementById('cemControls');
        const reinforceControls = document.getElementById('reinforceControls');
        
        if (algo === 'cem') {
            cemControls.style.display = 'block';
            reinforceControls.style.display = 'none';
        } else {
            cemControls.style.display = 'none';
            reinforceControls.style.display = 'block';
        }
    }
    
    async checkHealth() {
        try {
            const response = await fetch('/api/health');
            const health = await response.json();
            
            this.elements.policyStatus.textContent = health.policy_loaded ? 'Loaded' : 'Not loaded';
            this.elements.trainingStatus.textContent = health.train_enabled ? 'Enabled' : 'Disabled';
            
            // Enable/disable training button
            this.elements.quickTrainBtn.disabled = !health.train_enabled;
            
            if (health.train_enabled) {
                document.getElementById('trainControls').style.display = 'block';
            }
            
            this.log('System health check completed', 'success');
            
        } catch (error) {
            this.log('Health check failed: ' + error.message, 'error');
        }
    }
    
    async playEpisodes() {
        const episodes = parseInt(this.elements.episodes.value);
        const seed = this.elements.seed.value ? parseInt(this.elements.seed.value) : null;
        const algo = this.elements.algorithm.value;
        
        this.elements.playOnceBtn.disabled = true;
        this.elements.gameStatus.textContent = 'Playing...';
        
        try {
            const response = await fetch('/api/play', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ episodes, seed, algo })
            });
            
            const result = await response.json();
            
            // Update stats
            this.elements.totalEpisodes.textContent = result.episodes;
            this.elements.avgScore.textContent = result.avg_score.toFixed(1);
            this.elements.totalLines.textContent = result.total_lines;
            this.elements.bestScore.textContent = Math.max(...result.scores);
            
            this.elements.gameStatus.textContent = 'Completed';
            this.log(`Played ${episodes} episode(s). Avg score: ${result.avg_score.toFixed(1)}`, 'success');
            
        } catch (error) {
            this.log('Play failed: ' + error.message, 'error');
            this.elements.gameStatus.textContent = 'Error';
        } finally {
            this.elements.playOnceBtn.disabled = false;
        }
    }
    
    async quickTrain() {
        const algo = this.elements.trainAlgo.value;
        const seed = parseInt(this.elements.trainSeed.value);
        
        let requestBody = { algo, seed };
        
        if (algo === 'cem') {
            requestBody.generations = parseInt(this.elements.generations.value);
            requestBody.population_size = parseInt(this.elements.populationSize.value);
            requestBody.episodes_per_candidate = 2; // Fixed for quick training
        } else {
            requestBody.episodes = parseInt(this.elements.trainEpisodes.value);
            requestBody.learning_rate = parseFloat(this.elements.learningRate.value);
        }
        
        this.elements.quickTrainBtn.disabled = true;
        this.elements.gameStatus.textContent = 'Training...';
        this.log(`Starting ${algo.toUpperCase()} training...`, 'warning');
        
        try {
            const response = await fetch('/api/train', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(requestBody)
            });
            
            const result = await response.json();
            
            if (result.success) {
                this.elements.gameStatus.textContent = 'Training Complete';
                this.log(`Training completed! Best performance: ${result.best_performance.toFixed(1)} (${result.training_time.toFixed(1)}s)`, 'success');
                
                // Refresh health to update policy status
                await this.checkHealth();
            } else {
                this.elements.gameStatus.textContent = 'Training Failed';
                this.log('Training failed: ' + result.message, 'error');
            }
            
        } catch (error) {
            this.log('Training failed: ' + error.message, 'error');
            this.elements.gameStatus.textContent = 'Error';
        } finally {
            this.elements.quickTrainBtn.disabled = false;
        }
    }
    
    toggleStream() {
        if (this.isStreaming) {
            this.stopStream();
        } else {
            this.startStream();
        }
    }
    
    startStream() {
        const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${wsProtocol}//${window.location.host}/ws/stream`;
        
        this.websocket = new WebSocket(wsUrl);
        
        this.websocket.onopen = () => {
            this.isStreaming = true;
            this.elements.streamBtn.textContent = 'Stop Stream';
            this.elements.streamBtn.classList.remove('btn-primary');
            this.elements.streamBtn.classList.add('btn-warning');
            this.elements.gameStatus.textContent = 'Streaming...';
            this.log('WebSocket stream started', 'success');
        };
        
        this.websocket.onmessage = (event) => {
            const data = JSON.parse(event.data);
            
            if (data.error) {
                this.log('Stream error: ' + data.error, 'error');
                this.stopStream();
                return;
            }
            
            // Update stats
            this.elements.currentScore.textContent = data.score;
            this.elements.currentLines.textContent = data.lines;
            this.elements.currentSteps.textContent = data.step;
            
            // Render frame
            this.renderFrame(data.frame);
            
            if (data.done) {
                this.elements.gameStatus.textContent = 'Stream Complete';
                this.log(`Stream episode completed. Score: ${data.score}, Lines: ${data.lines}`, 'success');
                this.stopStream();
            }
        };
        
        this.websocket.onclose = () => {
            this.stopStream();
        };
        
        this.websocket.onerror = (error) => {
            this.log('WebSocket error: ' + error.message, 'error');
            this.stopStream();
        };
    }
    
    stopStream() {
        if (this.websocket) {
            this.websocket.close();
            this.websocket = null;
        }
        
        this.isStreaming = false;
        this.elements.streamBtn.textContent = 'Stream Agent';
        this.elements.streamBtn.classList.remove('btn-warning');
        this.elements.streamBtn.classList.add('btn-primary');
        
        if (this.elements.gameStatus.textContent === 'Streaming...') {
            this.elements.gameStatus.textContent = 'Ready';
        }
    }
    
    renderFrame(frameData) {
        // For now, just clear canvas with a pattern
        // In real implementation, decode base64 PNG and draw it
        this.ctx.fillStyle = '#000';
        this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);
        
        // Draw a simple grid pattern to simulate game board
        this.ctx.strokeStyle = '#333';
        this.ctx.lineWidth = 1;
        
        const cellWidth = this.canvas.width / 10;
        const cellHeight = this.canvas.height / 20;
        
        for (let x = 0; x <= 10; x++) {
            this.ctx.beginPath();
            this.ctx.moveTo(x * cellWidth, 0);
            this.ctx.lineTo(x * cellWidth, this.canvas.height);
            this.ctx.stroke();
        }
        
        for (let y = 0; y <= 20; y++) {
            this.ctx.beginPath();
            this.ctx.moveTo(0, y * cellHeight);
            this.ctx.lineTo(this.canvas.width, y * cellHeight);
            this.ctx.stroke();
        }
        
        // Draw some random filled cells to simulate pieces
        this.ctx.fillStyle = '#fff';
        const currentScore = parseInt(this.elements.currentScore.textContent);
        const numCells = Math.min(currentScore / 10, 50); // Simulate pieces based on score
        
        for (let i = 0; i < numCells; i++) {
            const x = Math.floor(Math.random() * 10);
            const y = Math.floor(15 + Math.random() * 5); // Bottom area
            this.ctx.fillRect(x * cellWidth + 1, y * cellHeight + 1, 
                             cellWidth - 2, cellHeight - 2);
        }
    }
    
    log(message, type = 'info') {
        const logEntry = document.createElement('div');
        logEntry.className = `log-entry ${type}`;
        logEntry.textContent = `[${new Date().toLocaleTimeString()}] ${message}`;
        
        this.elements.activityLog.appendChild(logEntry);
        this.elements.activityLog.scrollTop = this.elements.activityLog.scrollHeight;
        
        // Keep only last 100 log entries
        while (this.elements.activityLog.children.length > 100) {
            this.elements.activityLog.removeChild(this.elements.activityLog.firstChild);
        }
    }
}

// Initialize app when page loads
document.addEventListener('DOMContentLoaded', () => {
    new TetrisApp();
});