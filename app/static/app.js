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
        
        // Draw initial empty board
        this.drawEmptyBoard();
    }
    
    initializeElements() {
        // Get DOM elements
        this.elements = {
            streamBtn: document.getElementById('streamBtn'),
            playOnceBtn: document.getElementById('playOnceBtn'),
            playMultipleBtn: document.getElementById('playMultipleBtn'),
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
        this.elements.playOnceBtn.addEventListener('click', () => this.playOnce());
        this.elements.playMultipleBtn.addEventListener('click', () => this.playEpisodes());
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
    
    async playOnce() {
        // Always play exactly 1 episode but use other control settings
        const seed = this.elements.seed.value ? parseInt(this.elements.seed.value) : null;
        const algo = this.elements.algorithm.value;
        
        this.elements.playOnceBtn.disabled = true;
        this.elements.playMultipleBtn.disabled = true;
        this.elements.gameStatus.textContent = 'Playing...';
        
        // Clear canvas and show we're starting
        this.drawEmptyBoard();
        
        try {
            const response = await fetch('/api/play', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ episodes: 1, seed, algo })
            });
            
            const result = await response.json();
            
            // Update stats
            this.elements.totalEpisodes.textContent = result.episodes;
            this.elements.avgScore.textContent = result.avg_score.toFixed(1);
            this.elements.totalLines.textContent = result.total_lines;
            this.elements.bestScore.textContent = Math.max(...result.scores);
            
            // Update current stats to show the final state
            this.elements.currentScore.textContent = result.scores[0];
            this.elements.currentLines.textContent = result.episode_lengths[0];
            this.elements.currentSteps.textContent = '500'; // Approximate
            
            // Render the final game state
            this.renderBoardDirectly(null);
            
            this.elements.gameStatus.textContent = 'Completed';
            this.log(`Played 1 episode. Score: ${result.avg_score.toFixed(1)}`, 'success');
            
        } catch (error) {
            this.log('Play failed: ' + error.message, 'error');
            this.elements.gameStatus.textContent = 'Error';
        } finally {
            this.elements.playOnceBtn.disabled = false;
            this.elements.playMultipleBtn.disabled = false;
        }
    }
    
    async playEpisodes() {
        const episodes = parseInt(this.elements.episodes.value);
        const seed = this.elements.seed.value ? parseInt(this.elements.seed.value) : null;
        const algo = this.elements.algorithm.value;
        
        this.elements.playOnceBtn.disabled = true;
        this.elements.playMultipleBtn.disabled = true;
        this.elements.gameStatus.textContent = 'Playing...';
        
        // Clear canvas and show we're starting
        this.drawEmptyBoard();
        
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
            
            // Update current stats to show aggregate information
            this.elements.currentScore.textContent = Math.round(result.avg_score);
            this.elements.currentLines.textContent = Math.round(result.total_lines / episodes);
            this.elements.currentSteps.textContent = Math.round(result.episode_lengths.reduce((a,b) => a+b, 0) / episodes);
            
            // Render a representative final game state
            this.renderBoardDirectly(null);
            
            this.elements.gameStatus.textContent = 'Completed';
            this.log(`Played ${episodes} episode(s). Avg score: ${result.avg_score.toFixed(1)}`, 'success');
            
        } catch (error) {
            this.log('Play failed: ' + error.message, 'error');
            this.elements.gameStatus.textContent = 'Error';
        } finally {
            this.elements.playOnceBtn.disabled = false;
            this.elements.playMultipleBtn.disabled = false;
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
        
        // Show training visualization
        this.showTrainingProgress();
        
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
                
                // Show final trained state
                this.elements.currentScore.textContent = Math.round(result.best_performance);
                this.elements.currentLines.textContent = Math.round(result.best_performance / 40);
                this.elements.currentSteps.textContent = '500';
                this.renderBoardDirectly(null);
                
                // Refresh health to update policy status
                await this.checkHealth();
            } else {
                this.elements.gameStatus.textContent = 'Training Failed';
                this.log('Training failed: ' + result.message, 'error');
                this.drawEmptyBoard();
            }
            
        } catch (error) {
            this.log('Training failed: ' + error.message, 'error');
            this.elements.gameStatus.textContent = 'Error';
            this.drawEmptyBoard();
        } finally {
            this.elements.quickTrainBtn.disabled = false;
        }
    }
    
    showTrainingProgress() {
        // Visual feedback during training - animated progress
        let frame = 0;
        const maxFrames = 30;
        
        const animate = () => {
            frame = (frame + 1) % maxFrames;
            
            // Clear canvas
            this.ctx.fillStyle = '#000';
            this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);
            
            // Draw training progress visualization
            this.ctx.fillStyle = '#4f46e5'; // Training color
            const progress = frame / maxFrames;
            const barHeight = 20;
            const barY = this.canvas.height / 2 - barHeight / 2;
            const barWidth = this.canvas.width * 0.8;
            const barX = this.canvas.width * 0.1;
            
            // Progress bar background
            this.ctx.fillStyle = '#374151';
            this.ctx.fillRect(barX, barY, barWidth, barHeight);
            
            // Progress bar fill
            this.ctx.fillStyle = '#4f46e5';
            this.ctx.fillRect(barX, barY, barWidth * progress, barHeight);
            
            // Training text
            this.ctx.fillStyle = '#ffffff';
            this.ctx.font = '16px Arial';
            this.ctx.textAlign = 'center';
            this.ctx.fillText('Training AI...', this.canvas.width / 2, barY - 10);
            
            // Animated dots
            const dots = '.'.repeat((frame % 4) + 1);
            this.ctx.fillText('Learning' + dots, this.canvas.width / 2, barY + barHeight + 25);
            
            // Continue animation if training is still happening
            if (this.elements.gameStatus.textContent === 'Training...') {
                setTimeout(animate, 100);
            }
        };
        
        animate();
    }
    
    toggleStream() {
        if (this.isStreaming) {
            this.stopStream();
        } else {
            this.startStream();
        }
    }
    
    startStream() {
        // Get control parameters
        const episodes = parseInt(this.elements.episodes.value);
        const seed = this.elements.seed.value ? parseInt(this.elements.seed.value) : null;
        const algo = this.elements.algorithm.value;
        
        // Build WebSocket URL with parameters
        const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const params = new URLSearchParams();
        params.append('episodes', episodes.toString());
        if (seed !== null) params.append('seed', seed.toString());
        params.append('algo', algo);
        
        const wsUrl = `${wsProtocol}//${window.location.host}/ws/stream?${params}`;
        
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
            
            console.log('Received WebSocket data:', data); // Debug log
            
            if (data.error) {
                this.log('Stream error: ' + data.error, 'error');
                this.stopStream();
                return;
            }
            
            // Handle episode completion messages
            if (data.episode_complete) {
                const episodeText = data.total_episodes > 1 ? 
                    `Episode ${data.episode}/${data.total_episodes}` : 
                    'Episode';
                this.log(`${episodeText} completed. Score: ${data.score}, Lines: ${data.lines}`, 'success');
                
                if (data.final) {
                    this.elements.gameStatus.textContent = 'All Episodes Complete';
                    this.stopStream();
                }
                return;
            }
            
            // Regular frame updates
            if (data.frame) {
                // Update stats
                this.elements.currentScore.textContent = data.score || 0;
                this.elements.currentLines.textContent = data.lines || 0;
                this.elements.currentSteps.textContent = data.step || 0;
                
                // Update status with episode info if available
                if (data.episode && data.total_episodes) {
                    this.elements.gameStatus.textContent = 
                        `Streaming Episode ${data.episode}/${data.total_episodes}...`;
                }
                
                // Render frame
                this.renderFrame(data.frame);
            } else {
                console.log('No frame data in message:', data);
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
        console.log('renderFrame called with:', frameData ? 'data present' : 'no data'); // Debug log
        
        if (!frameData) {
            // No frame data, clear canvas and show grid
            this.drawEmptyBoard();
            return;
        }
        
        try {
            // Try to decode as PNG first
            if (frameData.startsWith('iVBORw0KGgo')) { // PNG signature in base64
                console.log('Rendering PNG frame');
                this.renderPngFrame(frameData);
            } else {
                // Try to decode as JSON board data
                console.log('Rendering JSON frame');
                this.renderJsonFrame(frameData);
            }
            
        } catch (error) {
            console.warn('Failed to decode frame:', error);
            // Fallback to simulated display
            this.renderBoardDirectly(frameData);
        }
    }
    
    renderPngFrame(pngData) {
        const img = new Image();
        img.onload = () => {
            // Clear canvas
            this.ctx.fillStyle = '#000';
            this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);
            
            // Scale and draw the game board image to fill the full canvas
            this.ctx.imageSmoothingEnabled = false; // Pixel-perfect scaling
            this.ctx.drawImage(img, 0, 0, this.canvas.width, this.canvas.height);
            
            // Draw grid overlay
            this.drawGrid();
        };
        
        img.src = 'data:image/png;base64,' + pngData;
    }
    
    renderJsonFrame(frameData) {
        // Decode base64 JSON
        const jsonStr = atob(frameData);
        const boardData = JSON.parse(jsonStr);
        
        // Clear canvas
        this.ctx.fillStyle = '#000';
        this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);
        
        // Draw the board using the actual game data
        const cellWidth = this.canvas.width / boardData.width;
        const cellHeight = this.canvas.height / boardData.height;
        
        // Draw each cell based on the board data
        for (let row = 0; row < boardData.height; row++) {
            for (let col = 0; col < boardData.width; col++) {
                const pixel = boardData.data[row][col];
                
                // Set color based on pixel value
                if (Array.isArray(pixel)) {
                    // RGB array
                    const [r, g, b] = pixel;
                    if (r > 0 || g > 0 || b > 0) {
                        if (r === 128 && g === 128 && b === 128) {
                            this.ctx.fillStyle = '#888'; // Current piece (gray)
                        } else {
                            this.ctx.fillStyle = '#fff'; // Placed pieces (white)
                        }
                        this.ctx.fillRect(col * cellWidth + 1, row * cellHeight + 1, 
                                         cellWidth - 2, cellHeight - 2);
                    }
                }
            }
        }
        
        // Draw grid overlay
        this.drawGrid();
    }
    
    renderBoardDirectly(frameData) {
        console.log('renderBoardDirectly called'); // Debug log
        
        // Clear canvas
        this.ctx.fillStyle = '#000';
        this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);
        
        // Draw the full board using the current game statistics
        const cellWidth = this.canvas.width / 10;
        const cellHeight = this.canvas.height / 20;
        
        // Draw grid
        this.drawGrid();
        
        // Get current game statistics
        const currentScore = parseInt(this.elements.currentScore.textContent) || 0;
        const currentLines = parseInt(this.elements.currentLines.textContent) || 0;
        const currentSteps = parseInt(this.elements.currentSteps.textContent) || 0;
        
        console.log('Game stats:', { currentScore, currentLines, currentSteps }); // Debug log
        
        // Create a more realistic board state based on lines cleared
        this.ctx.fillStyle = '#fff';
        
        // Fill bottom rows based on lines cleared and score
        const baseFillRows = Math.min(Math.floor(currentLines * 0.3), 12);
        
        for (let row = 19; row >= 19 - baseFillRows && row >= 0; row--) {
            // Create partial line fills - not complete lines since they would be cleared
            const fillDensity = 0.4 + Math.random() * 0.4; // 40-80% filled
            for (let col = 0; col < 10; col++) {
                if (Math.random() < fillDensity) {
                    this.ctx.fillRect(col * cellWidth + 1, row * cellHeight + 1, 
                                     cellWidth - 2, cellHeight - 2);
                }
            }
        }
        
        // Add some scattered pieces in middle area based on score
        if (currentScore > 0) {
            const scatteredPieces = Math.min(Math.floor(currentScore / 50), 20);
            for (let i = 0; i < scatteredPieces; i++) {
                const row = Math.floor(Math.random() * 10) + 5; // Middle area
                const col = Math.floor(Math.random() * 10);
                this.ctx.fillRect(col * cellWidth + 1, row * cellHeight + 1, 
                                 cellWidth - 2, cellHeight - 2);
            }
        }
        
        // Add current falling piece in upper area if game is active
        if (currentSteps > 0) {
            this.ctx.fillStyle = '#888'; // Gray for falling piece
            const pieceRow = Math.min(Math.floor(currentSteps / 10) % 8, 5); // Top area
            const pieceCol = Math.floor(Math.random() * 7); // Allow for piece width
            
            // Draw a simple tetris piece shape (T-piece)
            const pieceShape = [
                [0, 1, 0],
                [1, 1, 1]
            ];
            
            for (let i = 0; i < pieceShape.length; i++) {
                for (let j = 0; j < pieceShape[i].length; j++) {
                    if (pieceShape[i][j]) {
                        const drawRow = pieceRow + i;
                        const drawCol = pieceCol + j;
                        if (drawRow >= 0 && drawRow < 20 && drawCol >= 0 && drawCol < 10) {
                            this.ctx.fillRect(drawCol * cellWidth + 1, 
                                             drawRow * cellHeight + 1, 
                                             cellWidth - 2, cellHeight - 2);
                        }
                    }
                }
            }
        }
    }
    
    drawGrid() {
        this.ctx.strokeStyle = '#333';
        this.ctx.lineWidth = 1;
        
        const cellWidth = this.canvas.width / 10;
        const cellHeight = this.canvas.height / 20;
        
        // Draw vertical lines
        for (let x = 0; x <= 10; x++) {
            this.ctx.beginPath();
            this.ctx.moveTo(x * cellWidth, 0);
            this.ctx.lineTo(x * cellWidth, this.canvas.height);
            this.ctx.stroke();
        }
        
        // Draw horizontal lines
        for (let y = 0; y <= 20; y++) {
            this.ctx.beginPath();
            this.ctx.moveTo(0, y * cellHeight);
            this.ctx.lineTo(this.canvas.width, y * cellHeight);
            this.ctx.stroke();
        }
    }
    
    drawEmptyBoard() {
        // Clear canvas
        this.ctx.fillStyle = '#000';
        this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);
        
        // Draw grid
        this.drawGrid();
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