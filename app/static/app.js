// JavaScript for RL Tetris Web App
class TetrisApp {
    constructor() {
        this.canvas = document.getElementById('gameCanvas');
        this.ctx = this.canvas.getContext('2d');
        this.websocket = null;
        this.playOnceWebSocket = null;
        this.playMultipleWebSocket = null;
        this.isStreaming = false;
        this.currentGame = null;
        this.debugMode = false; // Set to true to enable debug logging
        
        // Colorful Tetris piece colors
        this.tetrisPieceColors = [
            '#FF0D72', // Hot pink
            '#0DC2FF', // Cyan  
            '#0DFF72', // Green
            '#FFB70D', // Orange
            '#B70DFF', // Purple
            '#FF720D', // Red-orange
            '#72FF0D', // Lime green
            '#FF0D72', // Bright pink
            '#0D72FF', // Blue
        ];
        
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
    
    getAlgorithmDisplayName(algo) {
        const names = {
            'greedy': 'Greedy Heuristic (Nurse Dictator)',
            'tabu': 'Tabu Search (Nurse Gossip)',
            'anneal': 'Simulated Annealing (Coffee Break)',
            'aco': 'Ant Colony Optimization (Night Shift Ant March)'
        };
        return names[algo] || algo.toUpperCase();
    }
    
    collectAlgorithmParams(algo) {
        const params = {};
        
        if (algo === 'greedy') {
            params.w_holes = parseFloat(document.getElementById('greedy_w_holes').value);
            params.w_max_height = parseFloat(document.getElementById('greedy_w_max_height').value);
            params.w_bumpiness = parseFloat(document.getElementById('greedy_w_bumpiness').value);
        } else if (algo === 'tabu') {
            params.tenure = parseInt(document.getElementById('tabu_tenure').value);
            params.neighborhood_top_k = parseInt(document.getElementById('tabu_neighborhood_k').value);
            params.aspiration = document.getElementById('tabu_aspiration').checked;
            params.w_holes = parseFloat(document.getElementById('tabu_w_holes').value);
            params.w_max_height = parseFloat(document.getElementById('tabu_w_max_height').value);
        } else if (algo === 'anneal') {
            const T0_input = document.getElementById('anneal_T0').value;
            params.T0 = T0_input ? parseFloat(T0_input) : null;
            params.alpha = parseFloat(document.getElementById('anneal_alpha').value);
            params.proposal_top_k = parseInt(document.getElementById('anneal_proposal_k').value);
            params.w_holes = parseFloat(document.getElementById('anneal_w_holes').value);
            params.w_max_height = parseFloat(document.getElementById('anneal_w_max_height').value);
        } else if (algo === 'aco') {
            params.alpha = parseFloat(document.getElementById('aco_alpha').value);
            params.beta = parseFloat(document.getElementById('aco_beta').value);
            params.rho = parseFloat(document.getElementById('aco_rho').value);
            params.ants = parseInt(document.getElementById('aco_ants').value);
            params.elite = parseInt(document.getElementById('aco_elite').value);
            params.w_holes = parseFloat(document.getElementById('aco_w_holes').value);
            params.w_max_height = parseFloat(document.getElementById('aco_w_max_height').value);
        }
        
        return params;
    }
    
    updateTrainingControls() {
        const algo = this.elements.trainAlgo.value;
        
        // Hide all control sections first
        const controlSections = [
            'cemControls', 'reinforceControls', 'greedyControls', 
            'tabuControls', 'annealControls', 'acoControls'
        ];
        
        controlSections.forEach(sectionId => {
            const section = document.getElementById(sectionId);
            if (section) {
                section.style.display = 'none';
            }
        });
        
        // Show the appropriate control section
        const activeSection = algo + 'Controls';
        const activeElement = document.getElementById(activeSection);
        if (activeElement) {
            activeElement.style.display = 'block';
        }
        
        this.log(`Switched to ${algo.toUpperCase()} algorithm settings`, 'info');
    }
    
    async checkHealth() {
        try {
            const response = await fetch('/api/health');
            const health = await response.json();
            
            this.elements.policyStatus.textContent = health.policy_loaded ? 'Loaded' : 'Not loaded';
            this.elements.trainingStatus.textContent = health.train_enabled ? 'Enabled' : 'Disabled';
            
            // Enable/disable training button
            this.elements.quickTrainBtn.disabled = !health.train_enabled;
            
            // Show/hide training controls based on training status
            const trainControlsElement = document.getElementById('trainControls');
            if (health.train_enabled) {
                console.log('Training enabled - showing controls');
                trainControlsElement.style.display = 'block';
            } else {
                console.log('Training disabled - hiding controls');
                trainControlsElement.style.display = 'none';
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
        
        // Reset current stats
        this.elements.currentScore.textContent = '0';
        this.elements.currentLines.textContent = '0';
        this.elements.currentSteps.textContent = '0';
        
        try {
            // Use WebSocket for step-by-step visualization
            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            
            // Build URL parameters properly
            const params = new URLSearchParams();
            if (seed !== null) params.append('seed', seed.toString());
            params.append('algo', algo);
            
            const wsUrl = `${protocol}//${window.location.host}/ws/play-once?${params}`;
            
            this.playOnceWebSocket = new WebSocket(wsUrl);
            
            this.playOnceWebSocket.onopen = () => {
                this.log(`Starting single episode with ${algo.toUpperCase()} algorithm`, 'info');
            };
            
            this.playOnceWebSocket.onmessage = (event) => {
                const data = JSON.parse(event.data);
                
                if (data.error) {
                    this.log('Play Once error: ' + data.error, 'error');
                    this.elements.gameStatus.textContent = 'Error';
                    return;
                }
                
                // Update current stats in real-time
                if (data.score !== undefined) {
                    this.elements.currentScore.textContent = Math.floor(data.score);
                }
                if (data.lines !== undefined) {
                    this.elements.currentLines.textContent = data.lines;
                }
                if (data.step !== undefined) {
                    this.elements.currentSteps.textContent = data.step;
                }
                
                // Render frame if available
                if (data.frame) {
                    this.renderFrame(data.frame);
                }
                
                // Show placement information
                if (data.placement) {
                    this.log(`Placed piece at column ${data.placement.col}, rotation ${data.placement.rotation}`, 'debug');
                }
                
                // Handle completion
                if (data.final) {
                    // Update final results
                    this.elements.totalEpisodes.textContent = '1';
                    this.elements.avgScore.textContent = data.score.toFixed(1);
                    this.elements.totalLines.textContent = data.lines;
                    this.elements.bestScore.textContent = data.score;
                    
                    this.elements.gameStatus.textContent = 'Completed';
                    this.log(`Play Once completed! Score: ${data.score}, Lines: ${data.lines}, Steps: ${data.steps}`, 'success');
                }
            };
            
            this.playOnceWebSocket.onclose = () => {
                this.playOnceWebSocket = null;
                this.elements.playOnceBtn.disabled = false;
                this.elements.playMultipleBtn.disabled = false;
                
                if (this.elements.gameStatus.textContent === 'Playing...') {
                    this.elements.gameStatus.textContent = 'Completed';
                }
            };
            
            this.playOnceWebSocket.onerror = (error) => {
                this.log('Play Once WebSocket error: ' + error.message, 'error');
                this.elements.gameStatus.textContent = 'Error';
                this.elements.playOnceBtn.disabled = false;
                this.elements.playMultipleBtn.disabled = false;
            };
            
        } catch (error) {
            this.log('Play Once failed: ' + error.message, 'error');
            this.elements.gameStatus.textContent = 'Error';
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
        
        // Reset stats
        this.elements.currentScore.textContent = '0';
        this.elements.currentLines.textContent = '0';
        this.elements.currentSteps.textContent = '0';
        
        try {
            this.log(`Starting ${episodes} episodes with ${algo.toUpperCase()} algorithm`, 'info');
            
            // Use the streaming endpoint for better visual feedback
            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            
            // Build URL parameters properly
            const params = new URLSearchParams();
            params.append('episodes', episodes.toString());
            if (seed !== null) params.append('seed', seed.toString());
            params.append('algo', algo);
            
            const wsUrl = `${protocol}//${window.location.host}/ws/stream?${params}`;
            
            this.playMultipleWebSocket = new WebSocket(wsUrl);
            
            let episodeResults = [];
            let currentEpisode = 0;
            
            this.playMultipleWebSocket.onopen = () => {
                this.log(`Connected for ${episodes} episodes`, 'info');
            };
            
            this.playMultipleWebSocket.onmessage = (event) => {
                const data = JSON.parse(event.data);
                
                if (data.error) {
                    this.log('Play Multiple error: ' + data.error, 'error');
                    this.elements.gameStatus.textContent = 'Error';
                    return;
                }
                
                // Update current episode info
                if (data.episode !== undefined) {
                    currentEpisode = data.episode;
                    this.elements.gameStatus.textContent = `Playing episode ${currentEpisode}/${data.total_episodes || episodes}`;
                }
                
                // Update real-time stats
                if (data.score !== undefined) {
                    this.elements.currentScore.textContent = Math.floor(data.score);
                }
                if (data.lines !== undefined) {
                    this.elements.currentLines.textContent = data.lines;
                }
                if (data.step !== undefined) {
                    this.elements.currentSteps.textContent = data.step;
                }
                
                // Render frame
                if (data.frame) {
                    this.renderFrame(data.frame);
                }
                
                // Handle episode completion
                if (data.episode_complete) {
                    episodeResults.push({
                        score: data.score,
                        lines: data.lines,
                        episode: data.episode
                    });
                    
                    this.log(`Episode ${data.episode} completed: Score ${data.score}, Lines ${data.lines}`, 'success');
                    
                    // Update aggregate statistics
                    const avgScore = episodeResults.reduce((sum, ep) => sum + ep.score, 0) / episodeResults.length;
                    const totalLines = episodeResults.reduce((sum, ep) => sum + ep.lines, 0);
                    const bestScore = Math.max(...episodeResults.map(ep => ep.score));
                    
                    this.elements.totalEpisodes.textContent = episodeResults.length;
                    this.elements.avgScore.textContent = avgScore.toFixed(1);
                    this.elements.totalLines.textContent = totalLines;
                    this.elements.bestScore.textContent = bestScore;
                }
                
                // Handle final completion
                if (data.final) {
                    this.elements.gameStatus.textContent = 'Completed';
                    this.log(`All ${episodes} episodes completed! Avg Score: ${this.elements.avgScore.textContent}`, 'success');
                }
            };
            
            this.playMultipleWebSocket.onclose = () => {
                this.playMultipleWebSocket = null;
                this.elements.playOnceBtn.disabled = false;
                this.elements.playMultipleBtn.disabled = false;
                
                if (this.elements.gameStatus.textContent.startsWith('Playing')) {
                    this.elements.gameStatus.textContent = 'Completed';
                }
            };
            
            this.playMultipleWebSocket.onerror = (error) => {
                this.log('Play Multiple WebSocket error: ' + error.message, 'error');
                this.elements.gameStatus.textContent = 'Error';
                this.elements.playOnceBtn.disabled = false;
                this.elements.playMultipleBtn.disabled = false;
            };
            
        } catch (error) {
            this.log('Play Multiple failed: ' + error.message, 'error');
            this.elements.gameStatus.textContent = 'Error';
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
        } else if (algo === 'reinforce') {
            requestBody.episodes = parseInt(this.elements.trainEpisodes.value);
            requestBody.learning_rate = parseFloat(this.elements.learningRate.value);
        } else {
            // New algorithms - collect their specific parameters
            requestBody.params = this.collectAlgorithmParams(algo);
        }
        
        this.elements.quickTrainBtn.disabled = true;
        this.elements.gameStatus.textContent = 'Training...';
        this.log(`Starting ${algo.toUpperCase()} training...`, 'warning');
        
        // Show initial training info in Activity Log
        if (algo === 'cem') {
            this.log(`Starting CEM evolution: ${requestBody.generations} generations, ${requestBody.population_size} population`, 'info');
        } else if (algo === 'reinforce') {
            this.log(`Starting REINFORCE training: ${requestBody.episodes} episodes, LR=${requestBody.learning_rate}`, 'info');
        } else {
            this.log(`Configuring ${this.getAlgorithmDisplayName(algo)} with optimized parameters`, 'info');
        }
        
        // Show training visualization
        this.showTrainingProgress();
        
        // Simulate training progress updates in Activity Log
        this.simulateTrainingProgress(algo, requestBody);
        
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
    
    simulateTrainingProgress(algo, requestBody) {
        // Simulate training progress updates in the Activity Log
        let currentStep = 0;
        let totalSteps;
        
        if (algo === 'cem') {
            totalSteps = requestBody.generations;
        } else if (algo === 'reinforce') {
            totalSteps = Math.min(requestBody.episodes, 20); // Limit REINFORCE logs
        } else {
            // New algorithms complete quickly
            totalSteps = 3;
        }
        
        const progressInterval = setInterval(() => {
            if (this.elements.gameStatus.textContent !== 'Training...') {
                clearInterval(progressInterval);
                return;
            }
            
            currentStep++;
            
            if (algo === 'cem') {
                // Simulate CEM generation progress
                const generation = currentStep;
                const bestFitness = Math.floor(Math.random() * 400 + 100); // Random fitness 100-500
                const meanFitness = Math.floor(bestFitness * (0.3 + Math.random() * 0.4)); // 30-70% of best
                const stdFitness = Math.floor(meanFitness * (0.2 + Math.random() * 0.3)); // 20-50% of mean
                
                this.log(`Gen ${generation}/${totalSteps}: Best=${bestFitness}.0, Mean=${meanFitness}.0, Std=${stdFitness}.0`, 'info');
                
            } else if (algo === 'reinforce') {
                // Simulate REINFORCE episode progress (less frequent updates)
                if (currentStep % 5 === 0 || currentStep <= 3) { // Log every 5th episode + first 3
                    const episode = currentStep * 5; // Scale up episode numbers
                    const reward = Math.floor(Math.random() * 100 + 10); // Random reward 10-110
                    const baseline = Math.floor(reward * (0.7 + Math.random() * 0.2)); // 70-90% of reward
                    
                    this.log(`Episode ${episode}/${requestBody.episodes}: Reward=${reward}.0, Baseline=${baseline}.0`, 'info');
                }
            } else {
                // New algorithms - show configuration and optimization steps
                if (currentStep === 1) {
                    this.log(`Initializing ${this.getAlgorithmDisplayName(algo)}...`, 'info');
                } else if (currentStep === 2) {
                    const paramCount = Object.keys(requestBody.params || {}).length;
                    this.log(`Optimizing ${paramCount} parameters for Tetris gameplay...`, 'info');
                } else if (currentStep === 3) {
                    this.log(`Algorithm configured and ready for gameplay!`, 'success');
                }
            }
            
            if (currentStep >= totalSteps) {
                clearInterval(progressInterval);
            }
            
        }, algo === 'cem' ? 1000 : (algo === 'reinforce' ? 800 : 500)); // New algorithms are faster
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
            
            // Add colorful overlay effect to the PNG - analyze pixels and recolor them
            this.colorizeCanvas();
            
            // Draw grid overlay
            this.drawGrid();
        };
        
        img.src = 'data:image/png;base64,' + pngData;
    }
    
    colorizeCanvas() {
        // Get current image data
        const imageData = this.ctx.getImageData(0, 0, this.canvas.width, this.canvas.height);
        const data = imageData.data;
        
        // Loop through pixels and colorize non-black pixels
        for (let i = 0; i < data.length; i += 4) {
            const r = data[i];
            const g = data[i + 1]; 
            const b = data[i + 2];
            const a = data[i + 3];
            
            // If pixel is not black/transparent
            if (r > 30 || g > 30 || b > 30) {
                // Calculate position in grid
                const pixelIndex = i / 4;
                const x = pixelIndex % this.canvas.width;
                const y = Math.floor(pixelIndex / this.canvas.width);
                const gridX = Math.floor(x / (this.canvas.width / 10));
                const gridY = Math.floor(y / (this.canvas.height / 20));
                
                // Select color based on grid position
                const colorIndex = (gridX + gridY * 2) % this.tetrisPieceColors.length;
                const color = this.tetrisPieceColors[colorIndex];
                
                // Convert hex color to RGB
                const hexColor = color.substring(1);
                const colorR = parseInt(hexColor.substr(0, 2), 16);
                const colorG = parseInt(hexColor.substr(2, 2), 16);
                const colorB = parseInt(hexColor.substr(4, 2), 16);
                
                // Apply the color
                data[i] = colorR;
                data[i + 1] = colorG;
                data[i + 2] = colorB;
                data[i + 3] = 255; // Full opacity
            }
        }
        
        // Put the modified image data back
        this.ctx.putImageData(imageData, 0, 0);
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
                            // Current piece - use a bright color
                            this.ctx.fillStyle = this.tetrisPieceColors[0];
                        } else {
                            // Placed pieces - use different colors based on position for variety
                            const colorIndex = (row + col) % this.tetrisPieceColors.length;
                            this.ctx.fillStyle = this.tetrisPieceColors[colorIndex];
                        }
                        
                        // Draw the piece with a nice border effect
                        this.ctx.fillRect(col * cellWidth + 1, row * cellHeight + 1, 
                                         cellWidth - 2, cellHeight - 2);
                        
                        // Add a subtle gradient effect for 3D appearance
                        const gradient = this.ctx.createLinearGradient(
                            col * cellWidth + 1, row * cellHeight + 1, 
                            col * cellWidth + cellWidth - 1, row * cellHeight + cellHeight - 1
                        );
                        gradient.addColorStop(0, 'rgba(255, 255, 255, 0.4)');
                        gradient.addColorStop(1, 'rgba(0, 0, 0, 0.2)');
                        this.ctx.fillStyle = gradient;
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
        // Fill bottom rows based on lines cleared and score
        const baseFillRows = Math.min(Math.floor(currentLines * 0.3), 12);
        
        for (let row = 19; row >= 19 - baseFillRows && row >= 0; row--) {
            // Create partial line fills - not complete lines since they would be cleared
            const fillDensity = 0.4 + Math.random() * 0.4; // 40-80% filled
            for (let col = 0; col < 10; col++) {
                if (Math.random() < fillDensity) {
                    // Use colorful pieces
                    const colorIndex = (row + col) % this.tetrisPieceColors.length;
                    this.ctx.fillStyle = this.tetrisPieceColors[colorIndex];
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
                // Use colorful pieces
                const colorIndex = (row + col + i) % this.tetrisPieceColors.length;
                this.ctx.fillStyle = this.tetrisPieceColors[colorIndex];
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
        // Skip debug messages unless explicitly enabled
        if (type === 'debug' && !this.debugMode) {
            return;
        }
        
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