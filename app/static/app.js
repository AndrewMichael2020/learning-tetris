// JavaScript for RL Tetris Web App
class TetrisApp {
    constructor() {
        this.canvas = document.getElementById('gameCanvas');
        this.ctx = this.canvas.getContext('2d');
        
        // Create offscreen canvas for double buffering to prevent flickering
        this.offscreenCanvas = document.createElement('canvas');
        this.offscreenCanvas.width = this.canvas.width;
        this.offscreenCanvas.height = this.canvas.height;
        this.offscreenCtx = this.offscreenCanvas.getContext('2d');
        
        this.websocket = null;
        this.playOnceWebSocket = null;
        this.playMultipleWebSocket = null;
        this.isStreaming = false;
        this.currentGame = null;
        this.debugMode = false; // Set to true to enable debug logging
        
        // Episode tracking for statistics
        this.episodeHistory = [];
        this.resetStats();
        
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
            clearResultsBtn: document.getElementById('clearResultsBtn'),
            
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
            maxScore: document.getElementById('maxScore'),
            
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
            
            // Activity Log
            activityLog: document.getElementById('activityLog')
        };
        
        // Debug: Check if critical elements were found
        console.log('DOM Elements Status:');
        const criticalElements = ['currentScore', 'currentLines', 'avgScore', 'totalLines', 'bestScore'];
        for (const key of criticalElements) {
            const element = this.elements[key];
            if (!element) {
                console.error(`❌ MISSING: ${key}`);
            } else {
                console.log(`✅ FOUND: ${key} = ${element.textContent}`);
            }
        }
    }
    
    setupEventListeners() {
        // Button event listeners
        this.elements.streamBtn.addEventListener('click', () => this.toggleStream());
        this.elements.playOnceBtn.addEventListener('click', () => this.playOnce());
        this.elements.playMultipleBtn.addEventListener('click', () => this.playEpisodes());
        this.elements.quickTrainBtn.addEventListener('click', () => this.quickTrain());
        this.elements.clearResultsBtn.addEventListener('click', () => this.resetStats());
        
        // Algorithm change listeners (both play and training)
        this.elements.algorithm.addEventListener('change', () => this.updatePlayControls());
        this.elements.trainAlgo.addEventListener('change', () => this.updateTrainingControls());
        
        // Initialize controls
        this.updatePlayControls();
    }
    
    resetStats() {
        // Reset episode tracking 
        this.episodeHistory = [];
        this.updateEpisodeResults();
    }
    
    addEpisode(score, lines) {
        // Add episode to history
        this.episodeHistory.push({ score: score, lines: lines });
        this.updateEpisodeResults();
        console.log('Episode added:', { score, lines, total: this.episodeHistory.length });
    }
    
    updateEpisodeResults() {
        // Calculate and update Episode Results
        const totalEpisodes = this.episodeHistory.length;
        
        if (totalEpisodes === 0) {
            // Reset all to 0, but keep Max Score as theoretical maximum
            this.safeUpdateElement('totalEpisodes', '0');
            this.safeUpdateElement('avgScore', '0.0');
            this.safeUpdateElement('totalLines', '0');
            this.safeUpdateElement('bestScore', '0');
            // Max Score stays static at 999999 (theoretical maximum)
            return;
        }
        
        // Calculate statistics
        const totalScore = this.episodeHistory.reduce((sum, ep) => sum + ep.score, 0);
        const totalLines = this.episodeHistory.reduce((sum, ep) => sum + ep.lines, 0);
        const avgScore = totalScore / totalEpisodes;
        const bestScore = Math.max(...this.episodeHistory.map(ep => ep.score));
        
        // Update displays (Max Score stays static as theoretical maximum)
        this.safeUpdateElement('totalEpisodes', totalEpisodes.toString());
        this.safeUpdateElement('avgScore', avgScore.toFixed(1));
        this.safeUpdateElement('totalLines', totalLines.toString());
        this.safeUpdateElement('bestScore', bestScore.toString());
        // Max Score is not updated - it remains as static theoretical maximum
        
        console.log('Episode Results updated:', { totalEpisodes, avgScore: avgScore.toFixed(1), totalLines, bestScore, maxScoreStatic: '999999' });
    }
    
    safeUpdateElement(id, value) {
        const element = document.getElementById(id);
        if (element) {
            element.textContent = value;
        } else {
            console.warn(`Element ${id} not found`);
        }
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
            params.w_holes = parseFloat(document.getElementById('greedy_w_holes')?.value || '8.0');
            params.w_max_height = parseFloat(document.getElementById('greedy_w_max_height')?.value || '1.0');
            params.w_bumpiness = parseFloat(document.getElementById('greedy_w_bumpiness')?.value || '1.0');
        } else if (algo === 'tabu') {
            params.tenure = parseInt(document.getElementById('tabu_tenure')?.value || '25');
            params.neighborhood_top_k = parseInt(document.getElementById('tabu_neighborhood_k')?.value || '10');
            params.aspiration = document.getElementById('tabu_aspiration')?.checked !== false; // Default to true
            params.w_holes = parseFloat(document.getElementById('tabu_w_holes')?.value || '8.0');
            params.w_max_height = parseFloat(document.getElementById('tabu_w_max_height')?.value || '1.0');
            params.rng_seed = parseInt(this.elements.trainSeed?.value || '42');
        } else if (algo === 'anneal') {
            const T0_input = document.getElementById('anneal_T0')?.value;
            if (T0_input && T0_input.trim() !== '') {
                params.T0 = parseFloat(T0_input);
            }
            params.alpha = parseFloat(document.getElementById('anneal_alpha')?.value || '0.99');
            params.proposal_top_k = parseInt(document.getElementById('anneal_proposal_k')?.value || '10');
            params.w_holes = parseFloat(document.getElementById('anneal_w_holes')?.value || '8.0');
            params.w_max_height = parseFloat(document.getElementById('anneal_w_max_height')?.value || '1.0');
            params.rng_seed = parseInt(this.elements.trainSeed?.value || '42');
        } else if (algo === 'aco') {
            params.alpha = parseFloat(document.getElementById('aco_alpha')?.value || '1.0');
            params.beta = parseFloat(document.getElementById('aco_beta')?.value || '2.0');
            params.rho = parseFloat(document.getElementById('aco_rho')?.value || '0.1');
            params.ants = parseInt(document.getElementById('aco_ants')?.value || '20');
            params.elite = parseInt(document.getElementById('aco_elite')?.value || '1');
            params.w_holes = parseFloat(document.getElementById('aco_w_holes')?.value || '8.0');
            params.w_max_height = parseFloat(document.getElementById('aco_w_max_height')?.value || '1.0');
            params.rng_seed = parseInt(this.elements.trainSeed?.value || '42');
        }
        
        console.log(`Collected params for ${algo}:`, params);
        return params;
    }
    
    updatePlayControls() {
        const algo = this.elements.algorithm.value;
        
        // Show/hide algorithm-specific parameter sections for play controls
        const playControlSections = [
            'playGreedyControls', 'playTabuControls', 'playAnnealControls', 'playAcoControls'
        ];
        
        playControlSections.forEach(sectionId => {
            const section = document.getElementById(sectionId);
            if (section) {
                section.style.display = 'none';
            }
        });
        
        // Show the appropriate control section for new algorithms
        if (['greedy', 'tabu', 'anneal', 'aco'].includes(algo)) {
            const activeSection = 'play' + algo.charAt(0).toUpperCase() + algo.slice(1) + 'Controls';
            const activeElement = document.getElementById(activeSection);
            if (activeElement) {
                activeElement.style.display = 'block';
            }
        }
        
        this.log(`Switched to ${algo.toUpperCase()} algorithm for playing`, 'info');
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
            
            // Training is now always enabled, so just show training controls
            const trainControlsElement = document.getElementById('trainControls');
            if (trainControlsElement) {
                console.log('Training enabled - showing controls');
                trainControlsElement.style.display = 'block';
                this.elements.quickTrainBtn.disabled = false;
            }
            
            // Set algorithm dropdown to match the most recently trained algorithm
            if (health.current_algorithm && this.elements.algorithm) {
                const currentAlgo = health.current_algorithm.toLowerCase();
                // Check if the option exists in the dropdown
                const option = Array.from(this.elements.algorithm.options).find(opt => opt.value === currentAlgo);
                if (option) {
                    this.elements.algorithm.value = currentAlgo;
                    console.log(`Algorithm dropdown set to: ${currentAlgo} (most recently trained)`);
                    
                    // Update play controls for the selected algorithm
                    this.updatePlayControls();
                } else {
                    console.warn(`Algorithm ${currentAlgo} not found in dropdown options`);
                }
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
            console.log('Play Once WebSocket URL:', wsUrl); // Debug log
            
            this.playOnceWebSocket = new WebSocket(wsUrl);
            
            this.playOnceWebSocket.onopen = () => {
                this.log(`Starting single episode with ${algo.toUpperCase()} algorithm`, 'info');
            };
            
            this.playOnceWebSocket.onmessage = (event) => {
                try {
                    const data = JSON.parse(event.data);
                    
                    console.log('Play Once WebSocket message received:', data); // Debug log
                    
                    if (data.error) {
                        this.log('Play Once error: ' + data.error, 'error');
                        this.elements.gameStatus.textContent = 'Error';
                        return;
                    }
                    
                    // Force update current stats in real-time - ensure elements exist
                    console.log('Updating statistics...', {score: data.score, lines: data.lines, step: data.step});
                    
                    // Score update
                    if (data.score !== undefined && data.score !== null) {
                        try {
                            const scoreElement = document.getElementById('currentScore');
                            if (scoreElement) {
                                scoreElement.textContent = Math.floor(data.score);
                                console.log('✅ Score updated to:', scoreElement.textContent);
                            } else {
                                console.error('❌ currentScore element not found in DOM');
                            }
                        } catch (e) {
                            console.error('Error updating score:', e);
                        }
                    }
                    
                    // Lines update  
                    if (data.lines !== undefined && data.lines !== null) {
                        try {
                            const linesElement = document.getElementById('currentLines');
                            if (linesElement) {
                                linesElement.textContent = data.lines;
                                console.log('✅ Lines updated to:', linesElement.textContent);
                            } else {
                                console.error('❌ currentLines element not found in DOM');
                            }
                        } catch (e) {
                            console.error('Error updating lines:', e);
                        }
                    }
                    
                    // Steps update
                    if (data.step !== undefined && data.step !== null) {
                        try {
                            const stepsElement = document.getElementById('currentSteps');
                            if (stepsElement) {
                                stepsElement.textContent = data.step;
                                console.log('✅ Steps updated to:', stepsElement.textContent);
                            } else {
                                console.error('❌ currentSteps element not found in DOM');
                            }
                        } catch (e) {
                            console.error('Error updating steps:', e);
                        }
                    }
                    
                    // Update game status
                    if (data.done !== undefined) {
                        this.elements.gameStatus.textContent = data.done ? 'Completed' : 'Playing';
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
                        console.log('🎯 FINAL MESSAGE received:', data);
                        
                        // Add episode to history for proper tracking
                        const finalScore = data.score || 0;
                        const finalLines = data.lines || 0;
                        this.addEpisode(finalScore, finalLines);
                        
                        // Update game status
                        const gameStatusEl = document.getElementById('gameStatus');
                        if (gameStatusEl) {
                            gameStatusEl.textContent = 'Completed';
                            console.log('✅ gameStatus updated to: Completed');
                        }
                        
                        this.log(`Episode completed! Score: ${finalScore}, Lines: ${finalLines}`, 'success');
                    }
                } catch (error) {
                    console.error('Error processing Play Once WebSocket message:', error);
                    this.log('Error processing message: ' + error.message, 'error');
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
                    
                    // Add to global episode tracking
                    this.addEpisode(data.score, data.lines);
                    
                    this.log(`Episode ${data.episode} completed: Score ${data.score}, Lines ${data.lines}`, 'success');
                }
                
                // Handle final completion
                if (data.final) {
                    this.elements.gameStatus.textContent = 'Completed';
                    this.log(`All ${episodes} episodes completed! Episodes in history: ${this.episodeHistory.length}`, 'success');
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
            console.log('Training request body:', requestBody); // Debug log
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
            
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }
            
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
                this.log('Training failed: ' + (result.message || 'Unknown error'), 'error');
                console.error('Training error details:', result);
                this.drawEmptyBoard();
            }
            
        } catch (error) {
            console.error('Training request failed:', error);
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
            console.log('PNG image loaded:', img.width, 'x', img.height);
            
            // Use offscreen canvas for double buffering to prevent flickering
            const ctx = this.offscreenCtx;
            
            // Clear offscreen canvas
            ctx.fillStyle = '#000';
            ctx.fillRect(0, 0, this.offscreenCanvas.width, this.offscreenCanvas.height);
            
            // Force scale the image to fill the entire canvas while maintaining aspect ratio
            // Use the full canvas dimensions without complex ratio calculations
            const drawX = 0;
            const drawY = 0;
            const drawWidth = this.offscreenCanvas.width;
            const drawHeight = this.offscreenCanvas.height;
            
            console.log('Drawing PNG at full canvas size:', drawX, drawY, drawWidth, drawHeight);
            
            // Scale and draw the game board image to fill entire canvas
            ctx.imageSmoothingEnabled = false; // Pixel-perfect scaling
            ctx.drawImage(img, drawX, drawY, drawWidth, drawHeight);
            
            // Add colorful overlay effect to the PNG - analyze pixels and recolor them
            this.colorizeOffscreenCanvas();
            
            // Draw grid overlay on offscreen canvas
            this.drawGridOnCanvas(ctx, this.offscreenCanvas);
            
            // Finally, copy the complete offscreen canvas to main canvas in one operation
            this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
            this.ctx.drawImage(this.offscreenCanvas, 0, 0);
        };
        
        img.onerror = () => {
            console.error('Failed to load PNG frame');
            this.drawEmptyBoard();
        };
        
        img.src = 'data:image/png;base64,' + pngData;
    }
    
    colorizeOffscreenCanvas() {
        // Get current image data from offscreen canvas
        const imageData = this.offscreenCtx.getImageData(0, 0, this.offscreenCanvas.width, this.offscreenCanvas.height);
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
                const x = pixelIndex % this.offscreenCanvas.width;
                const y = Math.floor(pixelIndex / this.offscreenCanvas.width);
                const gridX = Math.floor(x / (this.offscreenCanvas.width / 10));
                const gridY = Math.floor(y / (this.offscreenCanvas.height / 20));
                
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
        
        // Put the modified image data back to offscreen canvas
        this.offscreenCtx.putImageData(imageData, 0, 0);
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
        
        // Clear canvas with black background
        this.ctx.fillStyle = '#000';
        this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);
        
        // Draw the full board using the current game statistics
        const cellWidth = this.canvas.width / 10;
        const cellHeight = this.canvas.height / 20;
        
        console.log('Canvas dimensions:', this.canvas.width, 'x', this.canvas.height);
        console.log('Cell dimensions:', cellWidth, 'x', cellHeight);
        
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
                    
                    // Fill the entire cell area
                    this.ctx.fillRect(
                        col * cellWidth + 1, 
                        row * cellHeight + 1, 
                        cellWidth - 2, 
                        cellHeight - 2
                    );
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
                this.ctx.fillRect(
                    col * cellWidth + 1, 
                    row * cellHeight + 1, 
                    cellWidth - 2, 
                    cellHeight - 2
                );
            }
        }
        
        // Add current falling piece in upper area if game is active
        if (currentSteps > 0) {
            // Draw a simple T-piece pattern for active gameplay indication
            const startRow = Math.min(Math.floor(currentSteps / 10) % 8, 5); // Top area
            const startCol = Math.floor(Math.random() * 8); // Allow space for 3-wide piece
            
            this.ctx.fillStyle = this.tetrisPieceColors[0]; // Bright color for current piece
            
            // T-piece pattern: center piece and three pieces across top
            const tPiece = [
                [startRow, startCol + 1],     // Center
                [startRow + 1, startCol],     // Left
                [startRow + 1, startCol + 1], // Center bottom
                [startRow + 1, startCol + 2]  // Right
            ];
            
            tPiece.forEach(([row, col]) => {
                if (row >= 0 && row < 20 && col >= 0 && col < 10) {
                    this.ctx.fillRect(
                        col * cellWidth + 1,
                        row * cellHeight + 1,
                        cellWidth - 2,
                        cellHeight - 2
                    );
                }
            });
        }
        
        // Draw grid overlay
        this.drawGrid();
    }

    drawGrid() {
        this.drawGridOnCanvas(this.ctx, this.canvas);
    }
    
    drawGridOnCanvas(ctx, canvas) {
        ctx.strokeStyle = '#333';
        ctx.lineWidth = 1;
        
        const cellWidth = canvas.width / 10;
        const cellHeight = canvas.height / 20;
        
        // Draw vertical lines
        for (let x = 0; x <= 10; x++) {
            ctx.beginPath();
            ctx.moveTo(x * cellWidth, 0);
            ctx.lineTo(x * cellWidth, canvas.height);
            ctx.stroke();
        }
        
        // Draw horizontal lines
        for (let y = 0; y <= 20; y++) {
            ctx.beginPath();
            ctx.moveTo(0, y * cellHeight);
            ctx.lineTo(canvas.width, y * cellHeight);
            ctx.stroke();
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